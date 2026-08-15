# Diagnosis: TP4 C1 decode 33.5 tok/s (2026-08-15 bench) — config regression, not topology

**Verdict: config issue. RESOLVED same day** — record recipe restored on rail B at
~15:20 PDT; formal C1 median 101.93 tok/s (n=10 clean, max 109.16). Full resolution
data: `results/c1-decode-2026-08-15-restored.json`.

The TP4 world benchmarked at 04:00 and relaunched at 11:30 on 2026-08-15 is **not the
stack that set the 103.4 tok/s C1 record on 2026-08-14**. Speculative decoding is OFF
in the running container. 33.5 tok/s is raw autoregressive TP4 decode. The earlier
commit verdict "TP4 topology (not fabric) is the C1 bottleneck" was measured on this
broken stack and is retracted — see below.

## Evidence (live inspection, forge, container `deepseek-v4-flash-tp4-vllm-dspark-1`)

Running container vs README recipe (the 103.4 median / 113.8 record stack):

| knob | record stack (08-14) | running stack (08-15) |
|---|---|---|
| image | `dspark-vllm-gx10:0.1.1-flashinfer-0.6.15` (Anemll, vLLM 0.25.2) | `aidendle94/sparkrun-vllm-ds4-gb10:production-ready` (vLLM v0.21.1rc1 fork) |
| speculative decoding | k=7, `draft_sample_method=probabilistic`, custom `dspark_proposer` | **NONE — `speculative_config=None` in engine log; no spec counters in /metrics** |
| checkpoint | local `/models/abliterated-nvfp4` (32-32, fp8 e4m3 ue8m0, `num_nextn_predict_layers: 1`) | HF `prem-research/DeepSeek-V4-Flash-0731-abliterated` (quant `deepseek_v4_fp8`, MTP acceptance never validated) |
| KV cache dtype | `nvfp4_ds_mla` | `fp8` |
| generation temp | forced 0.0 via `--override-generation-config` | not forced (accept collapses off temp 0) |
| MoE backend | `--moe-backend flashinfer_b12x` | env `VLLM_USE_B12X_MOE=1` + `VLLM_B12X_W4A16_FORCE_BLOCKS_MAX_M=16` |
| async scheduling | on | absent |
| cudagraphs | `FULL_DECODE_ONLY`, capture 96 (= seqs × (k+1)) | `FULL_AND_PIECEWISE`, capture [1,2,4,8,16] |
| max-num-seqs / batched-tokens / mem-util | 12 / 8264 / 0.85 | 8 / 8192 / 0.80 |
| served name | `deepseek-v4-flash-0731-ablit-32-32` | `deepseek-v4-flash-abliterated` |

Root cause of the silent spec loss: the compose contract passes `MTP_NUM_TOKENS` as an
**env var** (`docker-compose.dspark-tp4.yml:113`). The Anemll image entrypoint consumes
it and builds `--speculative-config`. The sparkrun image entrypoint
(`/usr/local/bin/dsv4-vllm-entrypoint`) is a plain `exec vllm "$@"` — it ignores the env
contract entirely. Image was swapped; flags were not. Spec vanished with no error.

Timeline: `~/dspark-2x-0731-abliterated-nvfp4/docker-compose.dspark-tp4.yml` has backups
`.bak-b12x-2349`, `.bak-fp8test-2029`, `.bak-20260815-032737` — overnight backend/fp8
experiments, then the 03:30 container start on the sparkrun image. The 11:30 rail-B
relaunch reused the same config, which is why the rail A/B A/B showed no difference:
**both rails were measured on the no-spec stack.**

## Why TP2 "beats" TP4 in the bench JSON

TP2 0731 baseline (44.16 c1_256 / 67.7 soak_512 / 1576 prefill) vs TP4 08-15 (33.5 /
~950): TP4 raw decode has strictly more per-step allreduce than TP2 raw decode, so
*without speculative decoding* TP2 > TP4 is expected arithmetic, not a topology
indictment. With k=7 spec on the record stack, TP4 measured 84–107 tok/s clean trials
(`results/c1-decode-2026-08-14.json`). TP4-with-spec >> TP2.

## Fabric / switch

Rail A vs rail B A/B (this repo, 2026-08-15): C1/C4/prefill identical within noise.
At 100+ tok/s the per-step comm is ~1 MB and ~2 ms of latency on the current rails.
A new switch fixes rail-A RoCE stalls (PFC/lossless) — do it for fabric health, but
**it will not move serving tok/s.** Do not let switch work mask or confound the config
fix: fix config first, re-bench, then touch the fabric.

## Fast invariant check (run after every launch)

```bash
curl -fsS http://forge:18888/metrics | grep -c spec_decode   # 0  => spec OFF, stack is broken
docker logs <container> 2>&1 | grep speculative_config        # None => spec OFF
```

## Plan

- **P0 — contract guard.** Make spec non-optional: either launch only via
  `scripts/start-dspark-tp4.sh` + repo `.env`, or add the two checks above to
  `scripts/status-dspark-tp4.sh` and fail the boot gate. Record image digest in the
  bench JSON from now on.
- **P1 — restore known-good.** Relaunch the README recipe (Anemll
  0.1.1-flashinfer-0.6.15, local 32-32, k=7, nvfp4_ds_mla, temp 0 override, thinking
  off, flashinfer_b12x, capture 96, async-scheduling, util 0.85). Verify spec counters.
  Re-run `scripts/bench-decode.py --max-tokens 2048 --warmup 8 --n 10`. Expect 85–103.
- **P2 — checkpoint A/B.** If `prem-research/...-abliterated` is wanted, run it on the
  *good* image with identical flags and compare MTP acceptance
  (`vllm:spec_decode_num_accepted_tokens_total / num_draft_tokens_total`). Collapsed
  acceptance => abliteration damaged the NextN head => keep 32-32 or re-abliterate with
  the MTP module excluded.
- **P3 — measure Eva's actual target.** C1 decode at 5k and 10k prompt depth (long-prompt
  variant of bench-decode), publish short + deep numbers. nvfp4 KV halves KV bytes vs
  the fp8 KV the broken stack ran — this is exactly the knob that matters at 10k.
- **P4 — prefill re-baseline.** The TP2 1576 tok/s came from a different stack; re-measure
  prefill on the restored TP4 recipe before tuning. Then look at
  `--max-num-batched-tokens`, chunked-prefill sizing, flashinfer autotune. Cold prefill
  is what the catch-up sidecar exists to hide; decode is the metric that matters.
- **P5 — switch last.** Swap/retune the MikroTik for rail-A PFC after P1–P3 numbers are
  banked, one change at a time.

## Resolution (2026-08-15 ~15:20–16:00 PDT)

- P1 done: record compose + record env on rail B (`~/dspark-2x-0731-abliterated-nvfp4/.env.dspark.tp4.railb-200g`).
  Boot gates green: dspark k=7 spec loaded, nvfp4_ds_mla KV (5.55M tokens), capture 1..96,
  batched tokens 16384, util 0.85. Formal C1 n=10: **median 101.93, mean 102.48, max 109.16**.
- P0 done: `scripts/status-dspark-tp4.sh` serving-shape gate (spec counters + served name).
- Twins question (Eva): resolved — f0/f1 are NOT two halves of one 200G port. devlink maps
  PF0→physical port 0 (dark, `No cable`), PF1→physical port 1 (the cage, 200G). f1 alone IS
  the full port. NCCL_IB_HCA stays single-device per rail; the true 2x upgrade after the
  switch swap is dual-RAIL (rocep1s0f1 + roceP2p1s0f1). See results JSON for GID/perftest notes.
- Depth curve (512-token probes): code task 79–84 @5k, **92.8 @10k**; prose task 52–56 @5k.
  Depth costs ~15% step time; the big variable is MTP acceptance (code 4.6–4.9/step,
  repetitive prose 2.1–2.4/step). 100 tok/s average on code-heavy 5–10k agent work is
  nearly there; k sweep (8–10) is the next lever. Prose-heavy summaries will sit 50–65.
- C4 aggregate 182.2 (was 92.4 broken; TP2 baseline 93.15).
- Sidecar fix: eva-kv-catchup was warming stale served name `deepseek-v4-flash-abliterated`;
  updated to `deepseek-v4-flash-0731-ablit-32-32` and restarted.
- Bench hygiene: eva-kv-catchup + eva-hum must be paused (or trials dropped) during formal
  benches — the sidecar's own transcript warmups count as foreign traffic.
