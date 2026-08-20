# SGLang vs vLLang TP4 — overnight attempt 2026-08-20

## Verdict
Inconclusive / vLLM retained. SGLang served correctly on all 4 GB10s but could
not run a valid performance comparison: the DSpark speculative path is blocked
by a dsv4 kernel constraint in the current dev image. Rolled back per mission
constraints; fabric verified healthy on vLLM (canonical bench env) at ~04:45.

## What worked (verified)
- Image: lmsysorg/sglang:dev-cu13-inkling-dspark (arm64, 2026-07-30) — includes
  deepseek_v4_dspark model + DSPARK spec algorithm + dsv4 attention backend.
- TP4 multi-node boot on rail B with workers-first launch, dist-init-addr
  192.168.10.1:26000 (bare host:port — SGLang's NetworkAddress.parse does NOT
  strip tcp://; a scheme produces tcp://tcp://host:port → "port number missing").
- Weights: 48 shards loaded in ~5.5 min/node, ~40GB/node.
- KV capacity: 7.5–8.1M tokens (vs vLLM 5.5M) at mem_fraction 0.85.
- Correct output on the exact endpoint contract (:18888, same model id,
  max_model_len 1M). Math probe passed.

## Root-cause fixes discovered along the way
1. **NCCL_IB_GID_INDEX=3 is stale on ember/flame** — their RoCE v2 GID moved to
   index 5 after interface re-enumeration (empty GID at 3 → "unhandled system
   error" during ncclCommInitRank). Fix: NCCL_IB_GID_INDEX=auto (each node picks
   its own valid RoCE v2 GID). **vLLM's .env still hardcodes 3 — it worked only
   because ember/flame booted with it populated; consider switching the vLLM env
   to auto too.**
2. Container needs --gpus all (obvious, but the vLLM compose had it and my first
   script didn't) and MASTER_ADDR/MASTER_PORT env for SGLang worker rendezvous.

## What failed
1. **CUDA graphs + first long generation**: all 4 workers hit the 300s scheduler
   watchdog simultaneously during the first 2048-token generation (8x512-token
   warmups passed). With --disable-cuda-graph: stable but ~3 tok/s plain decode
   (no spec decode = no comparison; vLLM's 102 tok/s includes DSpark k=7).
2. **DSPARK spec decode**: draft runner loads (gamma=5, Markov head, from the
   same checkpoint), but CUDA graph capture dies on a dsv4 kernel guard:
   `Check failed: num_tokens > 64 (60 vs. 64): Decode (num_tokens <= 64) must go
   through sparse_mla_sm120_decode_dsv4` — spec verify shapes (bs x 6) below the
   64-token threshold trip it regardless of --cuda-graph-bs-decode filtering.
   This is an upstream image bug (or requires a flag we didn't find: possibly
   the sparse_mla decode path selection for spec-verify).

## Retry hooks
- Launch script preserved: scripts/launch-sglang-tp4.sh (worker-first, GID auto,
  spec args as last attempted).
- Watch SGLang releases for a fixed dsv4 spec-verify decode path
  (sparse_mla_sm120_decode_dsv4 selection), then relaunch and bench C1/C4 +
  shared-prefix (radix on by default — the RadixAttention win never got measured).
- The aarch64 pip route (sglang 0.5.17 + sgl-kernel 0.3.21 + torch cu130) is a
  dead end: sgl-kernel wheel is cu12-built (libnvrtc.so.12 / c10 ABI mismatch).
  Use the docker image.

## Post-mission corrections (2026-08-20 morning, Jun-reviewed)
1. **GID "latent fragility" retracted** — vLLM's start-dspark-tp4.sh already
   resolves NCCL_IB_GID_INDEX per host at launch (resolve_gid scans each node's
   GID table for the RoCE v2 entry matching its rail IP). Running containers
   verified: forge/anvil=3, ember/flame=5 — correct per-node despite the drift.
   No env change needed; the script's IP-pinned discovery is stricter than
   NCCL_IB_GID_INDEX=auto.
2. **Real fix applied**: the start script's DEFAULT env file was the stale
   `.env.dspark.tp4` (wrong image aidendle94/sparkrun, wrong model ref) — the
   exact trap that cost a failed rollback attempt during the mission. Default
   changed to the canonical `.env.dspark.tp4.railb-200g.bench`; verified via
   compose config interpolation (correct image + model path). Explicit
   ENV_FILE= overrides still win.

## NCCL dual-rail busbw test (4-node allreduce, torch, per-node GID resolution, 2026-08-20 ~19:15)
Fabric briefly down for the test (GPU memory), restored + gate passed after.

| config | 8MB busbw | 64MB busbw | 256MB busbw |
|---|---|---|---|
| rail B only (current serving) | 7.4 GB/s | 11.1 GB/s | 10.3 GB/s |
| rail A only | 7.3 GB/s | 13.5 GB/s | — |
| **dual-rail (A+B)** | 7.2 GB/s | **23.1 GB/s** | 14.9 GB/s |

**Verdict**: dual-rail delivers ~2.1× busbw at 64MB (23.1 vs 11.1 GB/s) — the
interconnect upgrade is real. Rail A is fully healthy under concurrent 4-node
load (new switch fixed the PFC issue). 256MB degraded on both configs (likely
perftest/QP tuning, not the rail).

**Blocker for serving adoption**: vLLM compose hardcodes single-rail env
(NCCL_IB_HCA=roceP2p1s0f1, NCCL_SOCKET_IFNAME=enP2p1s0f1np1). Switching to
dual-rail needs: both HCAs + both IFNAMEs + per-node GID resolution handling
(start script resolves ONE gid per host; dual-rail needs valid GID on BOTH
devices — worked with NCCL_IB_GID_INDEX=auto in the test, and auto is safe
now that the ulimit issue was the real failure cause).

Also learned: NCCL test containers NEED --ulimit memlock=-1 (pinning) — the
"unhandled system error"/proxy Connect failures were missing ulimits, not GIDs.
