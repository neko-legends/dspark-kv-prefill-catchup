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
