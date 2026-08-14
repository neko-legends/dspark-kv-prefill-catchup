# DeepSeek V4 Flash 0731 Abliterated · TP=4 · 4× DGX Spark

Uncensored DeepSeek V4 Flash 0731 NVFP4, tensor-parallel across four NVIDIA
DGX Sparks (GB10) on a switched CX-7 RoCE fabric.

This is a **Neko Legends** serving recipe. The checkpoint is the abliterated
NVFP4 32-32 build, not the stock official weights. Numbers below were measured
on that checkpoint.

**Record single-stream decode: 113.8 tok/s** (vLLM engine 10 s average).
Formal warmed C1 median: **103.4 tok/s**.

## Result

Abliterated NVFP4, thinking off, temperature 0, concurrency 1, 2048 completion
tokens, cluster idle. Server metric is
`Δ generation_tokens_total / Δ request_decode_time_seconds_sum`.
Client wall is `(completion_tokens − 1) / (t_last − t_first)` on streamed text.

| | tok/s |
|---| ---: |
| Record (engine 10 s window) | **113.8** |
| Formal C1 median (n=7 clean) | **103.4** |
| Formal C1 mean | 98.3 |
| Formal C1 min / max | 84.3 / 107.5 |
| Same cluster, previous k=5 shape, C1 2048 median | 85.7 |

Clean trials only. Windows with a second in-flight request were dropped.

Live boot after this recipe:

```text
GPU KV cache size: 4,793,645 tokens
Maximum concurrency for 393,216 tokens per request: 12.19x
```

A short-prompt C1 number and a long-session agent number are different
measurements. Deep cached trunks decode slower. Publish both if you quote one.

## Topology

Four GB10 nodes, one vLLM world, TP=4.

- Head serves the OpenAI-compatible API (`0.0.0.0:18888` here).
- Three workers are headless ranks.
- NCCL / Gloo / TP sockets stay on the CX-7 data NIC, never the tailnet.
- Management plane (SSH, dashboard) may use LAN or Tailscale.
- Fabric here: switched L2 RoCE on `enp1s0f1np1`, `192.168.2.0/24`, MTU 9000,
  HCA `rocep1s0f1`. A MikroTik CRS812 aggregates the QSFP-DD links.

Set hostnames, IPs, and the model host path in `.env` (not committed).

## Serving shape

Image: `dspark-vllm-gx10:0.1.1-flashinfer-0.6.15` (Anemll 0.1.1 / vLLM 0.25.2).

```text
--tensor-parallel-size 4 --nnodes 4
--kv-cache-dtype nvfp4_ds_mla --block-size 256
--max-model-len 393216
--max-num-seqs 12
--max-num-batched-tokens 8264
--max-cudagraph-capture-size 96          # seqs × (k + 1)
--gpu-memory-utilization 0.85
--speculative-config k=7, draft_sample_method=probabilistic
--compilation-config {"cudagraph_mode":"FULL_DECODE_ONLY"}
--override-generation-config {"temperature":0.0}
--default-chat-template-kwargs {"thinking":false}
--moe-backend flashinfer_b12x
--enable-prefix-caching --async-scheduling --enable-chunked-prefill
```

`--ulimit nofile=1048576` on every rank. TP=4 opens enough NCCL sockets that
the image default of 1024 dies with `Too many open files`.

Launch from the head: worker ranks first, then the API rank.
See `scripts/start-dspark-tp4.sh`.

## Why these knobs

| knob | we run | why |
|---|---|---|
| `num_speculative_tokens` | 7 | This image’s kernels are shaped for dspark7. k=5 left ~18 tok/s on the table. |
| `draft_sample_method` | probabilistic | Matches the target distribution. Greedy collapses acceptance off temp 0. |
| `max_cudagraph_capture_size` | `seqs × (k+1)` = 96 | A copied `36` truncates to 32 and dumps larger batches into eager. |
| `max_num_batched_tokens` | 8264 | vLLM subtracts `(k−1)×seqs` from the prefill budget and warns below 8192. `8192` raw lands under it. |
| `gpu_memory_utilization` | 0.85 | 0.80 wastes ~7 GiB. 0.90 does not boot on this weight split. |
| `max_model_len` | 393216 | KV pool GiB is almost flat from 327k–1M. The ceiling is a blast-radius cap, not extra speed. |
| `cudagraph_mode` | FULL_DECODE_ONLY | One graph set. No measured cost. |
| omitted `temperature` | forced 0.0 | `--generation-config vllm` otherwise defaults omitted temp to 1.0 and wrecks MTP accept. |
| thinking | off | On this checkpoint, thinking-on C1 was ~65 vs ~84–103 thinking-off. |

Raising `max_model_len` to 1M does **not** grow the KV pool in GiB and does
**not** raise C1 decode. Prefill of a true 500k–1M prompt is many minutes.
Cap the *client* if you care about TTFT.

## Landmines

1. **One IPv4 on the NCCL NIC.** A leftover switch-management address as the
   primary IP makes NCCL advertise that address. Workers cannot reach it and
   hang at `ncclCommInitRank`. Keep the fabric IPv4 primary.
2. **`ulimit -n` must be 1M inside the container.** `bash -lc` in the image
   drops nofile to 1024 unless you set it in compose *and* `ulimit` in the
   entrypoint.
3. **Do not `netplan apply` an old point-to-point mesh file** after moving to
   a switch. Persist the switched `/24` or the next reboot restores dead pair
   subnets.
4. **Quote decode with prompt length and warmup.** A 10 s engine average
   during a 400k prefill is not a decode record. Warm the graphs (several
   full-length generations) before publishing C1.

## Reproduce the C1 number

Cluster idle. No other clients. Thinking off. Temperature 0.

```bash
python3 scripts/bench-decode.py \
  --base-url http://HEAD:18888/v1 \
  --model deepseek-v4-flash-0731-ablit-32-32 \
  --max-tokens 2048 \
  --warmup 8 \
  --n 10
```

Report the Prometheus decode-only rate *and* the client stream wall rate.
Drop any trial where `request_success_total` increases by more than 1.

Raw trial lists: [`results/c1-decode-2026-08-14.json`](results/c1-decode-2026-08-14.json).

## What this is not

- Not official (non-abliterated) 0731 numbers.
- Not a 1M-prompt throughput claim. Nobody here has decoded *at* 1M.
- Not aggregate multi-stream throughput. C4/C12 is a different measurement.
- Not a license to ship prompts. Weights stay on the cluster.

## Layout

```text
.env.example                 fabric + serving knobs (copy to .env)
scripts/start-dspark-tp4.sh  worker-first launch
scripts/stop-dspark-tp4.sh
scripts/status-dspark-tp4.sh
scripts/bench-decode.py      C1 streaming + Prometheus decode rate
results/                     measured artifacts
```
