# Fabric: CX-7 RoCE between the four Sparks

Everything we learned the hard way about wiring four DGX Sparks into one
TP=4 world over a MikroTik CRS812. Read this *before* benching — most
"TP4 is slow" reports are fabric configuration, not the model.

## Topology and port anatomy

Each node has two CX-7 PCI devices, each with two PCI functions:

| netdev | RDMA device | devlink physical port | reality |
|---|---|---|---|
| `enp1s0f0np0` | `rocep1s0f0` | port 0 | **dark** — no carrier, nothing to give NCCL |
| `enp1s0f1np1` | `rocep1s0f1` | port 1 | rail A cage, 200G |
| `enP2p1s0f0np0` | `roceP2p1s0f0` | port 0 | **dark** — same |
| `enP2p1s0f1np1` | `roceP2p1s0f1` | port 1 | rail B cage, 200G |

The `f0` functions are the ASIC's *second physical port*, unconnected on this
board (`ethtool` says "No cable" at L1). They are **not** lane-halves of the
cabled port. One cable already links at `200000Mb/s` (`ethtool`); the
~104 Gbps you may measure is a single-QP artifact, not the port. The real
multi-HCA upgrade is **dual-rail** (`rocep1s0f1` + `roceP2p1s0f1` = 400G),
not dual-function.

## Addressing rules (each one cost us hours)

1. **Exactly one IPv4 per fabric interface, persisted in netplan.** A second
   address (old point-to-point mesh /24, a switch-management address) on the
   NCCL iface (a) hangs `ncclCommInitRank` when NCCL advertises the
   unreachable one, and (b) quietly taxed per-step routing ~30% of C1 decode
   for us (103.4 → 136.25 median on an otherwise identical config after
   cleanup). Runtime `ip addr del` is not enough — stale
   `/etc/netplan/40-*.yaml` mesh files resurrect addresses on every reboot.
   Rename them `.disabled`.
2. **MTU 9000 end to end — node *and* switch port.** One node at 1500 on a
   rail cost real bandwidth (48.5 vs 68 Gbps -P4 until fixed). Verify with
   `ibv_devinfo` (`active_mtu: 4096`), not just `ip link`.
3. **GID indices renumber whenever interface addresses change.** They
   differed per node (5/7/7/3) before cleanup, uniform (3) after. Never
   hardcode `-x 3` in perftest or `NCCL_IB_GID_INDEX` — resolve dynamically
   by matching the iface IP, like `scripts/start-dspark-tp4.sh` does.

## RoCE on a lossy switch (the PFC saga)

Our NICs run **PFC disabled on all priorities** (`mlnx_qos -i <iface>`) —
lossy RoCE relying on DCQCN. On the MikroTik path, single-QP
`ib_write_bw` at large sizes *stalls entirely* while NCCL serving works fine
on the same rail at the same time (verified twice, on both rails).

So:

- **Do not benchmark the fabric with single-QP perftest and panic.** Use
  `-q 8` for throughput, `ib_send_lat` for latency (2.5 µs RTT here), and
  judge the fabric by NCCL-level behavior: decode tok/s and prefill tok/s.
- If you want true lossless RoCE, PFC must be enabled on **both** ends
  together — switch port (RouterOS PFC/priority config) *and* NIC
  (`sudo mlnx_qos -i <iface> --pfc 0,0,0,1,0,0,0,0` style). Half-configured
  lossless is worse than lossy.
- LLDP on the CRS812 flooded/misreported neighbors for us; don't trust it
  alone to prove the physical map — verify with per-node ethtool/link and
  measured flows.

## Rail choice: measured, not guessed

We A/B'd the full serving stack on rail A vs rail B (2026-08-15): C1, C4,
and prefill identical within noise. At 100+ tok/s C1 the per-step collective
traffic is ~1 MB and ~2 ms — far under either rail's capability. Pick one
healthy rail, pin everything to it (`NCCL_IB_HCA`, `NCCL_SOCKET_IFNAME`,
`TP_SOCKET_IFNAME`, `GLOO_SOCKET_IFNAME`, `NCCL_IB_ADDR_RANGE`), and keep the
second rail as the future dual-rail upgrade.

## GPU clock (adjacent, belongs here because benches lie without it)

Unlocked DVFS dips to ~1970 MHz under sustained prefill and boosts ~2500 in
decode — decode benches are clock-insensitive, prefill benches are not.
`nvidia-smi -lgc 0,2200` on every node holds ~2171 sustained: prefill stays
fast and reproducible, decode is unchanged (not clock-bound), and it *saves*
power — same throughput as a 2400 lock at ~21% fewer watts. Persist it:
`scripts/spark-gpu-clock-lock.service`.

## Sanity checklist before publishing any bench number

1. `ibv_devinfo`: serving rail `PORT_ACTIVE`, `active_mtu: 4096`.
2. `ip -4 addr show <iface>`: exactly one inet line.
3. `nvidia-smi --query-gpu=clocks.sm`: ~2171 under load (lock applied).
4. `/metrics`: `vllm:spec_decode_num_*` counters exist (else spec is OFF and
   every decode number is ~3x low).
5. Cluster idle: `vllm:num_requests_running` == 0 before and between trials
   (watch for sidecars/monitors; drop contaminated trials).
