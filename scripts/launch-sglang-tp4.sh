#!/bin/bash
# SGLang TP4 launcher — run ON THE TARGET NODE: ./launch-sglang-node.sh <node_rank 0-3>
set -eu
RANK=${1:?usage: launch-sglang-node.sh <rank>}
MODEL_SRC=/home/jun/models/keys-deepseek-v4-flash-ga-0731-dspark-abliterated-32-32
IMG=lmsysorg/sglang:dev-cu13-inkling-dspark

docker rm -f sglang-tp4 2>/dev/null || true
docker run -d --name sglang-tp4 \
  --network host --ipc host --gpus all --shm-size 64g \
  --device /dev/infiniband \
  --restart no \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v ${MODEL_SRC}:/models/abliterated-nvfp4:ro \
  -e NCCL_CROSS_NIC=1 -e NCCL_IB_GID_INDEX=auto -e NCCL_NET=IB -e NCCL_NVLS_ENABLE=0 \
  -e TP_SOCKET_IFNAME=enP2p1s0f1np1 -e NCCL_IB_DISABLE=0 -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IGNORE_CPU_AFFINITY=1 -e NCCL_IB_HCA=roceP2p1s0f1 -e NCCL_CUMEM_ENABLE=0 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET -e NCCL_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e NCCL_IB_ADDR_RANGE=192.168.10.0/24 -e GLOO_SOCKET_IFNAME=enP2p1s0f1np1 \
  -e NCCL_DEBUG=INFO -e MASTER_ADDR=192.168.10.1 -e MASTER_PORT=26000 \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -e SGLANG_RAGGED_VERIFY_MODE=static \
  ${IMG} python3 -m sglang.launch_server \
    --model-path /models/abliterated-nvfp4 \
    --served-model-name deepseek-v4-flash-0731-ablit-32-32 \
    --trust-remote-code \
    --host 0.0.0.0 --port 18888 \
    --tp 4 --nnodes 4 --node-rank ${RANK} \
    --dist-init-addr 192.168.10.1:26000 --dist-timeout 3600 \
    --attention-backend dsv4 \
    --kv-cache-dtype fp8_e4m3 \
    --mem-fraction-static 0.85 \
    --context-length 1048576 \
    --cuda-graph-bs-decode 12 16 24 32 40 48 64 96 \
    --max-running-requests 12 \
    --stream-interval 8 --enable-metrics --watchdog-timeout 600 \
    --speculative-algorithm DSPARK --speculative-draft-model-path /models/abliterated-nvfp4 --speculative-num-steps 5 --speculative-num-draft-tokens 6
echo "launched rank ${RANK}"
