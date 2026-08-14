#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash-tp4}"
HEAD_HOST="${HEAD_HOST:-node0}"
IFS=',' read -r -a WORKERS <<< "${WORKER_HOSTS:-node1,node2,node3}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
for host in "$HEAD_HOST" "${WORKERS[@]}"; do
  echo "stopping $PROJECT_NAME on $host"
  if [[ "$host" == "$(hostname)" || "$host" == "$HEAD_HOST" ]]; then
    docker rm -f "${PROJECT_NAME}-vllm-dspark-1" >/dev/null 2>&1 || true
  else
    ssh "${SSH_OPTS[@]}" "$host" "docker rm -f ${PROJECT_NAME}-vllm-dspark-1 >/dev/null 2>&1 || true"
  fi
done
echo "TP-4 stopped"
