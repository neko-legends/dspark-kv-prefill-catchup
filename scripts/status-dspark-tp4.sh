#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
[[ -f "$ENV_FILE" ]] && set -a && source "$ENV_FILE" && set +a
PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash-tp4}"
HEAD_HOST="${HEAD_HOST:-node0}"
IFS=',' read -r -a WORKERS <<< "${WORKER_HOSTS:-node1,node2,node3}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10)
echo "=== containers ==="
for host in "$HEAD_HOST" "${WORKERS[@]}"; do
  if [[ "$host" == "$(hostname)" || "$host" == "$HEAD_HOST" ]]; then
    line="$(docker ps -a --filter name=${PROJECT_NAME}-vllm-dspark-1 --format '{{.Names}} {{.Status}}' 2>/dev/null || true)"
  else
    line="$(ssh "${SSH_OPTS[@]}" "$host" "docker ps -a --filter name=${PROJECT_NAME}-vllm-dspark-1 --format '{{.Names}} {{.Status}}'" 2>/dev/null || true)"
  fi
  printf '%-12s %s\n' "$host" "${line:-down}"
done
echo
echo "=== API ==="
curl -fsS --max-time 5 "http://127.0.0.1:${VLLM_PORT:-18888}/v1/models" && echo || echo "API not ready"
