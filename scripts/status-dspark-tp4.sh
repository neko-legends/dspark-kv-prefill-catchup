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

echo
echo "=== serving-shape gate ==="
# 2026-08-15 regression: an image swap silently dropped speculative decoding
# (env MTP contract ignored by the new entrypoint) and C1 fell 103 -> 33.5.
# If spec counters are absent from /metrics, the stack is NOT the record recipe.
metrics="$(curl -fsS --max-time 8 "http://127.0.0.1:${VLLM_PORT:-18888}/metrics" 2>/dev/null || true)"
if [[ -z "$metrics" ]]; then
  echo "FAIL: /metrics unreachable"
elif ! grep -q '^vllm:spec_decode_num_drafts_total' <<<"$metrics"; then
  echo "FAIL: no spec-decode counters -- speculative decoding is OFF (wrong image/entrypoint?)"
else
  drafts=$(grep '^vllm:spec_decode_num_drafts_total' <<<"$metrics" | awk '{print $2}')
  accepted=$(grep '^vllm:spec_decode_num_accepted_tokens_total' <<<"$metrics" | awk '{print $2}')
  echo "OK: speculative decoding live (drafts=$drafts accepted=$accepted)"
fi
served=$(curl -fsS --max-time 5 "http://127.0.0.1:${VLLM_PORT:-18888}/v1/models" 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4)
echo "served model: ${served:-unknown} (catch-up expects ${CATCHUP_MODEL:-unset})"
