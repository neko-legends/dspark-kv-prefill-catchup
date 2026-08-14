#!/usr/bin/env bash
# Run the catch-up sidecar on the harness host (not on a Spark rank).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env}"
if [[ -f "$ENV_FILE" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "$ENV_FILE"
  set +a
fi
cd "$ROOT"
exec python3 -m catchup \
  --listen "${CATCHUP_LISTEN:-127.0.0.1:18900}" \
  --vllm "${CATCHUP_VLLM_URL:-http://127.0.0.1:18888/v1}" \
  --model "${CATCHUP_MODEL:-${SERVED_MODEL_NAME:-}}" \
  --max-context "${CATCHUP_MAX_CONTEXT:-1000000}"
