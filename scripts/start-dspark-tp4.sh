#!/usr/bin/env bash
# Launch DeepSeek V4 Flash DSpark as TP=4 with forge as head.
# Workers first (anvil, ember, flame), then forge. Fabric is 192.168.2.0/24.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env.dspark.tp4.railb-200g.bench}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark-tp4.yml}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-180}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

if [[ ! -f "$ENV_FILE" ]]; then
  echo "missing env file: $ENV_FILE" >&2
  exit 1
fi
if [[ ! -f "$COMPOSE_FILE" ]]; then
  echo "missing compose file: $COMPOSE_FILE" >&2
  exit 1
fi

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash-tp4}"
MASTER_ADDR="${MASTER_ADDR:-192.168.2.1}"
MASTER_PORT="${MASTER_PORT:-25000}"
VLLM_PORT="${VLLM_PORT:-18888}"
API_URL="http://127.0.0.1:${VLLM_PORT}/v1/models"
HCA="${NCCL_IB_HCA:-rocep1s0f1}"
IFACE="${NCCL_SOCKET_IFNAME:-enp1s0f1np1}"

RANKS=(
  "0|forge|${FORGE_IP:-192.168.2.1}|"
  "1|anvil|${ANVIL_IP:-192.168.2.2}|1"
  "2|ember|${EMBER_IP:-192.168.2.3}|1"
  "3|flame|${FLAME_IP:-192.168.2.4}|1"
)

need_cmd() { command -v "$1" >/dev/null || { echo "missing command: $1" >&2; exit 1; }; }
need_cmd docker
need_cmd ssh
need_cmd scp
need_cmd curl
need_cmd python3

on_host() {
  local host="$1"
  shift
  if [[ "$host" == "$(hostname)" || "$host" == "forge" ]]; then
    bash -lc "$*"
  else
    ssh "${SSH_OPTS[@]}" "$host" "$*"
  fi
}

resolve_gid() {
  local host="$1" match_ip="$2"
  on_host "$host" "python3 -c \"
import ipaddress, os
hca = '$HCA'
want = ipaddress.IPv4Address('$match_ip')
base = f'/sys/class/infiniband/{hca}/ports/1'
for idx in map(str, range(16)):
    try:
        gid = open(f'{base}/gids/{idx}').read().strip()
        typ = open(f'{base}/gid_attrs/types/{idx}').read().strip()
    except OSError:
        continue
    if 'RoCE v2' not in typ:
        continue
    parts = gid.split(':')
    if len(parts) >= 8 and parts[5].lower() == 'ffff':
        ip = ipaddress.IPv4Address(int(parts[6] + parts[7], 16))
        if ip == want:
            print(idx)
            raise SystemExit(0)
raise SystemExit('no RoCEv2 GID for ' + str(want) + ' on ' + hca)
\""
}

compose_on() {
  local host="$1" rank="$2" ip="$3" headless="$4" gid="$5" action="$6"
  local extra=""
  if [[ -n "$headless" ]]; then
    extra="HEADLESS=1"
  else
    extra="HEADLESS="
  fi
  on_host "$host" "cd '$SCRIPT_DIR' && env -u NODE_RANK -u HEADLESS -u VLLM_HOST_IP -u NCCL_IB_GID_INDEX COMPOSE_DISABLE_ENV_FILE=1 \
    NODE_RANK='$rank' $extra VLLM_HOST_IP='$ip' NCCL_IB_GID_INDEX='$gid' \
    MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='$MASTER_PORT' \
    docker compose -p '$PROJECT_NAME' --env-file '$ENV_FILE' -f '$COMPOSE_FILE' $action"
}

echo "Resolving RoCEv2 GID indexes on $IFACE / $HCA..."
declare -A GID
for entry in "${RANKS[@]}"; do
  IFS='|' read -r rank host ip headless <<<"$entry"
  gid="$(resolve_gid "$host" "$ip")"
  GID["$host"]="$gid"
  echo "  rank $rank $host $ip gid=$gid"
done

echo "Syncing TP-4 compose/env to workers..."
for entry in "${RANKS[@]}"; do
  IFS='|' read -r rank host ip headless <<<"$entry"
  [[ "$host" == "forge" ]] && continue
  ssh "${SSH_OPTS[@]}" "$host" "mkdir -p '$SCRIPT_DIR'"
  scp -q "${SSH_OPTS[@]}" "$ENV_FILE" "$COMPOSE_FILE" \
    "$SCRIPT_DIR/start-dspark-tp4.sh" "$SCRIPT_DIR/stop-dspark-tp4.sh" \
    "$SCRIPT_DIR/status-dspark-tp4.sh" \
    "${host}:${SCRIPT_DIR}/"
done

echo "Stopping any previous $PROJECT_NAME containers..."
for entry in "${RANKS[@]}"; do
  IFS='|' read -r rank host ip headless <<<"$entry"
  on_host "$host" "docker rm -f ${PROJECT_NAME}-vllm-dspark-1 >/dev/null 2>&1 || true"
done

echo "Starting workers first..."
for entry in "${RANKS[@]}"; do
  IFS='|' read -r rank host ip headless <<<"$entry"
  [[ -z "$headless" ]] && continue
  echo "  rank $rank $host ($ip)"
  compose_on "$host" "$rank" "$ip" "$headless" "${GID[$host]}" "up -d"
done

echo "Starting forge head (rank 0)..."
compose_on forge 0 "${FORGE_IP:-192.168.2.1}" "" "${GID[forge]}" "up -d"

echo "Waiting for API at $API_URL ..."
for i in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null 2>&1; then
    echo "API is up - running serving-shape gate..."
    metrics="$(curl -fsS --max-time 8 "http://127.0.0.1:${VLLM_PORT}/metrics" 2>/dev/null || true)"
    if ! grep -q "^vllm:spec_decode_num_drafts_total" <<<"$metrics"; then
      echo "GATE FAIL: no spec-decode counters (wrong image/entrypoint?). Stopping all ranks; auto-restart stays DISARMED." >&2
      for entry in "${RANKS[@]}"; do
        IFS="|" read -r _ host _ _ <<<"$entry"
        on_host "$host" "docker stop ${PROJECT_NAME}-vllm-dspark-1 >/dev/null 2>&1 || true"
      done
      exit 1
    fi
    echo "Gate OK (speculative decoding live). Arming auto-restart..."
    for entry in "${RANKS[@]}"; do
      IFS="|" read -r _ host _ _ <<<"$entry"
      on_host "$host" "docker update --restart=unless-stopped ${PROJECT_NAME}-vllm-dspark-1 >/dev/null"
    done
    echo "TP-4 is up: $API_URL"
    curl -fsS --max-time 5 "$API_URL" || true
    echo
    exit 0
  fi
  if (( i % 6 == 0 )); then
    echo "  still waiting ($i/${WAIT_ATTEMPTS})..."
    on_host forge "docker logs --tail 8 ${PROJECT_NAME}-vllm-dspark-1 2>&1 | tail -8" || true
  fi
  sleep 10
done

echo "API did not come up within $((WAIT_ATTEMPTS * 10))s" >&2
echo "--- forge tail ---"
docker logs --tail 40 "${PROJECT_NAME}-vllm-dspark-1" >&2 || true
echo "--- anvil tail ---"
on_host anvil "docker logs --tail 20 ${PROJECT_NAME}-vllm-dspark-1" >&2 || true
exit 1
