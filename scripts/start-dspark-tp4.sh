#!/usr/bin/env bash
# Worker-first TP=4 launch. Run on the head. Requires .env and compose beside this script
# or ENV_FILE / COMPOSE_FILE overrides.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_FILE="${ENV_FILE:-$SCRIPT_DIR/.env}"
COMPOSE_FILE="${COMPOSE_FILE:-$SCRIPT_DIR/docker-compose.dspark-tp4.yml}"
WAIT_ATTEMPTS="${WAIT_ATTEMPTS:-180}"
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new)

[[ -f "$ENV_FILE" ]] || { echo "missing $ENV_FILE" >&2; exit 1; }
[[ -f "$COMPOSE_FILE" ]] || { echo "missing $COMPOSE_FILE" >&2; exit 1; }

set -a
# shellcheck disable=SC1090
source "$ENV_FILE"
set +a

PROJECT_NAME="${PROJECT_NAME:-deepseek-v4-flash-tp4}"
HEAD_HOST="${HEAD_HOST:-node0}"
IFS=',' read -r -a WORKERS <<< "${WORKER_HOSTS:-node1,node2,node3}"
IPS=("${NODE0_IP:-192.168.2.1}" "${NODE1_IP:-192.168.2.2}" "${NODE2_IP:-192.168.2.3}" "${NODE3_IP:-192.168.2.4}")
HOSTS=("$HEAD_HOST" "${WORKERS[0]}" "${WORKERS[1]}" "${WORKERS[2]}")
HCA="${NCCL_IB_HCA:-rocep1s0f1}"
API_URL="http://127.0.0.1:${VLLM_PORT:-18888}/v1/models"

on_host() {
  local host="$1"; shift
  if [[ "$host" == "$(hostname)" || "$host" == "$HEAD_HOST" ]]; then
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
raise SystemExit('no RoCEv2 GID for ' + str(want))
\""
}

compose_on() {
  local host="$1" rank="$2" ip="$3" headless="$4" gid="$5" action="$6"
  local extra="HEADLESS="
  [[ -n "$headless" ]] && extra="HEADLESS=1"
  on_host "$host" "cd '$SCRIPT_DIR' && env -u NODE_RANK -u HEADLESS -u VLLM_HOST_IP -u NCCL_IB_GID_INDEX COMPOSE_DISABLE_ENV_FILE=1 \
    NODE_RANK='$rank' $extra VLLM_HOST_IP='$ip' NCCL_IB_GID_INDEX='$gid' \
    MASTER_ADDR='$MASTER_ADDR' MASTER_PORT='${MASTER_PORT:-25000}' \
    docker compose -p '$PROJECT_NAME' --env-file '$ENV_FILE' -f '$COMPOSE_FILE' $action"
}

echo "Resolving RoCEv2 GID indexes..."
declare -A GID
for i in 0 1 2 3; do
  GID["${HOSTS[$i]}"]="$(resolve_gid "${HOSTS[$i]}" "${IPS[$i]}")"
  echo "  rank $i ${HOSTS[$i]} ${IPS[$i]} gid=${GID[${HOSTS[$i]}]}"
done

echo "Stopping previous $PROJECT_NAME containers..."
for host in "${HOSTS[@]}"; do
  on_host "$host" "docker rm -f ${PROJECT_NAME}-vllm-dspark-1 >/dev/null 2>&1 || true"
done

echo "Starting workers..."
for i in 1 2 3; do
  compose_on "${HOSTS[$i]}" "$i" "${IPS[$i]}" "1" "${GID[${HOSTS[$i]}]}" "up -d"
done
echo "Starting head..."
compose_on "${HOSTS[0]}" 0 "${IPS[0]}" "" "${GID[${HOSTS[0]}]}" "up -d"

echo "Waiting for $API_URL ..."
for i in $(seq 1 "$WAIT_ATTEMPTS"); do
  if curl -fsS --max-time 5 "$API_URL" >/dev/null 2>&1; then
    echo "API is up - running serving-shape gate..."
    metrics="$(curl -fsS --max-time 8 "http://127.0.0.1:${VLLM_PORT:-18888}/metrics" 2>/dev/null || true)"
    if ! grep -q '^vllm:spec_decode_num_drafts_total' <<<"$metrics"; then
      echo "GATE FAIL: no spec-decode counters (wrong image/entrypoint?). Stopping all ranks; auto-restart stays DISARMED." >&2
      for host in "${HOSTS[@]}"; do
        on_host "$host" "docker stop ${PROJECT_NAME}-vllm-dspark-1 >/dev/null 2>&1 || true"
      done
      exit 1
    fi
    echo "Gate OK (speculative decoding live). Arming auto-restart..."
    for host in "${HOSTS[@]}"; do
      on_host "$host" "docker update --restart=unless-stopped ${PROJECT_NAME}-vllm-dspark-1 >/dev/null"
    done
    echo "TP-4 is up: $API_URL"
    curl -fsS --max-time 5 "$API_URL" || true
    exit 0
  fi
  sleep 10
done
echo "API did not come up" >&2
exit 1
