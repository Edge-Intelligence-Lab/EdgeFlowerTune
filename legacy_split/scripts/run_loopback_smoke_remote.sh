#!/usr/bin/env bash
set -euo pipefail

ROOT="/datapool/BESTTOOLBOX/L-shaped"
PY="/home/BESTTOOLBOX/anaconda3/envs/oneshotLLM/bin/python"
CFG="configs/loopback_smoke.yaml"

cd "$ROOT"
mkdir -p outputs/loopback_smoke
fuser -k 19080/tcp 2>/dev/null || true

"$PY" -m lshaped.server.run_server --config "$CFG" > outputs/loopback_smoke/server.log 2>&1 &
SERVER_PID=$!
sleep 8

"$PY" -m lshaped.client.python_client --config "$CFG" --client-id nano_64 > outputs/loopback_smoke/client_nano_64.log 2>&1 &
CLIENT_PID=$!

wait $CLIENT_PID
wait $SERVER_PID
