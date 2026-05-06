#!/usr/bin/env bash
set -euo pipefail

ROOT="/datapool/BESTTOOLBOX/L-shaped"
PY="/home/BESTTOOLBOX/anaconda3/envs/oneshotLLM/bin/python"
CFG="${1:-configs/server_gemma270m.yaml}"

cd "$ROOT"
exec "$PY" -m lshaped.server.run_server --config "$CFG"

