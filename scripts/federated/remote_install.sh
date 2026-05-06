#!/usr/bin/env bash
set -euo pipefail

ROOT="/datapool/BESTTOOLBOX/L-shaped"
PY="/home/BESTTOOLBOX/anaconda3/envs/oneshotLLM/bin/python"

cd "$ROOT"
"$PY" -m pip install -U pip
"$PY" -m pip install -r requirements-server.txt
"$PY" -m pip install -e .

