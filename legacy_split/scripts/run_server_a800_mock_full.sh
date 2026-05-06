#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES=4
/home/BESTTOOLBOX/anaconda3/envs/oneshotLLM/bin/python -m lshaped.server.run_server --config configs/a800_mock_full.yaml
