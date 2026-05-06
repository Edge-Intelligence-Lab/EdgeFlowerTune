import os
import subprocess
import sys


def repo_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


BASE = repo_root()
SCRIPT = os.path.join(BASE, "scripts", "prepare_mmlu_jsonl.py")
OUT_DIR = os.path.join(BASE, "runs", "mmlu_jsonl_gemma270m_s128")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "pretrained")
SEQ_LEN = 128


subprocess.run(
    [
        sys.executable,
        SCRIPT,
        "--model_dir",
        MODEL_DIR,
        "--output_dir",
        OUT_DIR,
        "--seq_len",
        str(SEQ_LEN),
        "--no_use_fast",
        "--overwrite",
    ],
    check=True,
)

