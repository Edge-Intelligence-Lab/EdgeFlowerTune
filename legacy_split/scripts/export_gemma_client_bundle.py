from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from safetensors import safe_open
from safetensors.torch import save_file


TOKENIZER_FILES = [
    "tokenizer.json",
    "tokenizer.model",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "chat_template.jinja",
]

CONFIG_FILES = [
    "config.json",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a split-layer-0 slim Gemma client bundle")
    parser.add_argument("--source-model-dir", required=True, help="Full Gemma model directory")
    parser.add_argument("--output-dir", required=True, help="Output directory for the slim client bundle")
    parser.add_argument(
        "--tensor-key",
        default="model.embed_tokens.weight",
        help="Tensor key to keep in model.safetensors",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output-dir if it already exists",
    )
    return parser.parse_args()


def copy_if_exists(src_root: Path, dst_root: Path, relative_path: str) -> None:
    src = src_root / relative_path
    if not src.exists():
        return
    dst = dst_root / relative_path
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    source_model_dir = Path(args.source_model_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    if not source_model_dir.is_dir():
        raise RuntimeError(f"source-model-dir is not a directory: {source_model_dir}")
    if output_dir.exists():
        if not args.overwrite:
            raise RuntimeError(f"output-dir already exists: {output_dir}")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model_file = source_model_dir / "model.safetensors"
    if not model_file.is_file():
        raise RuntimeError(f"Missing model.safetensors under: {source_model_dir}")

    with safe_open(model_file, framework="pt", device="cpu") as handle:
        if args.tensor_key not in handle.keys():
            raise RuntimeError(
                f"Tensor key {args.tensor_key!r} not found in {model_file}"
            )
        tensor = handle.get_tensor(args.tensor_key)
    save_file({args.tensor_key: tensor}, str(output_dir / "model.safetensors"))

    for filename in CONFIG_FILES + TOKENIZER_FILES:
        copy_if_exists(source_model_dir, output_dir, filename)

    manifest = {
        "bundle_type": "gemma_split0_client_slim",
        "source_model_dir": str(source_model_dir),
        "split_layer": 0,
        "kept_tensors": [args.tensor_key],
        "copied_files": [
            filename
            for filename in CONFIG_FILES + TOKENIZER_FILES
            if (output_dir / filename).exists()
        ],
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    total_bytes = sum(p.stat().st_size for p in output_dir.rglob("*") if p.is_file())
    print(f"exported_bundle={output_dir}")
    print(f"total_bytes={total_bytes}")
    print(f"model_bytes={(output_dir / 'model.safetensors').stat().st_size}")


if __name__ == "__main__":
    main()
