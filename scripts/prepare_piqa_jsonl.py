#!/usr/bin/env python3
import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from transformers import AutoTokenizer


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build PIQA JSONL datasets with masked labels on the gold solution."
    )
    parser.add_argument(
        "--data_dir",
        default=str(repo_root() / "data" / "piqa" / "raw" / "physicaliqa-train-dev"),
        help="Directory containing PIQA train/dev jsonl and label files.",
    )
    parser.add_argument("--model_dir", required=True, help="Tokenizer/model directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files.")
    parser.add_argument("--seq_len", type=int, default=256, help="Maximum sequence length.")
    parser.add_argument("--train_file", default="train.jsonl")
    parser.add_argument("--train_labels", default="train-labels.lst")
    parser.add_argument("--valid_file", default="dev.jsonl")
    parser.add_argument("--valid_labels", default="dev-labels.lst")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--use_fast",
        dest="use_fast",
        action="store_true",
        help="Force Hugging Face fast tokenizer.",
    )
    parser.add_argument(
        "--no_use_fast",
        dest="use_fast",
        action="store_false",
        help="Force Hugging Face slow tokenizer.",
    )
    parser.set_defaults(use_fast=True)
    return parser.parse_args()


def load_tokenizer(model_dir: str, use_fast: Optional[bool]):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Tokenizer/model directory not found: {model_dir}")
    tok = AutoTokenizer.from_pretrained(model_dir, use_fast=use_fast)
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos is None:
        raise RuntimeError("Tokenizer must provide eos_token_id or pad_token_id.")
    return tok, eos


def normalize_text(text: str) -> str:
    return " ".join(text.strip().split())


def iter_examples(jsonl_path: Path, labels_path: Path):
    rows = jsonl_path.read_text(encoding="utf-8").splitlines()
    labels = labels_path.read_text(encoding="utf-8").splitlines()
    if len(rows) != len(labels):
        raise ValueError(f"Mismatched PIQA rows/labels: {jsonl_path} vs {labels_path}")
    for row, label in zip(rows, labels):
        row = row.strip()
        if not row:
            continue
        ex = json.loads(row)
        yield ex, int(label)


def build_prompt_and_answer(ex: Dict[str, object], label: int) -> Tuple[str, str]:
    goal = normalize_text(str(ex["goal"]))
    if label not in (0, 1):
        raise ValueError(f"Unexpected PIQA label: {label}")
    answer = normalize_text(str(ex["sol1"] if label == 0 else ex["sol2"]))
    prompt = f"Goal: {goal}\nSolution:"
    return prompt, answer


def build_ids_mask(tok, eos: int, ex: Dict[str, object], label: int, seq_len: int) -> Optional[Tuple[List[int], List[int]]]:
    prompt, answer = build_prompt_and_answer(ex, label)
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    answer_ids = tok.encode(" " + answer, add_special_tokens=False)
    ids = prompt_ids + answer_ids + [eos]
    if len(ids) > seq_len:
        return None
    mask = [0] * len(prompt_ids) + [1] * len(answer_ids) + [0]
    if len(ids) < seq_len:
        pad = seq_len - len(ids)
        ids = ids + [eos] * pad
        mask = mask + [0] * pad
    return ids, mask


def pair_signature(ids: List[int], mask: List[int]) -> str:
    payload = json.dumps({"ids": ids, "mask": mask}, separators=(",", ":"))
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()


def write_jsonl(path: Path, pairs: List[Tuple[List[int], List[int]]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for ids, mask in pairs:
            f.write(json.dumps({"ids": ids, "mask": mask}) + "\n")


def main() -> None:
    args = parse_args()
    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    outputs = {
        "train": output_dir / "train.jsonl",
        "valid": output_dir / "valid.jsonl",
    }
    if not args.overwrite:
        existing = [str(p) for p in outputs.values() if p.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Pass --overwrite to rebuild: " + ", ".join(existing)
            )

    tok, eos = load_tokenizer(args.model_dir, args.use_fast)
    split_files = {
        "train": (data_dir / args.train_file, data_dir / args.train_labels),
        "valid": (data_dir / args.valid_file, data_dir / args.valid_labels),
    }
    manifest: Dict[str, object] = {
        "data_dir": str(data_dir),
        "model_dir": str(Path(args.model_dir).resolve()),
        "seq_len": args.seq_len,
        "tokenizer_use_fast": args.use_fast,
        "split_files": {
            k: {"jsonl": str(v[0]), "labels": str(v[1])} for k, v in split_files.items()
        },
        "task": "piqa",
        "format": "goal + gold solution, masked on solution only",
        "splits": {},
    }

    global_seen = set()
    for logical_name, (jsonl_path, labels_path) in split_files.items():
        if not jsonl_path.is_file():
            raise FileNotFoundError(f"Missing split file: {jsonl_path}")
        if not labels_path.is_file():
            raise FileNotFoundError(f"Missing label file: {labels_path}")

        pairs: List[Tuple[List[int], List[int]]] = []
        local_seen = set()
        raw_count = 0
        too_long = 0
        dropped_local_dup = 0
        dropped_cross_dup = 0

        for ex, label in iter_examples(jsonl_path, labels_path):
            raw_count += 1
            built = build_ids_mask(tok, eos, ex, label, args.seq_len)
            if built is None:
                too_long += 1
                continue
            ids, mask = built
            sig = pair_signature(ids, mask)
            if sig in local_seen:
                dropped_local_dup += 1
                continue
            if sig in global_seen:
                dropped_cross_dup += 1
                continue
            local_seen.add(sig)
            global_seen.add(sig)
            pairs.append((ids, mask))

        write_jsonl(outputs[logical_name], pairs)
        manifest["splits"][logical_name] = {
            "raw_examples": raw_count,
            "kept_examples": len(pairs),
            "dropped_too_long": too_long,
            "dropped_local_duplicates": dropped_local_dup,
            "dropped_cross_split_duplicates": dropped_cross_dup,
            "output_file": str(outputs[logical_name]),
        }

    with (output_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[PIQA JSONL] Done: {output_dir}")
    for split_name in ("train", "valid"):
        info = manifest["splits"][split_name]
        print(
            f"  {split_name}: kept={info['kept_examples']} "
            f"too_long={info['dropped_too_long']} cross_dup={info['dropped_cross_split_duplicates']}"
        )


if __name__ == "__main__":
    main()
