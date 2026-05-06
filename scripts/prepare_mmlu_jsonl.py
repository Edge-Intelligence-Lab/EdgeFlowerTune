#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from transformers import AutoTokenizer


DEFAULT_TRAIN_SPLIT = "auxiliary_train"
DEFAULT_VALID_SPLIT = "val"
DEFAULT_TEST_SPLIT = "test"


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build MMLU JSONL datasets with official split boundaries."
    )
    parser.add_argument(
        "--data_dir",
        default=str(repo_root() / "data" / "mmlu" / "data"),
        help="Official MMLU data root containing auxiliary_train/dev/val/test.",
    )
    parser.add_argument("--model_dir", required=True, help="Tokenizer/model directory.")
    parser.add_argument("--output_dir", required=True, help="Output directory for JSONL files.")
    parser.add_argument("--seq_len", type=int, default=128, help="Maximum sequence length.")
    parser.add_argument(
        "--train_split",
        default=DEFAULT_TRAIN_SPLIT,
        help="Official subdirectory used for training.",
    )
    parser.add_argument(
        "--valid_split",
        default=DEFAULT_VALID_SPLIT,
        help="Official subdirectory used for validation.",
    )
    parser.add_argument(
        "--test_split",
        default=DEFAULT_TEST_SPLIT,
        help="Official subdirectory used for testing.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Metadata only. No random split is performed.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow overwriting existing train/valid/test JSONL files.",
    )
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
    parser.set_defaults(use_fast=None)
    return parser.parse_args()


def load_tokenizer(model_dir: str, use_fast: Optional[bool]):
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Tokenizer/model directory not found: {model_dir}")
    kwargs = {}
    if use_fast is not None:
        kwargs["use_fast"] = use_fast
    tok = AutoTokenizer.from_pretrained(model_dir, **kwargs)
    if tok.pad_token_id is None:
        if tok.eos_token is not None:
            tok.pad_token = tok.eos_token
        elif tok.unk_token is not None:
            tok.pad_token = tok.unk_token
    eos = tok.eos_token_id if tok.eos_token_id is not None else tok.pad_token_id
    if eos is None:
        raise RuntimeError("Tokenizer must provide eos_token_id or pad_token_id.")
    return tok, eos


def parse_csv(path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 6:
                continue
            ans = row[5].strip()
            if not ans or ans[0] not in "ABCDabcd":
                continue
            items.append(
                {
                    "question": row[0],
                    "A": row[1],
                    "B": row[2],
                    "C": row[3],
                    "D": row[4],
                    "answer": ans[0].upper(),
                    "source_file": path.name,
                }
            )
    return items


def iter_split_examples(data_dir: Path, split_name: str) -> Iterable[Dict[str, str]]:
    split_dir = data_dir / split_name
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing official MMLU split directory: {split_dir}")
    for csv_path in sorted(split_dir.glob("*.csv")):
        for ex in parse_csv(csv_path):
            yield ex


def build_ids_mask(tok, eos: int, ex: Dict[str, str], seq_len: int) -> Optional[Tuple[List[int], List[int]]]:
    prompt = (
        f"Question: {ex['question']}\n"
        f"A. {ex['A']}\n"
        f"B. {ex['B']}\n"
        f"C. {ex['C']}\n"
        f"D. {ex['D']}\n"
        f"Answer: "
    )
    prompt_ids = tok.encode(prompt, add_special_tokens=False)
    answer_ids = tok.encode(" " + ex["answer"], add_special_tokens=False)
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
        "test": output_dir / "test.jsonl",
    }
    if not args.overwrite:
        existing = [str(p) for p in outputs.values() if p.exists()]
        if existing:
            raise FileExistsError(
                "Output files already exist. Pass --overwrite to rebuild: " + ", ".join(existing)
            )

    tok, eos = load_tokenizer(args.model_dir, args.use_fast)

    split_plan = [
        ("train", args.train_split),
        ("valid", args.valid_split),
        ("test", args.test_split),
    ]
    global_seen = set()
    manifest: Dict[str, object] = {
        "data_dir": str(data_dir),
        "model_dir": str(Path(args.model_dir).resolve()),
        "seq_len": args.seq_len,
        "seed": args.seed,
        "tokenizer_use_fast": args.use_fast,
        "split_plan": {logical: split_name for logical, split_name in split_plan},
        "splits": {},
    }

    for logical_name, official_split in split_plan:
        pairs: List[Tuple[List[int], List[int]]] = []
        local_seen = set()
        raw_count = 0
        too_long = 0
        dropped_local_dup = 0
        dropped_cross_dup = 0

        for ex in iter_split_examples(data_dir, official_split):
            raw_count += 1
            built = build_ids_mask(tok, eos, ex, args.seq_len)
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
            "source_split": official_split,
            "raw_examples": raw_count,
            "kept_examples": len(pairs),
            "dropped_too_long": too_long,
            "dropped_local_duplicates": dropped_local_dup,
            "dropped_cross_split_duplicates": dropped_cross_dup,
            "output_file": str(outputs[logical_name]),
        }

    manifest_path = output_dir / "manifest.strict.json"
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"[Strict MMLU] Done: {output_dir}")
    for split_name in ("train", "valid", "test"):
        info = manifest["splits"][split_name]
        print(
            f"  {split_name}: source={info['source_split']} kept={info['kept_examples']} "
            f"too_long={info['dropped_too_long']} cross_dup={info['dropped_cross_split_duplicates']}"
        )


if __name__ == "__main__":
    main()
