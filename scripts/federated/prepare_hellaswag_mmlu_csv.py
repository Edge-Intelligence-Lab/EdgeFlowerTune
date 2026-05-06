from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local HellaSwag jsonl files to mmlu_csv-compatible files")
    parser.add_argument("--output-dir", default="data/hellaswag", help="Output directory for converted CSV files")
    parser.add_argument("--raw-dir", default="data/hellaswag", help="Directory containing HellaSwag train/validation/test jsonl")
    parser.add_argument(
        "--model-dir",
        default="${EDGEFLOWERTUNE_ROOT}/pretrained_models/gemma-3-270m",
        help="Local model/tokenizer directory",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum downstream sequence length")
    parser.add_argument("--overwrite", action="store_true", help="Rebuild outputs even if they already exist")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def encode_text(tokenizer: Any, text: str) -> list[int]:
    if isinstance(tokenizer, SentencePieceProcessor):
        return list(tokenizer.encode(text))
    return list(tokenizer.encode(text, add_special_tokens=False))


def decode_prefix(tokenizer: Any, token_ids: list[int], keep: int) -> str:
    if keep <= 0:
        return ""
    if isinstance(tokenizer, SentencePieceProcessor):
        return tokenizer.decode(token_ids[:keep]).strip()
    return tokenizer.decode(
        token_ids[:keep],
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    ).strip()


def fit_fields(
    tokenizer: Any,
    ctx: str,
    ending_a: str,
    ending_b: str,
    ending_c: str,
    ending_d: str,
    *,
    seq_len: int,
) -> tuple[str, str, str, str, str]:
    answer_probe = " A"
    ctx_ids = encode_text(tokenizer, ctx)
    a_ids = encode_text(tokenizer, ending_a)
    b_ids = encode_text(tokenizer, ending_b)
    c_ids = encode_text(tokenizer, ending_c)
    d_ids = encode_text(tokenizer, ending_d)

    keep = {
        "ctx": min(len(ctx_ids), 14),
        "a": min(len(a_ids), 8),
        "b": min(len(b_ids), 8),
        "c": min(len(c_ids), 8),
        "d": min(len(d_ids), 8),
    }
    id_map = {"ctx": ctx_ids, "a": a_ids, "b": b_ids, "c": c_ids, "d": d_ids}

    while min(keep.values()) >= 0:
        ctx_text = decode_prefix(tokenizer, id_map["ctx"], keep["ctx"])
        a_text = decode_prefix(tokenizer, id_map["a"], keep["a"])
        b_text = decode_prefix(tokenizer, id_map["b"], keep["b"])
        c_text = decode_prefix(tokenizer, id_map["c"], keep["c"])
        d_text = decode_prefix(tokenizer, id_map["d"], keep["d"])
        question_field = normalize_text(f"C: {ctx_text}")
        prompt = (
            "Question: "
            + question_field
            + "\nA. "
            + normalize_text(a_text)
            + "\nB. "
            + normalize_text(b_text)
            + "\nC. "
            + normalize_text(c_text)
            + "\nD. "
            + normalize_text(d_text)
            + "\nAnswer: "
        )
        prompt_ids = encode_text(tokenizer, prompt)
        answer_ids = encode_text(tokenizer, answer_probe)
        if len(prompt_ids) + len(answer_ids) + 1 <= seq_len:
            return (
                question_field,
                normalize_text(a_text),
                normalize_text(b_text),
                normalize_text(c_text),
                normalize_text(d_text),
            )

        largest_key = max(keep, key=lambda k: keep[k])
        if keep[largest_key] <= 0:
            break
        keep[largest_key] -= 1

    raise RuntimeError("Failed to fit HellaSwag sample within seq_len after truncation")


def read_rows(jsonl_path: Path) -> list[tuple[str, list[str], int]]:
    rows: list[tuple[str, list[str], int]] = []
    with jsonl_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            label = item.get("label")
            if label in (None, "", -1):
                continue
            endings = item.get("endings")
            if not isinstance(endings, list) or len(endings) != 4:
                raise ValueError(f"Expected 4 endings in {jsonl_path}")
            rows.append(
                (
                    normalize_text(str(item["ctx"])),
                    [normalize_text(str(x)) for x in endings],
                    int(label),
                )
            )
    return rows


def convert_split(rows: list[tuple[str, list[str], int]], output_path: Path, *, tokenizer: Any, seq_len: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "A", "B", "C", "D", "answer"],
            lineterminator="\n",
        )
        writer.writeheader()
        for ctx, endings, label in rows:
            question_field, a_text, b_text, c_text, d_text = fit_fields(
                tokenizer,
                ctx,
                endings[0],
                endings[1],
                endings[2],
                endings[3],
                seq_len=seq_len,
            )
            writer.writerow(
                {
                    "question": question_field,
                    "A": a_text,
                    "B": b_text,
                    "C": c_text,
                    "D": d_text,
                    "answer": ["A", "B", "C", "D"][label],
                }
            )
            count += 1
    return count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    if not model_dir.is_dir():
        raise RuntimeError(f"Missing model/tokenizer directory: {model_dir}")

    tokenizer_model = model_dir / "tokenizer.model"
    if tokenizer_model.is_file():
        tokenizer: Any = SentencePieceProcessor(model_file=str(tokenizer_model))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    train_rows = read_rows(raw_dir / "train.jsonl")
    val_rows = read_rows(raw_dir / "validation.jsonl")

    train_path = output_dir / "hellaswag_train_mmlu.csv"
    val_path = output_dir / "hellaswag_validation_mmlu.csv"
    test_path = output_dir / "hellaswag_test_mmlu.csv"

    if (
        not args.overwrite
        and train_path.is_file()
        and val_path.is_file()
        and test_path.is_file()
    ):
        print(f"train_csv={train_path}")
        print(f"validation_csv={val_path}")
        print(f"test_csv={test_path}")
        print("reused_existing=1")
        print(f"raw_dir={raw_dir}")
        print(f"model_dir={model_dir}")
        print(f"seq_len={args.seq_len}")
        return

    train_count = convert_split(train_rows, train_path, tokenizer=tokenizer, seq_len=args.seq_len)
    val_count = convert_split(val_rows, val_path, tokenizer=tokenizer, seq_len=args.seq_len)
    test_count = convert_split(val_rows, test_path, tokenizer=tokenizer, seq_len=args.seq_len)

    print(f"train_csv={train_path}")
    print(f"validation_csv={val_path}")
    print(f"test_csv={test_path}")
    print(f"train_rows={train_count}")
    print(f"validation_rows={val_count}")
    print(f"test_rows={test_count}")
    print(f"raw_dir={raw_dir}")
    print(f"model_dir={model_dir}")
    print(f"seq_len={args.seq_len}")


if __name__ == "__main__":
    main()
