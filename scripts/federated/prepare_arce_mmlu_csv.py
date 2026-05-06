from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer


LABEL_MAP = {"1": "A", "2": "B", "3": "C", "4": "D", "A": "A", "B": "B", "C": "C", "D": "D"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local ARC-Easy raw files to mmlu_csv-compatible files")
    parser.add_argument("--output-dir", default="data/arc", help="Output directory for converted CSV files")
    parser.add_argument(
        "--raw-dir",
        default="data/arc/ARC-Easy",
        help="Directory containing ARC-Easy train/validation/test jsonl files",
    )
    parser.add_argument(
        "--model-dir",
        default="${EDGEFLOWERTUNE_ROOT}/pretrained_models/gemma-3-270m",
        help="Local model/tokenizer directory",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum downstream sequence length")
    parser.add_argument("--overwrite", action="store_true", help="Force re-conversion")
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
    question: str,
    option_a: str,
    option_b: str,
    option_c: str,
    option_d: str,
    *,
    seq_len: int,
) -> tuple[str, str, str, str, str]:
    answer_probe = " A"
    question_ids = encode_text(tokenizer, question)
    option_a_ids = encode_text(tokenizer, option_a)
    option_b_ids = encode_text(tokenizer, option_b)
    option_c_ids = encode_text(tokenizer, option_c)
    option_d_ids = encode_text(tokenizer, option_d)

    keep_q = min(len(question_ids), 18)
    keep_a = min(len(option_a_ids), 8)
    keep_b = min(len(option_b_ids), 8)
    keep_c = min(len(option_c_ids), 8)
    keep_d = min(len(option_d_ids), 8)

    while keep_q >= 0 and keep_a >= 0 and keep_b >= 0 and keep_c >= 0 and keep_d >= 0:
        q = normalize_text(decode_prefix(tokenizer, question_ids, keep_q))
        a = normalize_text(decode_prefix(tokenizer, option_a_ids, keep_a))
        b = normalize_text(decode_prefix(tokenizer, option_b_ids, keep_b))
        c = normalize_text(decode_prefix(tokenizer, option_c_ids, keep_c))
        d = normalize_text(decode_prefix(tokenizer, option_d_ids, keep_d))
        prompt = f"Question: {q}\nA. {a}\nB. {b}\nC. {c}\nD. {d}\nAnswer: "
        if len(encode_text(tokenizer, prompt)) + len(encode_text(tokenizer, answer_probe)) + 1 <= seq_len:
            return q, a, b, c, d
        largest = max(
            [(keep_q, "q"), (keep_a, "a"), (keep_b, "b"), (keep_c, "c"), (keep_d, "d")],
            key=lambda item: item[0],
        )[1]
        if largest == "q" and keep_q > 0:
            keep_q -= 1
        elif largest == "a" and keep_a > 0:
            keep_a -= 1
        elif largest == "b" and keep_b > 0:
            keep_b -= 1
        elif largest == "c" and keep_c > 0:
            keep_c -= 1
        elif keep_d > 0:
            keep_d -= 1
        else:
            break
    raise RuntimeError("Failed to fit ARC-E sample within seq_len after truncation")


def convert_split(
    raw_path: Path,
    output_path: Path,
    *,
    tokenizer: Any,
    seq_len: int,
) -> tuple[int, int]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    kept = 0
    skipped = 0
    with raw_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8", newline="") as dst:
        writer = csv.DictWriter(
            dst,
            fieldnames=["question", "A", "B", "C", "D", "answer"],
            lineterminator="\n",
        )
        writer.writeheader()
        for line in src:
            row = json.loads(line)
            choices = row["choices"]
            texts = [normalize_text(text) for text in choices["text"]]
            labels = [str(label) for label in choices["label"]]
            answer_key = str(row["answerKey"])
            if len(texts) == 5:
                skipped += 1
                continue
            if len(texts) not in (3, 4):
                skipped += 1
                continue
            if answer_key not in LABEL_MAP:
                skipped += 1
                continue
            normalized_answer = LABEL_MAP[answer_key]
            option_by_label = dict(zip(labels, texts))
            if set(labels) == {"1", "2", "3", "4"}:
                option_a = option_by_label.get("1", ".")
                option_b = option_by_label.get("2", ".")
                option_c = option_by_label.get("3", ".")
                option_d = option_by_label.get("4", ".")
            else:
                option_a = option_by_label.get("A", ".")
                option_b = option_by_label.get("B", ".")
                option_c = option_by_label.get("C", ".")
                option_d = option_by_label.get("D", ".")
            if len(texts) == 3:
                option_d = "."
                if normalized_answer == "D":
                    skipped += 1
                    continue
            q, a, b, c, d = fit_fields(
                tokenizer,
                normalize_text(str(row["question"])),
                option_a,
                option_b,
                option_c,
                option_d,
                seq_len=seq_len,
            )
            writer.writerow({"question": q, "A": a, "B": b, "C": c, "D": d, "answer": normalized_answer})
            kept += 1
    return kept, skipped


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    train_path = output_dir / "arce_train_mmlu.csv"
    val_path = output_dir / "arce_validation_mmlu.csv"
    test_path = output_dir / "arce_test_mmlu.csv"

    if not args.overwrite and train_path.is_file() and val_path.is_file() and test_path.is_file():
        print(f"train_csv={train_path}")
        print(f"validation_csv={val_path}")
        print(f"test_csv={test_path}")
        return

    tokenizer_model = model_dir / "tokenizer.model"
    if tokenizer_model.is_file():
        tokenizer: Any = SentencePieceProcessor(model_file=str(tokenizer_model))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    train_kept, train_skipped = convert_split(raw_dir / "train.jsonl", train_path, tokenizer=tokenizer, seq_len=args.seq_len)
    val_kept, val_skipped = convert_split(raw_dir / "validation.jsonl", val_path, tokenizer=tokenizer, seq_len=args.seq_len)
    test_kept, test_skipped = convert_split(raw_dir / "test.jsonl", test_path, tokenizer=tokenizer, seq_len=args.seq_len)

    print(f"train_csv={train_path}")
    print(f"validation_csv={val_path}")
    print(f"test_csv={test_path}")
    print(f"train_rows={train_kept}")
    print(f"validation_rows={val_kept}")
    print(f"test_rows={test_kept}")
    print(f"train_skipped={train_skipped}")
    print(f"validation_skipped={val_skipped}")
    print(f"test_skipped={test_skipped}")
    print(f"raw_dir={raw_dir}")
    print(f"model_dir={model_dir}")
    print(f"seq_len={args.seq_len}")


if __name__ == "__main__":
    main()
