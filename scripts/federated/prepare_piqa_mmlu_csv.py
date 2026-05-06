from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert local PIQA raw files to mmlu_csv-compatible files")
    parser.add_argument("--output-dir", default="data/piqa", help="Output directory for converted CSV files")
    parser.add_argument(
        "--raw-dir",
        default="data/piqa/raw/physicaliqa-train-dev",
        help="Directory containing PIQA train/dev jsonl and label files",
    )
    parser.add_argument(
        "--model-dir",
        default="${EDGEFLOWERTUNE_ROOT}/pretrained_models/gemma-3-270m",
        help="Local model/tokenizer directory",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum downstream sequence length")
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


def fit_fields(tokenizer: Any, goal: str, sol1: str, sol2: str, *, seq_len: int) -> tuple[str, str, str]:
    option_c = "."
    option_d = "."
    answer_probe = " A"
    goal_ids = encode_text(tokenizer, goal)
    sol1_ids = encode_text(tokenizer, sol1)
    sol2_ids = encode_text(tokenizer, sol2)

    keep_goal = len(goal_ids)
    keep_sol1 = len(sol1_ids)
    keep_sol2 = len(sol2_ids)

    while keep_goal >= 0 and keep_sol1 >= 0 and keep_sol2 >= 0:
        goal_text = decode_prefix(tokenizer, goal_ids, keep_goal)
        sol1_text = decode_prefix(tokenizer, sol1_ids, keep_sol1)
        sol2_text = decode_prefix(tokenizer, sol2_ids, keep_sol2)
        question_field = normalize_text(f"G: {goal_text}")
        prompt = (
            "Question: "
            + question_field
            + "\nA. "
            + normalize_text(sol1_text)
            + "\nB. "
            + normalize_text(sol2_text)
            + "\nC. "
            + option_c
            + "\nD. "
            + option_d
            + "\nAnswer: "
        )
        prompt_ids = encode_text(tokenizer, prompt)
        answer_ids = encode_text(tokenizer, answer_probe)
        if len(prompt_ids) + len(answer_ids) + 1 <= seq_len:
            return question_field, normalize_text(sol1_text), normalize_text(sol2_text)

        largest = max(
            (keep_goal, "goal"),
            (keep_sol1, "sol1"),
            (keep_sol2, "sol2"),
            key=lambda item: item[0],
        )[1]
        if largest == "goal" and keep_goal > 0:
            keep_goal -= 1
        elif largest == "sol1" and keep_sol1 > 0:
            keep_sol1 -= 1
        elif keep_sol2 > 0:
            keep_sol2 -= 1
        else:
            break

    raise RuntimeError("Failed to fit PIQA sample within seq_len after truncation")


def read_rows(jsonl_path: Path, labels_path: Path) -> list[tuple[str, str, str, int]]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [int(line.strip()) for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(labels):
        raise ValueError(f"Mismatched PIQA rows/labels: {jsonl_path} vs {labels_path}")
    result: list[tuple[str, str, str, int]] = []
    for row, label in zip(rows, labels):
        result.append(
            (
                normalize_text(str(row["goal"])),
                normalize_text(str(row["sol1"])),
                normalize_text(str(row["sol2"])),
                label,
            )
        )
    return result


def convert_split(rows: list[tuple[str, str, str, int]], output_path: Path, *, tokenizer: Any, seq_len: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "A", "B", "C", "D", "answer"],
            lineterminator="\n",
        )
        writer.writeheader()
        for goal, sol1, sol2, label in rows:
            question_field, option_a, option_b = fit_fields(tokenizer, goal, sol1, sol2, seq_len=seq_len)
            writer.writerow(
                {
                    "question": question_field,
                    "A": option_a,
                    "B": option_b,
                    "C": ".",
                    "D": ".",
                    "answer": "A" if label == 0 else "B",
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

    train_rows = read_rows(raw_dir / "train.jsonl", raw_dir / "train-labels.lst")
    val_rows = read_rows(raw_dir / "dev.jsonl", raw_dir / "dev-labels.lst")

    train_path = output_dir / "piqa_train_mmlu.csv"
    val_path = output_dir / "piqa_validation_mmlu.csv"
    test_path = output_dir / "piqa_test_mmlu.csv"

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
