from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download BoolQ locally and convert it to mmlu_csv-compatible files")
    parser.add_argument(
        "--output-dir",
        default="data/boolq",
        help="Output directory for converted CSV files",
    )
    parser.add_argument(
        "--model-dir",
        default="${MODEL_ROOT}/gemma-3-270m",
        help="Local Gemma model/tokenizer directory used to enforce seq_len compatibility",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum sequence length for downstream training")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def build_question_field(tokenizer, passage: str, question: str, *, seq_len: int) -> str:
    option_suffix = "\nA. yes\nB. no\nC. .\nD. .\nAnswer: "
    answer_probe = " A"
    question_suffix = f" Q: {question} Y/N?"
    prefix_without_passage = "Question: P: "
    fixed_probe = prefix_without_passage + question_suffix + option_suffix + answer_probe
    fixed_tokens = tokenizer.encode(fixed_probe, add_special_tokens=False)
    if len(fixed_tokens) + 1 > seq_len:
        raise RuntimeError(f"BoolQ question is too long even without passage: {question!r}")

    available_passage_tokens = max(0, seq_len - (len(fixed_tokens) + 1))
    passage_ids = tokenizer.encode(passage, add_special_tokens=False)
    kept = min(len(passage_ids), available_passage_tokens)
    while kept >= 0:
        truncated_passage = tokenizer.decode(
            passage_ids[:kept],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        ).strip()
        field = f"P: {truncated_passage} Q: {question} Y/N?"
        prompt = "Question: " + field + option_suffix
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        answer_ids = tokenizer.encode(answer_probe, add_special_tokens=False)
        if len(prompt_ids) + len(answer_ids) + 1 <= seq_len:
            return normalize_text(field)
        kept -= 1
    raise RuntimeError("Failed to fit BoolQ sample within seq_len after passage truncation")


def convert_split(rows, output_path: Path, *, tokenizer, seq_len: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "A", "B", "C", "D", "answer"],
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            passage = normalize_text(str(row["passage"]))
            question = normalize_text(str(row["question"]))
            answer = bool(row["answer"])
            question_field = build_question_field(tokenizer, passage, question, seq_len=seq_len)
            writer.writerow(
                {
                    "question": question_field,
                    "A": "yes",
                    "B": "no",
                    "C": ".",
                    "D": ".",
                    "answer": "A" if answer else "B",
                }
            )
            count += 1
    return count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).resolve()
    if not model_dir.is_dir():
        raise RuntimeError(f"Missing model/tokenizer directory: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    dataset = load_dataset("google/boolq")

    train_path = output_dir / "boolq_train_mmlu.csv"
    val_path = output_dir / "boolq_validation_mmlu.csv"
    test_path = output_dir / "boolq_test_mmlu.csv"

    train_count = convert_split(dataset["train"], train_path, tokenizer=tokenizer, seq_len=args.seq_len)
    val_count = convert_split(dataset["validation"], val_path, tokenizer=tokenizer, seq_len=args.seq_len)
    test_count = convert_split(dataset["validation"], test_path, tokenizer=tokenizer, seq_len=args.seq_len)

    print(f"train_csv={train_path}")
    print(f"validation_csv={val_path}")
    print(f"test_csv={test_path}")
    print(f"train_rows={train_count}")
    print(f"validation_rows={val_count}")
    print(f"test_rows={test_count}")
    print(f"model_dir={model_dir}")
    print(f"seq_len={args.seq_len}")


if __name__ == "__main__":
    main()
