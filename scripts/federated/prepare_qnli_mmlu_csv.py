from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download QNLI locally and convert it to mmlu_csv-compatible files")
    parser.add_argument("--output-dir", default="data/qnli", help="Output directory for converted CSV files")
    parser.add_argument(
        "--model-dir",
        default="${EDGEFLOWERTUNE_ROOT}/pretrained_models/gemma-3-270m",
        help="Local model/tokenizer directory",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum downstream sequence length")
    return parser.parse_args()


def normalize_text(value: str) -> str:
    return " ".join(value.strip().split())


def fit_question_field(tokenizer, sentence: str, question: str, *, seq_len: int) -> str:
    option_suffix = "\nA. yes\nB. no\nC. .\nD. .\nAnswer: "
    answer_probe = " A"
    base_prefix = "Question: S: "
    infix = " Q: "
    suffix = " entailment?"
    fixed_probe = base_prefix + infix + suffix + option_suffix + answer_probe
    fixed_tokens = tokenizer.encode(fixed_probe, add_special_tokens=False)
    if len(fixed_tokens) + 1 > seq_len:
        raise RuntimeError(f"QNLI sample is too long even without content: {question!r}")

    available_tokens = max(0, seq_len - (len(fixed_tokens) + 1))
    sentence_ids = tokenizer.encode(sentence, add_special_tokens=False)
    question_ids = tokenizer.encode(question, add_special_tokens=False)

    keep_question = min(len(question_ids), available_tokens)
    keep_sentence = min(len(sentence_ids), max(0, available_tokens - keep_question))

    while keep_question >= 0:
        while keep_sentence >= 0:
            truncated_sentence = tokenizer.decode(
                sentence_ids[:keep_sentence],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            truncated_question = tokenizer.decode(
                question_ids[:keep_question],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            ).strip()
            field = f"S: {truncated_sentence} Q: {truncated_question} entailment?"
            prompt = "Question: " + field + option_suffix
            prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
            answer_ids = tokenizer.encode(answer_probe, add_special_tokens=False)
            if len(prompt_ids) + len(answer_ids) + 1 <= seq_len:
                return normalize_text(field)
            keep_sentence -= 1
        keep_question -= 1
        keep_sentence = min(len(sentence_ids), max(0, available_tokens - keep_question))

    raise RuntimeError("Failed to fit QNLI sample within seq_len after truncation")


def convert_split(rows, output_path: Path, *, tokenizer, seq_len: int, include_labels: bool) -> int:
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
            sentence = normalize_text(str(row["sentence"]))
            question = normalize_text(str(row["question"]))
            question_field = fit_question_field(tokenizer, sentence, question, seq_len=seq_len)
            answer = "A" if (include_labels and int(row["label"]) == 0) else "B"
            if not include_labels:
                answer = "A"
            writer.writerow(
                {
                    "question": question_field,
                    "A": "yes",
                    "B": "no",
                    "C": ".",
                    "D": ".",
                    "answer": answer,
                }
            )
            count += 1
    return count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    dataset = load_dataset("glue", "qnli")

    train_path = output_dir / "qnli_train_mmlu.csv"
    val_path = output_dir / "qnli_validation_mmlu.csv"
    test_path = output_dir / "qnli_test_mmlu.csv"

    train_count = convert_split(dataset["train"], train_path, tokenizer=tokenizer, seq_len=args.seq_len, include_labels=True)
    val_count = convert_split(dataset["validation"], val_path, tokenizer=tokenizer, seq_len=args.seq_len, include_labels=True)
    test_count = convert_split(dataset["validation"], test_path, tokenizer=tokenizer, seq_len=args.seq_len, include_labels=True)

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
