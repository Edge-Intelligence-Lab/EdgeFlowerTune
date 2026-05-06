from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from datasets import load_dataset
from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Winogrande locally and convert it to mmlu_csv-compatible files")
    parser.add_argument("--output-dir", default="data/winogrande", help="Output directory for converted CSV files")
    parser.add_argument("--raw-dir", default="data/winogrande/raw", help="Directory to cache raw Winogrande files")
    parser.add_argument(
        "--model-dir",
        default="${EDGEFLOWERTUNE_ROOT}/pretrained_models/gemma-3-270m",
        help="Local model/tokenizer directory",
    )
    parser.add_argument("--seq-len", type=int, default=64, help="Maximum downstream sequence length")
    parser.add_argument("--overwrite", action="store_true", help="Force re-download/re-conversion")
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


def ensure_placeholder(sentence: str) -> str:
    if "_" not in sentence:
        raise ValueError(f"Winogrande sentence missing placeholder _: {sentence}")
    return sentence


def fit_fields(
    tokenizer: Any,
    sentence: str,
    option1: str,
    option2: str,
    *,
    seq_len: int,
) -> tuple[str, str, str]:
    option_c = "."
    option_d = "."
    answer_probe = " A"
    sentence = ensure_placeholder(sentence)
    left, right = sentence.split("_", 1)
    left_ids = encode_text(tokenizer, normalize_text(left))
    right_ids = encode_text(tokenizer, normalize_text(right))
    opt1_ids = encode_text(tokenizer, normalize_text(option1))
    opt2_ids = encode_text(tokenizer, normalize_text(option2))

    keep_left = len(left_ids)
    keep_right = len(right_ids)
    keep_opt1 = len(opt1_ids)
    keep_opt2 = len(opt2_ids)

    while keep_left >= 0 and keep_right >= 0 and keep_opt1 >= 0 and keep_opt2 >= 0:
        left_text = decode_prefix(tokenizer, left_ids, keep_left)
        right_text = decode_prefix(tokenizer, right_ids, keep_right)
        opt1_text = decode_prefix(tokenizer, opt1_ids, keep_opt1)
        opt2_text = decode_prefix(tokenizer, opt2_ids, keep_opt2)
        question_field = normalize_text(f"S: {left_text} _ {right_text}")
        prompt = (
            "Question: "
            + question_field
            + "\nA. "
            + normalize_text(opt1_text)
            + "\nB. "
            + normalize_text(opt2_text)
            + "\nC. "
            + option_c
            + "\nD. "
            + option_d
            + "\nAnswer: "
        )
        prompt_ids = encode_text(tokenizer, prompt)
        answer_ids = encode_text(tokenizer, answer_probe)
        if len(prompt_ids) + len(answer_ids) + 1 <= seq_len:
            return question_field, normalize_text(opt1_text), normalize_text(opt2_text)

        largest = max(
            (keep_left, "left"),
            (keep_right, "right"),
            (keep_opt1, "opt1"),
            (keep_opt2, "opt2"),
            key=lambda item: item[0],
        )[1]
        if largest == "left" and keep_left > 0:
            keep_left -= 1
        elif largest == "right" and keep_right > 0:
            keep_right -= 1
        elif largest == "opt1" and keep_opt1 > 0:
            keep_opt1 -= 1
        elif keep_opt2 > 0:
            keep_opt2 -= 1
        else:
            break

    raise RuntimeError("Failed to fit Winogrande sample within seq_len after truncation")


def dataset_to_rows(ds_split) -> list[tuple[str, str, str, str]]:
    rows: list[tuple[str, str, str, str]] = []
    for row in ds_split:
        answer = str(row.get("answer", "")).strip()
        if answer not in {"1", "2"}:
            continue
        rows.append(
            (
                normalize_text(str(row["sentence"])),
                normalize_text(str(row["option1"])),
                normalize_text(str(row["option2"])),
                answer,
            )
        )
    return rows


def convert_split(rows: list[tuple[str, str, str, str]], output_path: Path, *, tokenizer, seq_len: int) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "A", "B", "C", "D", "answer"],
            lineterminator="\n",
        )
        writer.writeheader()
        for sentence, option1, option2, answer in rows:
            question_field, option_a, option_b = fit_fields(
                tokenizer,
                sentence,
                option1,
                option2,
                seq_len=seq_len,
            )
            writer.writerow(
                {
                    "question": question_field,
                    "A": option_a,
                    "B": option_b,
                    "C": ".",
                    "D": ".",
                    "answer": "A" if answer == "1" else "B",
                }
            )
            count += 1
    return count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "winogrande_train_mmlu.csv"
    val_path = output_dir / "winogrande_validation_mmlu.csv"
    test_path = output_dir / "winogrande_test_mmlu.csv"

    if not args.overwrite and train_path.is_file() and val_path.is_file() and test_path.is_file():
        print(f"train_csv={train_path}")
        print(f"validation_csv={val_path}")
        print(f"test_csv={test_path}")
        print("reused_existing=1")
        print(f"raw_dir={raw_dir}")
        print(f"model_dir={model_dir}")
        print(f"seq_len={args.seq_len}")
        return

    if not model_dir.is_dir():
        raise RuntimeError(f"Missing model/tokenizer directory: {model_dir}")

    tokenizer_model = model_dir / "tokenizer.model"
    if tokenizer_model.is_file():
        tokenizer = SentencePieceProcessor(model_file=str(tokenizer_model))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)
    dataset = load_dataset("allenai/winogrande", "winogrande_xl", cache_dir=str(raw_dir))

    train_rows = dataset_to_rows(dataset["train"])
    val_rows = dataset_to_rows(dataset["validation"])
    if "test" in dataset:
        test_rows = dataset_to_rows(dataset["test"])
        if not test_rows:
            test_rows = val_rows
    else:
        test_rows = val_rows

    train_count = convert_split(train_rows, train_path, tokenizer=tokenizer, seq_len=args.seq_len)
    val_count = convert_split(val_rows, val_path, tokenizer=tokenizer, seq_len=args.seq_len)
    test_count = convert_split(test_rows, test_path, tokenizer=tokenizer, seq_len=args.seq_len)

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
