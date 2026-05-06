from __future__ import annotations

import argparse
import csv
import json
import urllib.request
import zipfile
from pathlib import Path
from typing import Any

from sentencepiece import SentencePieceProcessor
from transformers import AutoTokenizer


SOCIALIQA_URL = "https://storage.googleapis.com/ai2-mosaic/public/socialiqa/socialiqa-train-dev.zip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download Social IQa locally and convert it to mmlu_csv-compatible files")
    parser.add_argument("--output-dir", default="data/socialqa", help="Output directory for converted CSV files")
    parser.add_argument(
        "--raw-dir",
        default="data/socialqa/raw/socialiqa-train-dev",
        help="Directory containing Social IQa raw jsonl/label files",
    )
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


def ensure_raw_data(raw_dir: Path, *, overwrite: bool) -> None:
    train_jsonl = raw_dir / "train.jsonl"
    train_labels = raw_dir / "train-labels.lst"
    dev_jsonl = raw_dir / "dev.jsonl"
    dev_labels = raw_dir / "dev-labels.lst"
    if not overwrite and all(p.is_file() for p in (train_jsonl, train_labels, dev_jsonl, dev_labels)):
        return

    raw_root = raw_dir.parent
    raw_root.mkdir(parents=True, exist_ok=True)
    zip_path = raw_root / "socialiqa-train-dev.zip"
    urllib.request.urlretrieve(SOCIALIQA_URL, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_root)


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
    context: str,
    question: str,
    answer_a: str,
    answer_b: str,
    answer_c: str,
    *,
    seq_len: int,
) -> tuple[str, str, str, str]:
    option_d = "."
    answer_probe = " A"
    context_ids = encode_text(tokenizer, context)
    question_ids = encode_text(tokenizer, question)
    answer_a_ids = encode_text(tokenizer, answer_a)
    answer_b_ids = encode_text(tokenizer, answer_b)
    answer_c_ids = encode_text(tokenizer, answer_c)

    keep_context = min(len(context_ids), 20)
    keep_question = min(len(question_ids), 10)
    keep_a = min(len(answer_a_ids), 8)
    keep_b = min(len(answer_b_ids), 8)
    keep_c = min(len(answer_c_ids), 8)

    while keep_context >= 0 and keep_question >= 0 and keep_a >= 0 and keep_b >= 0 and keep_c >= 0:
        context_text = decode_prefix(tokenizer, context_ids, keep_context)
        question_text = decode_prefix(tokenizer, question_ids, keep_question)
        option_a = normalize_text(decode_prefix(tokenizer, answer_a_ids, keep_a))
        option_b = normalize_text(decode_prefix(tokenizer, answer_b_ids, keep_b))
        option_c = normalize_text(decode_prefix(tokenizer, answer_c_ids, keep_c))
        question_field = normalize_text(f"C: {context_text} Q: {question_text}")
        prompt = (
            "Question: "
            + question_field
            + "\nA. "
            + option_a
            + "\nB. "
            + option_b
            + "\nC. "
            + option_c
            + "\nD. "
            + option_d
            + "\nAnswer: "
        )
        prompt_ids = encode_text(tokenizer, prompt)
        answer_ids = encode_text(tokenizer, answer_probe)
        if len(prompt_ids) + len(answer_ids) + 1 <= seq_len:
            return question_field, option_a, option_b, option_c

        largest = max(
            [
                (keep_context, "context"),
                (keep_question, "question"),
                (keep_a, "a"),
                (keep_b, "b"),
                (keep_c, "c"),
            ],
            key=lambda item: item[0],
        )[1]
        if largest == "context" and keep_context > 0:
            keep_context -= 1
        elif largest == "question" and keep_question > 0:
            keep_question -= 1
        elif largest == "a" and keep_a > 0:
            keep_a -= 1
        elif largest == "b" and keep_b > 0:
            keep_b -= 1
        elif keep_c > 0:
            keep_c -= 1
        else:
            break

    raise RuntimeError("Failed to fit Social IQa sample within seq_len after truncation")


def read_rows(jsonl_path: Path, labels_path: Path) -> list[tuple[str, str, str, str, str, str]]:
    rows = [json.loads(line) for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = [line.strip() for line in labels_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != len(labels):
        raise ValueError(f"Mismatched Social IQa rows/labels: {jsonl_path} vs {labels_path}")
    result: list[tuple[str, str, str, str, str, str]] = []
    for row, label in zip(rows, labels):
        result.append(
            (
                normalize_text(str(row["context"])),
                normalize_text(str(row["question"])),
                normalize_text(str(row["answerA"])),
                normalize_text(str(row["answerB"])),
                normalize_text(str(row["answerC"])),
                label,
            )
        )
    return result


def convert_split(
    rows: list[tuple[str, str, str, str, str, str]],
    output_path: Path,
    *,
    tokenizer: Any,
    seq_len: int,
) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["question", "A", "B", "C", "D", "answer"],
            lineterminator="\n",
        )
        writer.writeheader()
        for context, question, answer_a, answer_b, answer_c, label in rows:
            question_field, option_a, option_b, option_c = fit_fields(
                tokenizer,
                context,
                question,
                answer_a,
                answer_b,
                answer_c,
                seq_len=seq_len,
            )
            label_map = {"1": "A", "2": "B", "3": "C"}
            writer.writerow(
                {
                    "question": question_field,
                    "A": option_a,
                    "B": option_b,
                    "C": option_c,
                    "D": ".",
                    "answer": label_map[label],
                }
            )
            count += 1
    return count


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).resolve()
    raw_dir = Path(args.raw_dir).resolve()
    model_dir = Path(args.model_dir).expanduser().resolve()
    train_path = output_dir / "socialqa_train_mmlu.csv"
    val_path = output_dir / "socialqa_validation_mmlu.csv"
    test_path = output_dir / "socialqa_test_mmlu.csv"

    if not args.overwrite and train_path.is_file() and val_path.is_file() and test_path.is_file():
        print(f"train_csv={train_path}")
        print(f"validation_csv={val_path}")
        print(f"test_csv={test_path}")
        return

    if not model_dir.is_dir():
        raise RuntimeError(f"Missing model/tokenizer directory: {model_dir}")

    ensure_raw_data(raw_dir, overwrite=args.overwrite)
    tokenizer_model = model_dir / "tokenizer.model"
    if tokenizer_model.is_file():
        tokenizer: Any = SentencePieceProcessor(model_file=str(tokenizer_model))
    else:
        tokenizer = AutoTokenizer.from_pretrained(str(model_dir), use_fast=True)

    train_rows = read_rows(raw_dir / "train.jsonl", raw_dir / "train-labels.lst")
    val_rows = read_rows(raw_dir / "dev.jsonl", raw_dir / "dev-labels.lst")

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
