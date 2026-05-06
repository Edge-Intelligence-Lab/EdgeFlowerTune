from __future__ import annotations

import argparse
import csv
from pathlib import Path

from datasets import load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download real MMLU and export to a flat CSV for Nano clients")
    parser.add_argument("--output", required=True, help="Output CSV path")
    parser.add_argument("--split", default="test", help="Dataset split, e.g. test/dev/validation")
    parser.add_argument("--limit", type=int, default=0, help="Optional max rows to export; 0 means all")
    parser.add_argument("--dataset-id", default="cais/mmlu")
    parser.add_argument("--config-name", default="all")
    parser.add_argument("--cache-dir", default="", help="Optional datasets cache directory")
    parser.add_argument(
        "--local-csv-root",
        default="",
        help="Optional local root containing official MMLU CSV splits, e.g. .../data",
    )
    parser.add_argument("--shuffle-seed", type=int, default=7)
    parser.add_argument(
        "--trust-remote-code",
        action="store_true",
        help="Pass trust_remote_code=True when the dataset loader requires it",
    )
    return parser.parse_args()


def answer_to_letter(value) -> str:
    if isinstance(value, int):
        assert 0 <= value <= 3, f"Integer answer out of range: {value}"
        return "ABCD"[value]
    text = str(value).strip().upper()
    if text in {"0", "1", "2", "3"}:
        return "ABCD"[int(text)]
    assert text in {"A", "B", "C", "D"}, f"Unsupported answer label: {value!r}"
    return text


def extract_choices(example: dict) -> list[str]:
    if "choices" in example:
        choices = list(example["choices"])
    else:
        choices = [example[key] for key in ("A", "B", "C", "D")]
    assert len(choices) == 4, f"Expected 4 choices, got {len(choices)}"
    return [str(choice) for choice in choices]


def export_local_csv_root(output_path: Path, local_csv_root: Path, split: str, limit: int) -> int:
    split_dir = local_csv_root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"Missing split directory under local-csv-root: {split_dir}")

    rows_written = 0
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "question", "A", "B", "C", "D", "answer", "subject"],
        )
        writer.writeheader()
        for csv_path in sorted(split_dir.glob("*.csv")):
            subject = csv_path.stem.rsplit("_", 1)[0]
            with open(csv_path, "r", encoding="utf-8") as src:
                reader = csv.reader(src)
                for idx, row in enumerate(reader):
                    if len(row) < 6:
                        continue
                    question, a, b, c, d, answer = row[:6]
                    writer.writerow(
                        {
                            "sample_id": f"{subject}-{idx}",
                            "question": question,
                            "A": a,
                            "B": b,
                            "C": c,
                            "D": d,
                            "answer": answer_to_letter(answer),
                            "subject": subject,
                        }
                    )
                    rows_written += 1
                    if limit > 0 and rows_written >= limit:
                        return rows_written
    return rows_written


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.local_csv_root:
        rows_written = export_local_csv_root(
            output_path=output_path,
            local_csv_root=Path(args.local_csv_root).resolve(),
            split=args.split,
            limit=args.limit,
        )
        print(f"output={output_path}")
        print(f"rows={rows_written}")
        return

    dataset = load_dataset(
        args.dataset_id,
        args.config_name,
        split=args.split,
        cache_dir=args.cache_dir or None,
        trust_remote_code=args.trust_remote_code,
    )
    if args.shuffle_seed is not None:
        dataset = dataset.shuffle(seed=args.shuffle_seed)
    if args.limit > 0:
        dataset = dataset.select(range(min(args.limit, len(dataset))))

    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["sample_id", "question", "A", "B", "C", "D", "answer", "subject"],
        )
        writer.writeheader()
        for idx, example in enumerate(dataset):
            choices = extract_choices(example)
            writer.writerow(
                {
                    "sample_id": example.get("id", example.get("sample_id", f"mmlu-{idx}")),
                    "question": str(example["question"]),
                    "A": choices[0],
                    "B": choices[1],
                    "C": choices[2],
                    "D": choices[3],
                    "answer": answer_to_letter(example["answer"]),
                    "subject": str(example.get("subject", args.config_name)),
                }
            )

    print(f"output={output_path}")
    print(f"rows={len(dataset)}")


if __name__ == "__main__":
    main()
