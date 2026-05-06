from __future__ import annotations

import csv
import json
import logging
import time
from pathlib import Path
from typing import Iterable


def configure_logging(output_dir: str | Path, name: str = "lshaped") -> logging.Logger:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / f"{name}.log", encoding="utf-8")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream)
    logger.addHandler(file_handler)
    return logger


def append_jsonl(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def append_csv(path: str | Path, row: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row.keys())
    file_exists = path.exists()
    if file_exists:
        with open(path, "r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, [])
        if existing_header and existing_header != fieldnames:
            rotated = path.with_name(f"{path.stem}.schema_mismatch_{int(time.time())}{path.suffix}")
            path.replace(rotated)
            file_exists = False
    with open(path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def mean_of(metrics: Iterable[dict[str, float]]) -> dict[str, float]:
    metrics = list(metrics)
    if not metrics:
        return {}
    keys = sorted({k for item in metrics for k in item.keys()})
    out: dict[str, float] = {}
    for key in keys:
        vals = [float(item[key]) for item in metrics if key in item]
        if vals:
            out[key] = sum(vals) / len(vals)
    return out
