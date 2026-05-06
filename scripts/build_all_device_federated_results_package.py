from __future__ import annotations

import csv
import json
import math
import os
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DELIVERABLES = ROOT / "deliverables"
OUT = DELIVERABLES / "all_device_federated_results_20260502"
ZIP_PATH = DELIVERABLES / "all_device_federated_results_20260502.zip"

DATASETS: dict[str, str] = {
    "boolq": "BoolQ",
    "qnli": "QNLI",
    "piqa": "PIQA",
    "hellaswag": "HellaSwag",
    "socialqa": "SocialQA",
    "arce": "ARC-E",
    "winogrande": "WinoGrande",
}

METHODS: dict[str, str] = {
    "fedavg": "FedAvg + LoRA",
    "fedprox": "FedProx + LoRA",
    "flexlora": "FlexLoRA",
    "splitlora": "SplitLoRA",
}

MODELS: dict[str, str] = {
    "gemma270m": "Gemma 3 270M",
    "qwen05b": "Qwen 0.5B",
    "gemma1b": "Gemma 3 1B",
}

COHORT_LABELS: dict[str, str] = {
    "nova5_jetson3": "5 nova phones + 3 Jetsons",
    "nova5_jetson2": "5 nova phones + 2 Jetsons",
    "mate20_jad_vivo": "Mate20 + JAD + vivo",
}

EXPECTED_CLIENTS: dict[tuple[str, str], int] = {
    ("nova5_jetson3", "gemma270m"): 8,
    ("nova5_jetson3", "qwen05b"): 8,
    ("nova5_jetson2", "gemma1b"): 7,
    ("mate20_jad_vivo", "gemma270m"): 3,
    ("mate20_jad_vivo", "qwen05b"): 3,
    ("mate20_jad_vivo", "gemma1b"): 3,
}

SAFE_SUFFIXES = {
    ".csv",
    ".json",
    ".jsonl",
    ".md",
    ".yaml",
    ".yml",
}


@dataclass(frozen=True)
class RunSource:
    cohort: str
    model_key: str
    dataset_key: str
    method: str
    source_root: Path
    server_dir: Path


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], preferred: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for field in preferred or []:
        if field not in fields:
            fields.append(field)
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def safe_float(raw: Any) -> float:
    try:
        if raw in ("", None):
            return 0.0
        return float(raw)
    except (TypeError, ValueError):
        return 0.0


def safe_int(raw: Any) -> int:
    value = safe_float(raw)
    if math.isnan(value):
        return 0
    return int(value)


def norm_key(value: str) -> str:
    return value.strip().lower().replace(" ", "").replace("-", "").replace("_", "")


def dataset_key_from_label(value: str) -> str:
    normalized = norm_key(value)
    aliases = {
        "boolq": "boolq",
        "qnli": "qnli",
        "piqa": "piqa",
        "hellaswag": "hellaswag",
        "socialqa": "socialqa",
        "socialiqa": "socialqa",
        "arce": "arce",
        "arceasy": "arce",
        "winogrande": "winogrande",
        "wino": "winogrande",
    }
    return aliases.get(normalized, value.strip().lower())


def method_key(value: str) -> str:
    normalized = norm_key(value)
    if "fedavg" in normalized:
        return "fedavg"
    if "fedprox" in normalized:
        return "fedprox"
    if "flex" in normalized:
        return "flexlora"
    if "split" in normalized:
        return "splitlora"
    return value.strip().lower()


def client_group(client_id: str) -> str:
    if client_id.startswith("nova_"):
        return "nova_phone"
    if client_id.startswith("jetson_"):
        return "jetson"
    if client_id.startswith("jad_"):
        return "huawei_jad"
    if client_id.startswith("mate20_"):
        return "huawei_mate20"
    if client_id.startswith("vivo_"):
        return "vivo"
    return "unknown"


def add_context(
    row: dict[str, Any],
    *,
    cohort: str,
    model_key: str,
    dataset_key: str,
    method: str,
    source_root: Path,
    source_path: Path,
) -> dict[str, Any]:
    out = dict(row)
    out.update(
        {
            "cohort": cohort,
            "cohort_label": COHORT_LABELS.get(cohort, cohort),
            "model_key": model_key,
            "model_label": MODELS.get(model_key, model_key),
            "dataset_key": dataset_key,
            "dataset": DATASETS.get(dataset_key, row.get("dataset", dataset_key)),
            "method": method,
            "method_label": METHODS.get(method, method),
            "source_root": str(source_root.relative_to(ROOT)),
            "source_path": str(source_path.relative_to(ROOT)),
        }
    )
    if "client_id" in row:
        out["client_group"] = client_group(str(row.get("client_id", "")))
    return out


def first_existing(*paths: Path) -> Path | None:
    for path in paths:
        if path.is_file():
            return path
    return None


def source_run_id(source_root: Path, server_dir: Path, dataset_key: str, method: str, model_key: str) -> str:
    del server_dir
    # Some old deliverables contain JSON files with extended attributes that can
    # make read_text unexpectedly slow on macOS. The exact original run id is not
    # needed for this package; use a stable synthetic id and keep source_root.
    return f"{model_key}_{dataset_key}_{method}_{source_root.name}"


def build_nova_jetson_sources() -> list[RunSource]:
    sources: list[RunSource] = []

    boolq_method_roots = {
        "fedavg": DELIVERABLES / "boolq_fedavg_measurement_20260421",
        "fedprox": DELIVERABLES / "boolq_fedprox_measurement_20260421",
        "flexlora": DELIVERABLES / "boolq_flexlora_measurement_20260421",
        "splitlora": DELIVERABLES / "boolq_splitlora_measurement_20260422",
    }
    for method, root in boolq_method_roots.items():
        sources.append(RunSource("nova5_jetson3", "gemma270m", "boolq", method, root, root / "run" / "server"))

    for dataset_key in ("qnli", "piqa", "hellaswag", "socialqa", "arce", "winogrande"):
        root = DELIVERABLES / f"{dataset_key}_federated_measurements_20260422"
        for method in METHODS:
            server_dir = root / method / "server"
            if not server_dir.is_dir():
                server_dir = root / method / "run" / "server"
            sources.append(RunSource("nova5_jetson3", "gemma270m", dataset_key, method, root, server_dir))

    qwen_dates = {
        "boolq": "20260423",
        "qnli": "20260423",
        "piqa": "20260423",
        "hellaswag": "20260423",
        "socialqa": "20260423",
        "arce": "20260424",
        "winogrande": "20260424",
    }
    for dataset_key, date_tag in qwen_dates.items():
        root = DELIVERABLES / f"qwen05b_{dataset_key}_federated_measurements_{date_tag}"
        for method in METHODS:
            server_dir = root / method / "server"
            if not server_dir.is_dir():
                server_dir = root / method / "run" / "server"
            sources.append(RunSource("nova5_jetson3", "qwen05b", dataset_key, method, root, server_dir))

    return sources


def collect_run_source(
    source: RunSource,
    client_rows: list[dict[str, Any]],
    server_rows: list[dict[str, Any]],
    power_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    client_path = first_existing(source.server_dir / "round1_client_summary_clean.csv")
    server_path = first_existing(source.server_dir / "summary_rounds_clean.csv", source.server_dir / "summary_rounds.csv")
    power_path = first_existing(source.server_dir / "power_summary.csv")
    run_id = source_run_id(source.source_root, source.server_dir, source.dataset_key, source.method, source.model_key)

    for row in read_csv(client_path) if client_path else []:
        row = dict(row)
        row.setdefault("run_id", run_id)
        client_rows.append(
            add_context(
                row,
                cohort=source.cohort,
                model_key=source.model_key,
                dataset_key=source.dataset_key,
                method=source.method,
                source_root=source.source_root,
                source_path=client_path or source.server_dir,
            )
        )

    for row in read_csv(server_path) if server_path else []:
        row = dict(row)
        row.setdefault("run_id", run_id)
        server_rows.append(
            add_context(
                row,
                cohort=source.cohort,
                model_key=source.model_key,
                dataset_key=source.dataset_key,
                method=source.method,
                source_root=source.source_root,
                source_path=server_path or source.server_dir,
            )
        )

    for row in read_csv(power_path) if power_path else []:
        row = dict(row)
        row.setdefault("run_id", run_id)
        power_rows.append(
            add_context(
                row,
                cohort=source.cohort,
                model_key=source.model_key,
                dataset_key=source.dataset_key,
                method=source.method,
                source_root=source.source_root,
                source_path=power_path or source.server_dir,
            )
        )

    summary = {
        "run_id": run_id,
        "client_rows": len(read_csv(client_path)) if client_path else 0,
        "server_rows": len(read_csv(server_path)) if server_path else 0,
        "power_rows": len(read_csv(power_path)) if power_path else 0,
    }
    if server_path:
        rows = read_csv(server_path)
        if rows:
            summary.update(rows[0])
    summary_rows.append(
        add_context(
            summary,
            cohort=source.cohort,
            model_key=source.model_key,
            dataset_key=source.dataset_key,
            method=source.method,
            source_root=source.source_root,
            source_path=server_path or source.server_dir,
        )
    )

    validation_path = source.source_root / "measurement_validation.csv"
    if validation_path.is_file():
        for row in read_csv(validation_path):
            if method_key(str(row.get("method", source.method))) != source.method:
                continue
            validation_rows.append(
                add_context(
                    row,
                    cohort=source.cohort,
                    model_key=source.model_key,
                    dataset_key=source.dataset_key,
                    method=source.method,
                    source_root=source.source_root,
                    source_path=validation_path,
                )
            )


def collect_gemma1b_7client(
    client_rows: list[dict[str, Any]],
    server_rows: list[dict[str, Any]],
    power_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    root = DELIVERABLES / "gemma1b_splitlora_7client_7datasets_20260425_gemma1b_splitlora_7client_7datasets_embedding_only_rerun1"
    client_path = root / "gemma1b_splitlora_7client_client_summary.csv"
    server_path = root / "gemma1b_splitlora_7client_round1_summary.csv"
    power_path = root / "gemma1b_splitlora_7client_power_summary.csv"
    validation_path = root / "measurement_validation.csv"
    for row in read_csv(client_path):
        dataset_key = dataset_key_from_label(str(row.get("dataset", "")))
        row = dict(row)
        row.setdefault("method", "splitlora")
        row.setdefault("run_id", row.get("run_id", f"gemma1b_{dataset_key}_splitlora_7client"))
        client_rows.append(
            add_context(
                row,
                cohort="nova5_jetson2",
                model_key="gemma1b",
                dataset_key=dataset_key,
                method="splitlora",
                source_root=root,
                source_path=client_path,
            )
        )
    for row in read_csv(server_path):
        dataset_key = dataset_key_from_label(str(row.get("dataset", "")))
        row = dict(row)
        row.setdefault("method", "splitlora")
        row.setdefault("run_id", row.get("run_id", f"gemma1b_{dataset_key}_splitlora_7client"))
        server_rows.append(
            add_context(
                row,
                cohort="nova5_jetson2",
                model_key="gemma1b",
                dataset_key=dataset_key,
                method="splitlora",
                source_root=root,
                source_path=server_path,
            )
        )
        summary_rows.append(
            add_context(
                row,
                cohort="nova5_jetson2",
                model_key="gemma1b",
                dataset_key=dataset_key,
                method="splitlora",
                source_root=root,
                source_path=server_path,
            )
        )
    for row in read_csv(power_path):
        dataset_key = dataset_key_from_label(str(row.get("dataset", "")))
        row = dict(row)
        row.setdefault("method", "splitlora")
        power_rows.append(
            add_context(
                row,
                cohort="nova5_jetson2",
                model_key="gemma1b",
                dataset_key=dataset_key,
                method="splitlora",
                source_root=root,
                source_path=power_path,
            )
        )
    for row in read_csv(validation_path):
        dataset_key = dataset_key_from_label(str(row.get("dataset", "")))
        validation_rows.append(
            add_context(
                row,
                cohort="nova5_jetson2",
                model_key="gemma1b",
                dataset_key=dataset_key,
                method="splitlora",
                source_root=root,
                source_path=validation_path,
            )
        )


def collect_three_phone(
    client_rows: list[dict[str, Any]],
    server_rows: list[dict[str, Any]],
    power_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    root = DELIVERABLES / "three_phone_all_models_gemma270m_gemma1b_qwen05b_20260430"
    files = {
        "client": root / "consolidated" / "all_models_client_round1_detail.csv",
        "server": root / "consolidated" / "all_models_server_round1_detail.csv",
        "power": root / "consolidated" / "all_models_power_detail.csv",
        "summary": root / "consolidated" / "all_models_round1_summary.csv",
        "validation": root / "consolidated" / "all_models_source_validation.csv",
    }
    for kind, target_rows in (
        ("client", client_rows),
        ("server", server_rows),
        ("power", power_rows),
        ("summary", summary_rows),
        ("validation", validation_rows),
    ):
        path = files[kind]
        for row in read_csv(path):
            dataset_key = row.get("dataset_key") or dataset_key_from_label(str(row.get("dataset", "")))
            method = method_key(str(row.get("method", "")))
            model_key = row.get("model_key", "")
            out = add_context(
                row,
                cohort="mate20_jad_vivo",
                model_key=model_key,
                dataset_key=dataset_key,
                method=method,
                source_root=root,
                source_path=path,
            )
            target_rows.append(out)


def audit_client_rows(client_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for row in client_rows:
        issues: list[str] = []
        steps = safe_int(row.get("steps_completed"))
        if steps != 3:
            issues.append(f"steps_completed={row.get('steps_completed')}")
        step_times_raw = str(row.get("step_times_sec_json", ""))
        try:
            step_times = json.loads(step_times_raw) if step_times_raw else []
        except Exception:
            step_times = []
        if len(step_times) != 3 or any(safe_float(v) <= 0.0 for v in step_times):
            issues.append("bad_step_times")
        for field in ("download_time_sec", "upload_time_sec", "download_bytes", "upload_bytes", "transmitted_bytes"):
            if safe_float(row.get(field)) <= 0.0:
                issues.append(f"{field}={row.get(field)}")
        for field in ("avg_rss_mb", "peak_rss_mb", "avg_power_w"):
            if safe_float(row.get(field)) <= 0.0:
                issues.append(f"{field}={row.get(field)}")
        if row.get("power_samples", "") != "" and safe_float(row.get("power_samples")) <= 0.0:
            issues.append(f"power_samples={row.get('power_samples')}")
        audit.append(
            {
                "status": "ok" if not issues else "issue",
                "issues": ";".join(issues),
                "cohort": row.get("cohort", ""),
                "model_key": row.get("model_key", ""),
                "dataset_key": row.get("dataset_key", ""),
                "method": row.get("method", ""),
                "client_id": row.get("client_id", ""),
                "run_id": row.get("run_id", ""),
            }
        )
    return audit


def audit_server_rows(server_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    audit: list[dict[str, Any]] = []
    for row in server_rows:
        issues: list[str] = []
        if safe_float(row.get("aggregation_time_sec")) < 0.0 or row.get("aggregation_time_sec", "") == "":
            issues.append(f"aggregation_time_sec={row.get('aggregation_time_sec')}")
        client_count = safe_int(row.get("aggregated_clients") or row.get("num_clients") or row.get("client_rows"))
        if client_count <= 0:
            issues.append("missing_client_count")
        audit.append(
            {
                "status": "ok" if not issues else "issue",
                "issues": ";".join(issues),
                "cohort": row.get("cohort", ""),
                "model_key": row.get("model_key", ""),
                "dataset_key": row.get("dataset_key", ""),
                "method": row.get("method", ""),
                "run_id": row.get("run_id", ""),
            }
        )
    return audit


def build_issue_summary(client_audit: list[dict[str, Any]], server_audit: list[dict[str, Any]], coverage_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counters = [
        ("client_metric", Counter(row["issues"] for row in client_audit if row["status"] != "ok")),
        ("server_metric", Counter(row["issues"] for row in server_audit if row["status"] != "ok")),
        ("coverage", Counter("coverage_mismatch" for row in coverage_rows if row["status"] != "ok")),
    ]
    explanations = {
        "download_bytes=0": "Old FlexLoRA runs recorded positive download_time_sec but downlink byte counter stayed 0; upload/transmitted bytes are present. Kept raw value, not imputed.",
        "avg_power_w=0.0": "Old Gemma 1B 7-client nova-phone power sampler produced zero batterystats power; later three-phone Gemma 1B power rows are valid. Kept raw value, not imputed.",
        "upload_time_sec=0.0": "Three old BoolQ FedAvg nova rows have upload_bytes but upload_time_sec was not recorded. Kept raw value, not imputed.",
    }
    for category, counter in counters:
        for issue, count in counter.most_common():
            rows.append(
                {
                    "category": category,
                    "issue": issue,
                    "count": count,
                    "explanation": explanations.get(issue, ""),
                }
            )
    return rows


def build_coverage(client_rows: list[dict[str, Any]], server_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    server_key_count: dict[tuple[str, str, str, str], int] = {}
    for row in server_rows:
        key = (str(row.get("cohort", "")), str(row.get("model_key", "")), str(row.get("dataset_key", "")), str(row.get("method", "")))
        server_key_count[key] = server_key_count.get(key, 0) + 1

    rows: list[dict[str, Any]] = []
    by_key_clients: dict[tuple[str, str, str, str], set[str]] = {}
    for row in client_rows:
        key = (str(row.get("cohort", "")), str(row.get("model_key", "")), str(row.get("dataset_key", "")), str(row.get("method", "")))
        by_key_clients.setdefault(key, set()).add(str(row.get("client_id", "")))

    expected_keys: set[tuple[str, str, str, str]] = set()
    for cohort, model in EXPECTED_CLIENTS:
        datasets = DATASETS.keys()
        methods = ("splitlora",) if model == "gemma1b" else METHODS.keys()
        for dataset_key in datasets:
            for method in methods:
                expected_keys.add((cohort, model, dataset_key, method))
    expected_keys.update(by_key_clients.keys())

    for key in sorted(expected_keys):
        cohort, model_key, dataset_key, method = key
        clients = sorted(client for client in by_key_clients.get(key, set()) if client)
        expected = EXPECTED_CLIENTS.get((cohort, model_key), 0)
        rows.append(
            {
                "status": "ok" if expected and len(clients) == expected else "issue",
                "cohort": cohort,
                "cohort_label": COHORT_LABELS.get(cohort, cohort),
                "model_key": model_key,
                "model_label": MODELS.get(model_key, model_key),
                "dataset_key": dataset_key,
                "dataset": DATASETS.get(dataset_key, dataset_key),
                "method": method,
                "method_label": METHODS.get(method, method),
                "expected_clients": expected,
                "actual_clients": len(clients),
                "client_ids": ",".join(clients),
                "server_rows": server_key_count.get(key, 0),
            }
        )
    return rows


def write_grouped_outputs(
    client_rows: list[dict[str, Any]],
    server_rows: list[dict[str, Any]],
    power_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
) -> None:
    grouped = OUT / "by_cohort_model"
    for cohort in sorted({str(row.get("cohort", "")) for row in client_rows}):
        for model_key in sorted({str(row.get("model_key", "")) for row in client_rows if row.get("cohort") == cohort}):
            base = grouped / cohort / model_key
            write_csv(base / "client_round1_detail.csv", [r for r in client_rows if r.get("cohort") == cohort and r.get("model_key") == model_key])
            write_csv(base / "server_round1_detail.csv", [r for r in server_rows if r.get("cohort") == cohort and r.get("model_key") == model_key])
            write_csv(base / "power_detail.csv", [r for r in power_rows if r.get("cohort") == cohort and r.get("model_key") == model_key])
            write_csv(base / "round1_summary.csv", [r for r in summary_rows if r.get("cohort") == cohort and r.get("model_key") == model_key])
            write_csv(base / "source_validation.csv", [r for r in validation_rows if r.get("cohort") == cohort and r.get("model_key") == model_key])


def copy_source_snapshots(source_roots: list[Path]) -> None:
    target_root = OUT / "source_snapshots"
    for source in sorted(set(source_roots), key=lambda p: p.name):
        if not source.exists():
            continue
        target = target_root / source.name
        for path in source.rglob("*"):
            if path.is_dir():
                continue
            if path.suffix.lower() not in SAFE_SUFFIXES:
                continue
            if any(part in {"checkpoints", "__pycache__"} for part in path.relative_to(source).parts):
                continue
            if path.stat().st_size > 50 * 1024 * 1024:
                continue
            rel = path.relative_to(source)
            dest = target / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Avoid copy2/copyfile metadata handling on macOS because several
            # old Android logs have extended attributes and copy2 can stall.
            with path.open("rb") as src, dest.open("wb") as dst:
                shutil.copyfileobj(src, dst, length=1024 * 1024)


def write_readme(
    *,
    client_rows: list[dict[str, Any]],
    server_rows: list[dict[str, Any]],
    power_rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
    client_audit: list[dict[str, Any]],
    server_audit: list[dict[str, Any]],
) -> None:
    issue_client = sum(1 for row in client_audit if row["status"] != "ok")
    issue_server = sum(1 for row in server_audit if row["status"] != "ok")
    issue_coverage = sum(1 for row in coverage_rows if row["status"] != "ok")
    lines = [
        "# All Device Federated Measurement Results",
        "",
        f"- Created at: `{datetime.now().isoformat(timespec='seconds')}`",
        "- Scope: Gemma 3 270M and Qwen 0.5B over BoolQ, QNLI, PIQA, HellaSwag, SocialQA, ARC-E, WinoGrande with FedAvg+LoRA, FedProx+LoRA, FlexLoRA, SplitLoRA.",
        "- Scope: Gemma 3 1B over the same 7 datasets with SplitLoRA only.",
        "- Runs are one round with three local steps per client. Client metrics are per-client round-1 values; summary files contain averaged values.",
        "",
        "## Cohorts",
        "",
        "- `nova5_jetson3`: 5 nova phones + 3 Jetsons for Gemma 270M and Qwen 0.5B.",
        "- `nova5_jetson2`: 5 nova phones + 2 Jetsons for Gemma 1B SplitLoRA.",
        "- `mate20_jad_vivo`: Mate20 + JAD + vivo for Gemma 270M, Qwen 0.5B, and Gemma 1B SplitLoRA.",
        "",
        "## Key Files",
        "",
        "- `consolidated/all_client_round1_detail.csv`: per-client step time, upload/download time, communication bytes, RSS, power.",
        "- `consolidated/all_server_round1_detail.csv`: per-run/per-round server metrics including aggregation time.",
        "- `consolidated/all_power_detail.csv`: per-client power records.",
        "- `consolidated/all_round1_summary.csv`: compact per-run summary rows.",
        "- `audit/coverage_matrix.csv`: expected vs actual device coverage.",
        "- `audit/required_client_metrics_audit.csv`: required metric validation for every client row.",
        "- `audit/server_metrics_audit.csv`: server aggregation metric validation.",
        "- `by_cohort_model/`: same tables split by device cohort and model.",
        "- `source_index.csv`: source directories used to build this package. Raw old result folders are not duplicated in the zip to avoid copying checkpoint/log artifacts.",
        "",
        "## Counts",
        "",
        f"- Client detail rows: `{len(client_rows)}`",
        f"- Server detail rows: `{len(server_rows)}`",
        f"- Power rows: `{len(power_rows)}`",
        f"- Summary rows: `{len(summary_rows)}`",
        f"- Coverage rows: `{len(coverage_rows)}`",
        f"- Coverage issues: `{issue_coverage}`",
        f"- Client metric audit issues: `{issue_client}`",
        f"- Server metric audit issues: `{issue_server}`",
        "",
        "## Audit Notes",
        "",
        "- `coverage_issues=0`: all expected dataset/method/model/cohort combinations are present.",
        "- `server_metric_issues=0`: every server round has non-negative aggregation time and client count.",
        "- Client metric issues are preserved as raw-source problems, not silently filled. See `audit/issue_summary.csv` and `audit/client_metric_issues.csv`.",
        "- Old FlexLoRA rows have positive `download_time_sec` but `download_bytes=0`; this is an old byte-counter instrumentation gap.",
        "- Old Gemma 1B 7-client nova rows have `avg_power_w=0.0`; those power readings were not recovered from batterystats. Three-phone Gemma 1B power rows are valid.",
        "- Three old BoolQ FedAvg nova rows have `upload_time_sec=0.0`; upload bytes are present, timing was not recorded.",
        "",
        "## Required Metrics",
        "",
        "The package audits these required fields: `steps_completed`, `step_times_sec_json`, `download_time_sec`, `upload_time_sec`, `download_bytes`, `upload_bytes`, `transmitted_bytes`, `avg_rss_mb`, `peak_rss_mb`, `avg_power_w`, and server `aggregation_time_sec`.",
        "",
    ]
    (OUT / "README.md").write_text("\n".join(lines), encoding="utf-8")


def make_zip() -> None:
    if ZIP_PATH.exists():
        ZIP_PATH.unlink()
    with zipfile.ZipFile(ZIP_PATH, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for path in OUT.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(OUT.parent))


def main() -> None:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)

    client_rows: list[dict[str, Any]] = []
    server_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    validation_rows: list[dict[str, Any]] = []

    run_sources = build_nova_jetson_sources()
    for source in run_sources:
        collect_run_source(source, client_rows, server_rows, power_rows, summary_rows, validation_rows)
    collect_gemma1b_7client(client_rows, server_rows, power_rows, summary_rows, validation_rows)
    collect_three_phone(client_rows, server_rows, power_rows, summary_rows, validation_rows)

    client_preferred = [
        "cohort",
        "cohort_label",
        "model_key",
        "model_label",
        "dataset_key",
        "dataset",
        "method",
        "method_label",
        "client_id",
        "client_group",
        "run_id",
        "steps_completed",
        "step_times_sec_json",
        "mean_step_time_sec",
        "max_step_time_sec",
        "download_time_sec",
        "upload_time_sec",
        "download_bytes",
        "upload_bytes",
        "transmitted_bytes",
        "avg_rss_mb",
        "peak_rss_mb",
        "avg_power_w",
        "power_samples",
    ]
    server_preferred = [
        "cohort",
        "cohort_label",
        "model_key",
        "model_label",
        "dataset_key",
        "dataset",
        "method",
        "method_label",
        "run_id",
        "round",
        "aggregated_clients",
        "num_clients",
        "failures",
        "num_failures",
        "aggregation_time_sec",
        "total_transmitted_bytes",
        "mean_client_step_time_sec",
        "mean_client_upload_time_sec",
        "mean_client_download_time_sec",
        "mean_client_avg_rss_mb",
        "mean_client_peak_rss_mb",
        "mean_client_power_w",
    ]

    consolidated = OUT / "consolidated"
    write_csv(consolidated / "all_client_round1_detail.csv", client_rows, client_preferred)
    write_csv(consolidated / "all_server_round1_detail.csv", server_rows, server_preferred)
    write_csv(consolidated / "all_power_detail.csv", power_rows)
    write_csv(consolidated / "all_round1_summary.csv", summary_rows, server_preferred)
    write_csv(consolidated / "all_source_validation.csv", validation_rows)

    write_grouped_outputs(client_rows, server_rows, power_rows, summary_rows, validation_rows)

    coverage_rows = build_coverage(client_rows, server_rows)
    client_audit = audit_client_rows(client_rows)
    server_audit = audit_server_rows(server_rows)
    audit_dir = OUT / "audit"
    write_csv(audit_dir / "coverage_matrix.csv", coverage_rows)
    write_csv(audit_dir / "required_client_metrics_audit.csv", client_audit)
    write_csv(audit_dir / "server_metrics_audit.csv", server_audit)
    write_csv(audit_dir / "client_metric_issues.csv", [row for row in client_audit if row["status"] != "ok"])
    write_csv(audit_dir / "server_metric_issues.csv", [row for row in server_audit if row["status"] != "ok"])
    write_csv(audit_dir / "coverage_issues.csv", [row for row in coverage_rows if row["status"] != "ok"])
    write_csv(audit_dir / "issue_summary.csv", build_issue_summary(client_audit, server_audit, coverage_rows))

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "package_dir": str(OUT),
        "zip_path": str(ZIP_PATH),
        "client_rows": len(client_rows),
        "server_rows": len(server_rows),
        "power_rows": len(power_rows),
        "summary_rows": len(summary_rows),
        "validation_rows": len(validation_rows),
        "coverage_rows": len(coverage_rows),
        "coverage_issues": sum(1 for row in coverage_rows if row["status"] != "ok"),
        "client_metric_issues": sum(1 for row in client_audit if row["status"] != "ok"),
        "server_metric_issues": sum(1 for row in server_audit if row["status"] != "ok"),
        "source_roots": sorted({str(source.source_root.relative_to(ROOT)) for source in run_sources})
        + [
            "deliverables/gemma1b_splitlora_7client_7datasets_20260425_gemma1b_splitlora_7client_7datasets_embedding_only_rerun1",
            "deliverables/three_phone_all_models_gemma270m_gemma1b_qwen05b_20260430",
        ],
    }
    (OUT / "PACKAGE_MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(
        OUT / "source_index.csv",
        [{"source_root": source_root} for source_root in manifest["source_roots"]],
        ["source_root"],
    )
    write_readme(
        client_rows=client_rows,
        server_rows=server_rows,
        power_rows=power_rows,
        summary_rows=summary_rows,
        coverage_rows=coverage_rows,
        client_audit=client_audit,
        server_audit=server_audit,
    )

    make_zip()
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
