from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
LROOT = ROOT / "L-shaped_code_docs_backup"
TEMPLATE_CONFIG = LROOT / "legacy_split" / "configs" / "splitlora_gemma1b_three_client_boolq_seq64_b8_r1_l3.yaml"
CLIENT_SPECS = LROOT / "legacy_split" / "configs" / "three_phone_mixed_split_mft_gemma1b_embedding.json"
MEASUREMENT_SCRIPT = LROOT / "legacy_split" / "scripts" / "run_boolq_split_measurement.py"
ADB = Path("${ANDROID_HOME}/platform-tools/adb")

CLIENT_IDS = ["vivo_73", "jad_24", "mate20_144"]
METHOD = "splitlora"

DATASETS = [
    ("boolq", "BoolQ", "data/boolq_gemma1b/boolq_train_mmlu.csv"),
    ("qnli", "QNLI", "data/qnli/qnli_train_mmlu.csv"),
    ("piqa", "PIQA", "data/piqa/piqa_train_mmlu.csv"),
    ("hellaswag", "HellaSwag", "data/hellaswag/hellaswag_train_mmlu.csv"),
    ("socialqa", "SocialQA", "data/socialqa/socialqa_train_mmlu.csv"),
    ("arce", "ARC-E", "data/arc/arce_train_mmlu.csv"),
    ("winogrande", "WinoGrande", "data/winogrande/winogrande_train_mmlu.csv"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Gemma 3 1B SplitLoRA on vivo + JAD + Mate20")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--only-dataset", choices=[item[0] for item in DATASETS], default="")
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--server-port-base", type=int, default=19620)
    parser.add_argument("--server-exit-timeout", type=int, default=14400)
    parser.add_argument("--client-exit-timeout", type=int, default=7200)
    parser.add_argument("--timeout-sec", type=int, default=21600)
    parser.add_argument("--server-wait-timeout", type=int, default=600)
    parser.add_argument("--no-skip-binary-push", action="store_true")
    parser.add_argument("--skip-model-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def run_command(cmd: list[str], log_path: Path, timeout: int) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("$ " + " ".join(cmd) + "\n\n")
        handle.flush()
        proc = subprocess.Popen(
            cmd,
            cwd=str(ROOT),
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            return proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            handle.write(f"\n[TIMEOUT] killed after {timeout}s\n")
            return 124


def generate_config(dataset_key: str, dataset_csv: str, port: int, batch_id: str) -> Path:
    cfg = load_yaml(TEMPLATE_CONFIG)
    run_name = f"{batch_id}_gemma1b_{dataset_key}_splitlora_3phone_seq64_b8_r1_l3"
    cfg["runtime"]["run_name"] = run_name
    cfg["runtime"]["output_dir"] = f"outputs/runs/{run_name}/server"
    cfg["flower"]["server_address"] = f"0.0.0.0:{port}"
    cfg["flower"]["num_rounds"] = 1
    cfg["flower"]["min_available_clients"] = len(CLIENT_IDS)
    cfg["flower"]["min_fit_clients"] = len(CLIENT_IDS)
    cfg["flower"]["sample_clients"] = len(CLIENT_IDS)
    cfg["flower"]["round_timeout"] = 14400
    cfg["flower"]["client_wait_timeout"] = 7200
    cfg["federated"]["local_steps"] = 3
    cfg["dataset"]["source"] = "mmlu_csv"
    cfg["dataset"]["source_path"] = dataset_csv
    cfg["dataset"]["split"] = "train"
    cfg["dataset"]["eval_split"] = "validation"
    cfg["dataset"]["num_clients"] = len(CLIENT_IDS)
    cfg["dataset"]["client_ids"] = list(CLIENT_IDS)
    cfg["dataset"]["batch_size"] = 8
    cfg["dataset"]["max_seq_len"] = 64
    cfg["client"]["split_layer"] = 0
    cfg["client"]["client_mode"] = "split_lora"
    cfg["model"]["target_embedding_mode"] = "scaled_input_embedding"
    cfg["model"]["freeze_input_embeddings"] = True
    cfg["logging"]["save_every_rounds"] = 1
    cfg["logging"]["log_every_rounds"] = 1
    out = LROOT / "legacy_split" / "configs" / "gemma1b_splitlora_3phone_7datasets" / (
        f"splitlora_gemma1b_3phone_{dataset_key}_seq64_b8_r1_l3.yaml"
    )
    write_yaml(out, cfg)
    return out


def validate_inputs(selected: list[tuple[str, str, str]]) -> None:
    missing: list[str] = []
    for _, _, rel_path in selected:
        if not (ROOT / rel_path).is_file():
            missing.append(rel_path)
    for path in (TEMPLATE_CONFIG, CLIENT_SPECS, MEASUREMENT_SCRIPT, ADB):
        if not path.is_file():
            missing.append(str(path))
    specs = json.loads(CLIENT_SPECS.read_text(encoding="utf-8"))
    for item in specs:
        model_dir = Path(item["local_model_dir"])
        stage_dir = ROOT / item["local_stage_dir"]
        if not model_dir.joinpath("model.safetensors").is_file():
            missing.append(str(model_dir / "model.safetensors"))
        if not stage_dir.joinpath("lshaped_flower_client").is_file():
            missing.append(str(stage_dir / "lshaped_flower_client"))
    if missing:
        raise RuntimeError("Missing required files:\n" + "\n".join(missing))


def power_detail_rows(run_dir: Path, dataset_label: str, run_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for client_id in CLIENT_IDS:
        power_csv = run_dir / "clients" / client_id / "power_samples.csv"
        samples = read_csv(power_csv)
        if not samples:
            rows.append(
                {
                    "dataset": dataset_label,
                    "method": METHOD,
                    "run_id": run_id,
                    "client_id": client_id,
                    "avg_power_w": "",
                    "power_samples": 0,
                    "raw_power_quality": "missing",
                    "raw_power_flags": "missing_power_samples",
                }
            )
            continue
        sample = samples[-1]
        row = {
            "dataset": dataset_label,
            "method": METHOD,
            "run_id": run_id,
            "client_id": client_id,
            "avg_power_w": sample.get("avg_power_w", ""),
            "power_samples": len(samples),
        }
        for key, value in sample.items():
            row[f"raw_{key}"] = value
        rows.append(row)
    return rows


def collect_outputs(
    deliverable_dir: Path,
    statuses: list[dict[str, Any]],
    *,
    batch_id: str,
) -> None:
    validation_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    server_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []

    for status in statuses:
        dataset_key = status["dataset"]
        dataset_label = status["label"]
        run_id = status["run_id"]
        run_dir = Path(status["run_dir"])
        log_path = Path(status["log_path"])
        valid = status.get("returncode") == 0
        failures = ""
        if not valid:
            failures = f"returncode={status.get('returncode')}"
        clean_client = read_csv(run_dir / "server" / "round1_client_summary_clean.csv")
        clean_summary = read_csv(run_dir / "server" / "summary_rounds_clean.csv")
        if valid and len(clean_client) != len(CLIENT_IDS):
            valid = False
            failures = f"client_rows={len(clean_client)}"
        if valid and len(clean_summary) != 1:
            valid = False
            failures = f"summary_rows={len(clean_summary)}"
        raw_power = power_detail_rows(run_dir, dataset_label, run_id)
        if valid and len(raw_power) != len(CLIENT_IDS):
            valid = False
            failures = f"power_rows={len(raw_power)}"
        validation_rows.append(
            {
                "dataset": dataset_label,
                "dataset_key": dataset_key,
                "method": METHOD,
                "run_id": run_id,
                "run_dir": str(run_dir),
                "log_path": str(log_path),
                "started_at": status.get("started", ""),
                "elapsed_sec": status.get("elapsed_sec", ""),
                "valid": str(valid),
                "failures": failures,
                "client_rows": len(clean_client),
                "summary_rows": len(clean_summary),
                "power_rows": len(raw_power),
            }
        )
        for row in clean_client:
            merged = {"dataset": dataset_label, "method": METHOD, "run_id": run_id, **row}
            client_rows.append(merged)
        for row in clean_summary:
            merged = {"dataset": dataset_label, "method": METHOD, "run_id": run_id, **row}
            server_rows.append(merged)
            summary_rows.append({"dataset": dataset_label, "method": METHOD, "run_id": run_id, "valid": str(valid), **row})
        power_rows.extend(raw_power)

    for row in validation_rows:
        if row["valid"] != "True":
            issues.append({"level": "error", "dataset_key": row["dataset_key"], "issue": row["failures"]})
    for row in client_rows:
        try:
            step_times = json.loads(row.get("step_times_sec_json", "[]"))
        except json.JSONDecodeError:
            step_times = []
        checks = {
            "steps_completed": row.get("steps_completed") == "3" or str(row.get("steps_completed")) == "3",
            "step_times": isinstance(step_times, list) and len(step_times) == 3,
            "upload_time": float(row.get("upload_time_sec", 0) or 0) > 0,
            "download_time": float(row.get("download_time_sec", 0) or 0) >= 0,
            "upload_bytes": float(row.get("upload_bytes", 0) or 0) > 0,
            "download_bytes": float(row.get("download_bytes", 0) or 0) > 0,
            "avg_rss": float(row.get("avg_rss_mb", 0) or 0) > 0,
            "peak_rss": float(row.get("peak_rss_mb", 0) or 0) > 0,
            "avg_power": float(row.get("avg_power_w", 0) or 0) > 0,
        }
        if not all(checks.values()):
            issues.append(
                {
                    "level": "error",
                    "dataset": row.get("dataset", ""),
                    "client_id": row.get("client_id", ""),
                    "issue": "client_metric_failed",
                    "checks": json.dumps(checks, ensure_ascii=False),
                }
            )
    for row in power_rows:
        quality = row.get("raw_power_quality", "")
        flags = row.get("raw_power_flags", "")
        try:
            avg_power = float(row.get("avg_power_w", 0) or 0)
        except ValueError:
            avg_power = 0.0
        if quality != "ok" or flags or avg_power <= 0 or avg_power > 20:
            issues.append(
                {
                    "level": "error",
                    "dataset": row.get("dataset", ""),
                    "client_id": row.get("client_id", ""),
                    "issue": "power_metric_failed",
                    "avg_power_w": row.get("avg_power_w", ""),
                    "raw_power_quality": quality,
                    "raw_power_flags": flags,
                }
            )

    write_csv(deliverable_dir / "measurement_validation.csv", validation_rows)
    write_csv(deliverable_dir / "gemma1b_3phone_client_round1_detail.csv", client_rows)
    write_csv(deliverable_dir / "gemma1b_3phone_server_round1_detail.csv", server_rows)
    write_csv(deliverable_dir / "gemma1b_3phone_round1_summary.csv", summary_rows)
    write_csv(deliverable_dir / "gemma1b_3phone_power_detail.csv", power_rows)
    write_csv(deliverable_dir / "audit_issues.csv", issues)


def main() -> int:
    args = parse_args()
    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    selected = [item for item in DATASETS if not args.only_dataset or item[0] == args.only_dataset]
    validate_inputs(selected)

    deliverable_dir = ROOT / "deliverables" / f"gemma1b_splitlora_3phone_7datasets_{batch_id}"
    configs_dir = deliverable_dir / "configs"
    logs_dir = deliverable_dir / "logs"
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    (deliverable_dir / "status.jsonl").write_text("", encoding="utf-8")

    statuses: list[dict[str, Any]] = []
    for idx, (dataset_key, dataset_label, dataset_csv) in enumerate(selected):
        port = args.server_port_base + idx
        config_path = generate_config(dataset_key, dataset_csv, port, batch_id)
        local_config_copy = configs_dir / config_path.name
        local_config_copy.parent.mkdir(parents=True, exist_ok=True)
        local_config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        run_id = f"{batch_id}_gemma1b_{dataset_key}_splitlora_3phone"
        log_path = logs_dir / f"{dataset_key}.log"
        cmd = [
            sys.executable,
            str(MEASUREMENT_SCRIPT),
            "--base-config",
            str(config_path),
            "--client-specs-json",
            str(CLIENT_SPECS),
            "--run-id",
            run_id,
            "--run-label",
            f"gemma1b_{dataset_key}_splitlora_3phone",
            "--skip-prepare-script",
            "--adb-path",
            str(ADB),
            "--cuda-visible-devices",
            args.cuda_visible_devices,
            "--server-port",
            str(port),
            "--server-wait-timeout",
            str(args.server_wait_timeout),
            "--server-exit-timeout",
            str(args.server_exit_timeout),
            "--client-exit-timeout",
            str(args.client_exit_timeout),
            "--timeout-sec",
            str(args.timeout_sec),
        ]
        if not args.no_skip_binary_push:
            cmd.append("--skip-android-binary-push")
        if args.skip_model_push:
            cmd.append("--skip-android-model-push")
        status = {
            "dataset": dataset_key,
            "label": dataset_label,
            "run_id": run_id,
            "run_dir": str(LROOT / "legacy_split" / "outputs" / "runs" / run_id),
            "config": str(config_path),
            "port": port,
            "log_path": str(log_path),
            "started": datetime.now().isoformat(timespec="seconds"),
            "returncode": None,
        }
        statuses.append(status)
        with (deliverable_dir / "status.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False) + "\n")
        if args.dry_run:
            status["returncode"] = 0
            status["finished"] = datetime.now().isoformat(timespec="seconds")
            status["elapsed_sec"] = 0.0
        else:
            start = time.time()
            rc = run_command(cmd, log_path, timeout=args.timeout_sec + 120)
            status["returncode"] = rc
            status["finished"] = datetime.now().isoformat(timespec="seconds")
            status["elapsed_sec"] = round(time.time() - start, 3)
        with (deliverable_dir / "status.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False) + "\n")
        collect_outputs(deliverable_dir, statuses, batch_id=batch_id)
        if status["returncode"] != 0:
            break

    collect_outputs(deliverable_dir, statuses, batch_id=batch_id)
    (deliverable_dir / "status_latest.json").write_text(
        json.dumps(statuses, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Gemma 3 1B SplitLoRA 3-Phone 7-Dataset Run",
        "",
        f"Batch id: `{batch_id}`",
        "",
        "Clients: `vivo_73`, `jad_24`, `mate20_144`.",
        "",
        "Client model bundle: `gemma-3-1b-pt-split0-embedding` (only `model.embed_tokens.weight`; hidden layers stay on server).",
        "",
        "Parameters: Gemma 3 1B, SplitLoRA, batch 8, seq 64, 1 round, 3 local steps.",
        "",
        "Outputs:",
        "",
        "- `measurement_validation.csv`",
        "- `gemma1b_3phone_client_round1_detail.csv`",
        "- `gemma1b_3phone_server_round1_detail.csv`",
        "- `gemma1b_3phone_round1_summary.csv`",
        "- `gemma1b_3phone_power_detail.csv`",
        "- `audit_issues.csv`",
        "- `status.jsonl`",
        "- `logs/*.log`",
        "",
    ]
    (deliverable_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(deliverable_dir)
    return 0 if all(item.get("returncode") == 0 for item in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
