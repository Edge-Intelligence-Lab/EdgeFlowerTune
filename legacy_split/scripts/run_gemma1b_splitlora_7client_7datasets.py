from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[3]
LROOT = ROOT / "L-shaped_code_docs_backup"
TEMPLATE_CONFIG = LROOT / "legacy_split" / "configs" / "splitlora_gemma1b_seven_client_boolq_seq64_b8_r1_l3.yaml"
CLIENT_SPECS = LROOT / "legacy_split" / "configs" / "seven_clients_mixed_split_mft_gemma1b.json"
MEASUREMENT_SCRIPT = LROOT / "legacy_split" / "scripts" / "run_boolq_split_measurement.py"
ADB = Path("${ANDROID_HOME}/platform-tools/adb")

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
    parser = argparse.ArgumentParser(description="Run Gemma 3 1B SplitLoRA on 5 nova phones + 2 Jetsons")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--only-dataset", choices=[item[0] for item in DATASETS], default="")
    parser.add_argument("--cuda-visible-devices", default="4")
    parser.add_argument("--server-port-base", type=int, default=19420)
    parser.add_argument("--server-exit-timeout", type=int, default=14400)
    parser.add_argument("--client-exit-timeout", type=int, default=7200)
    parser.add_argument("--timeout-sec", type=int, default=21600)
    parser.add_argument("--no-skip-binary-push", action="store_true")
    parser.add_argument("--skip-model-push", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def run_command(cmd: list[str], log_path: Path) -> int:
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
        return proc.wait()


def generate_config(dataset_key: str, dataset_label: str, dataset_csv: str, port: int, batch_id: str) -> Path:
    cfg = load_yaml(TEMPLATE_CONFIG)
    run_name = f"{batch_id}_gemma1b_{dataset_key}_splitlora_7client_seq64_b8_r1_l3"
    cfg["runtime"]["run_name"] = run_name
    cfg["runtime"]["output_dir"] = f"outputs/runs/{run_name}/server"
    cfg["flower"]["server_address"] = f"0.0.0.0:{port}"
    cfg["flower"]["num_rounds"] = 1
    cfg["flower"]["min_available_clients"] = 7
    cfg["flower"]["min_fit_clients"] = 7
    cfg["flower"]["sample_clients"] = 7
    cfg["flower"]["round_timeout"] = 14400
    cfg["flower"]["client_wait_timeout"] = 7200
    cfg["federated"]["local_steps"] = 3
    cfg["dataset"]["source"] = "mmlu_csv"
    cfg["dataset"]["source_path"] = dataset_csv
    cfg["dataset"]["split"] = "train"
    cfg["dataset"]["eval_split"] = "validation"
    cfg["dataset"]["num_clients"] = 7
    cfg["dataset"]["client_ids"] = [
        "jetson_121",
        "jetson_88",
        "nova_78",
        "nova_252",
        "nova_19",
        "nova_72",
        "nova_49",
    ]
    cfg["dataset"]["batch_size"] = 8
    cfg["dataset"]["max_seq_len"] = 64
    cfg["logging"]["save_every_rounds"] = 1
    cfg["logging"]["log_every_rounds"] = 1
    out = LROOT / "legacy_split" / "configs" / "gemma1b_splitlora_7client_7datasets" / (
        f"splitlora_gemma1b_7client_{dataset_key}_seq64_b8_r1_l3.yaml"
    )
    write_yaml(out, cfg)
    return out


def validate_inputs(selected: list[tuple[str, str, str]]) -> None:
    missing = []
    for _, _, rel_path in selected:
        if not (ROOT / rel_path).is_file():
            missing.append(rel_path)
    for path in (TEMPLATE_CONFIG, CLIENT_SPECS, MEASUREMENT_SCRIPT, ADB):
        if not path.is_file():
            missing.append(str(path))
    if missing:
        raise RuntimeError("Missing required files:\n" + "\n".join(missing))


def main() -> int:
    args = parse_args()
    batch_id = args.batch_id or datetime.now().strftime("%Y%m%d_%H%M%S")
    selected = [item for item in DATASETS if not args.only_dataset or item[0] == args.only_dataset]
    validate_inputs(selected)

    deliverable_dir = ROOT / "deliverables" / f"gemma1b_splitlora_7client_7datasets_{batch_id}"
    configs_dir = deliverable_dir / "configs"
    logs_dir = deliverable_dir / "logs"
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    (deliverable_dir / "status.jsonl").write_text("", encoding="utf-8")

    statuses: list[dict[str, Any]] = []
    for idx, (dataset_key, dataset_label, dataset_csv) in enumerate(selected):
        port = args.server_port_base + idx
        config_path = generate_config(dataset_key, dataset_label, dataset_csv, port, batch_id)
        local_config_copy = configs_dir / config_path.name
        local_config_copy.parent.mkdir(parents=True, exist_ok=True)
        local_config_copy.write_text(config_path.read_text(encoding="utf-8"), encoding="utf-8")
        run_id = f"{batch_id}_gemma1b_{dataset_key}_splitlora_7client"
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
            f"gemma1b_{dataset_key}_splitlora_7client",
            "--skip-prepare-script",
            "--adb-path",
            str(ADB),
            "--cuda-visible-devices",
            args.cuda_visible_devices,
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
            "config": str(config_path),
            "port": port,
            "started": datetime.now().isoformat(timespec="seconds"),
            "returncode": None,
        }
        statuses.append(status)
        with (deliverable_dir / "status.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False) + "\n")
        if args.dry_run:
            status["returncode"] = 0
            status["finished"] = datetime.now().isoformat(timespec="seconds")
            continue
        rc = run_command(cmd, logs_dir / f"{dataset_key}.log")
        status["returncode"] = rc
        status["finished"] = datetime.now().isoformat(timespec="seconds")
        with (deliverable_dir / "status.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(status, ensure_ascii=False) + "\n")
        if rc != 0:
            break

    (deliverable_dir / "status_latest.json").write_text(
        json.dumps(statuses, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    readme = [
        "# Gemma 3 1B SplitLoRA 7-Client 7-Dataset Run",
        "",
        f"Batch id: `{batch_id}`",
        "",
        "Clients: `nova_78`, `nova_252`, `nova_19`, `nova_72`, `nova_49`, `jetson_121`, `jetson_88`.",
        "",
        "Parameters: Gemma 3 1B, SplitLoRA, batch 8, seq 64, 1 round, 3 local steps.",
        "",
        "Status files:",
        "",
        "- `status.jsonl`",
        "- `status_latest.json`",
        "- `logs/*.log`",
        "",
    ]
    (deliverable_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")
    print(deliverable_dir)
    return 0 if all(item.get("returncode") == 0 for item in statuses) else 1


if __name__ == "__main__":
    raise SystemExit(main())
