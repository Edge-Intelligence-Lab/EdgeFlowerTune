from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko
import yaml


ROOT = Path(__file__).resolve().parents[2]
LROOT = ROOT / "L-shaped_code_docs_backup"
ADB = Path("${ANDROID_HOME}/platform-tools/adb")

SERVER_HOST = "10.200.14.82"
SERVER_USER = "AndyLu666"
SERVER_PYTHON = "/home/AndyLu666/gemma3_server_eval_env/bin/python"
SERVER_CUDA_VISIBLE_DEVICES = "3"
CLASSIC_SERVER_ROOT = "/home/AndyLu666/gemma3_mixed_formal"
SPLIT_SERVER_ROOT = "/home/AndyLu666/L-shaped-run-classic"
GEMMA270M_SERVER_MODEL = "/home/AndyLu666/gemma3_mixed_formal/models/gemma-3-270m"
QWEN05B_SERVER_MODEL = "/home/AndyLu666/gemma3_mixed_formal/models/qwen2.5-0.5b"
GEMMA1B_SERVER_MODEL = "/home/AndyLu666/gemma3_mixed_formal/models/gemma-3-1b-pt"

GEMMA270M_LOCAL_MODEL = Path("${MODEL_ROOT}/gemma-3-270m")
QWEN05B_LOCAL_MODEL = ROOT / "qwen_lora_finetune" / "pretrained"
GEMMA1B_EMBED_LOCAL_MODEL = LROOT / "outputs" / "client_bundles" / "gemma-3-1b-pt-split0-embedding"
ANDROID_STAGE_DIR = "outputs/android_client/arm64-v8a/mft"

CLASSIC_MEASUREMENT = ROOT / "smoke" / "jetson_gemma3_fl_fedavg_20260415" / "run_boolq_fedavg_measurement.py"
SPLIT_MEASUREMENT = LROOT / "legacy_split" / "scripts" / "run_boolq_split_measurement.py"

CLIENT_IDS = ["nova_78", "nova_252", "nova_19", "nova_72", "nova_49", "jetson_121", "jetson_88"]
NOVA_DEVICES = [
    ("nova_78", "PHONE_ADB_SERIAL"),
    ("nova_252", "PHONE_ADB_SERIAL"),
    ("nova_19", "PHONE_ADB_SERIAL"),
    ("nova_72", "PHONE_ADB_SERIAL"),
    ("nova_49", "PHONE_ADB_SERIAL"),
]
JETSON_DEVICES = [
    ("jetson_121", "10.200.20.121"),
    ("jetson_88", "10.200.21.88"),
]
FLEXLORA_RANKS = {
    "nova_78": 4,
    "nova_252": 8,
    "nova_19": 4,
    "nova_72": 8,
    "nova_49": 16,
    "jetson_121": 4,
    "jetson_88": 16,
}


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    data_dir: str
    prefix: str
    prepare_script: str


GEMMA_DATASETS = [
    DatasetSpec("boolq", "BoolQ", "data/boolq", "boolq", "L-shaped_code_docs_backup/scripts/prepare_boolq_mmlu_csv.py"),
    DatasetSpec("qnli", "QNLI", "data/qnli", "qnli", "L-shaped_code_docs_backup/scripts/prepare_qnli_mmlu_csv.py"),
    DatasetSpec("piqa", "PIQA", "data/piqa", "piqa", "L-shaped_code_docs_backup/scripts/prepare_piqa_mmlu_csv.py"),
    DatasetSpec("hellaswag", "HellaSwag", "data/hellaswag", "hellaswag", "L-shaped_code_docs_backup/scripts/prepare_hellaswag_mmlu_csv.py"),
    DatasetSpec("socialqa", "SocialQA", "data/socialqa", "socialqa", "L-shaped_code_docs_backup/scripts/prepare_socialqa_mmlu_csv.py"),
    DatasetSpec("arce", "ARC-E", "data/arc", "arce", "L-shaped_code_docs_backup/scripts/prepare_arce_mmlu_csv.py"),
    DatasetSpec("winogrande", "WinoGrande", "data/winogrande", "winogrande", "L-shaped_code_docs_backup/scripts/prepare_winogrande_mmlu_csv.py"),
]
QWEN_DATASETS = [
    DatasetSpec("boolq", "BoolQ", "data/boolq", "boolq", "L-shaped_code_docs_backup/scripts/prepare_boolq_mmlu_csv.py"),
    DatasetSpec("qnli", "QNLI", "data/qnli", "qnli", "L-shaped_code_docs_backup/scripts/prepare_qnli_mmlu_csv.py"),
    DatasetSpec("piqa", "PIQA", "data/piqa_qwen", "piqa", "L-shaped_code_docs_backup/scripts/prepare_piqa_mmlu_csv.py"),
    DatasetSpec("hellaswag", "HellaSwag", "data/hellaswag_qwen", "hellaswag", "L-shaped_code_docs_backup/scripts/prepare_hellaswag_mmlu_csv.py"),
    DatasetSpec("socialqa", "SocialQA", "data/socialqa_qwen", "socialqa", "L-shaped_code_docs_backup/scripts/prepare_socialqa_mmlu_csv.py"),
    DatasetSpec("arce", "ARC-E", "data/arce_qwen", "arce", "L-shaped_code_docs_backup/scripts/prepare_arce_mmlu_csv.py"),
    DatasetSpec("winogrande", "WinoGrande", "data/winogrande_qwen", "winogrande", "L-shaped_code_docs_backup/scripts/prepare_winogrande_mmlu_csv.py"),
]
GEMMA1B_SPLIT_DATASETS = [
    DatasetSpec("boolq", "BoolQ", "data/boolq_gemma1b", "boolq", "L-shaped_code_docs_backup/scripts/prepare_boolq_mmlu_csv.py"),
    DatasetSpec("qnli", "QNLI", "data/qnli", "qnli", "L-shaped_code_docs_backup/scripts/prepare_qnli_mmlu_csv.py"),
    DatasetSpec("piqa", "PIQA", "data/piqa", "piqa", "L-shaped_code_docs_backup/scripts/prepare_piqa_mmlu_csv.py"),
    DatasetSpec("hellaswag", "HellaSwag", "data/hellaswag", "hellaswag", "L-shaped_code_docs_backup/scripts/prepare_hellaswag_mmlu_csv.py"),
    DatasetSpec("socialqa", "SocialQA", "data/socialqa", "socialqa", "L-shaped_code_docs_backup/scripts/prepare_socialqa_mmlu_csv.py"),
    DatasetSpec("arce", "ARC-E", "data/arc", "arce", "L-shaped_code_docs_backup/scripts/prepare_arce_mmlu_csv.py"),
    DatasetSpec("winogrande", "WinoGrande", "data/winogrande", "winogrande", "L-shaped_code_docs_backup/scripts/prepare_winogrande_mmlu_csv.py"),
]


def load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def to_float(value: Any, default: float = 0.0) -> float:
    if value in ("", None):
        return default
    return float(value)


def to_int(value: Any, default: int = 0) -> int:
    if value in ("", None):
        return default
    return int(float(value))


def dataset_paths(ds: DatasetSpec) -> dict[str, str]:
    return {
        "source_path": f"{ds.data_dir}/{ds.prefix}_train_mmlu.csv",
        "eval_source_path": f"{ds.data_dir}/{ds.prefix}_validation_mmlu.csv",
        "final_test_source_path": f"{ds.data_dir}/{ds.prefix}_test_mmlu.csv",
    }


def make_android_specs(model: str) -> list[dict[str, Any]]:
    local_model = GEMMA270M_LOCAL_MODEL if model == "gemma270m" else QWEN05B_LOCAL_MODEL
    remote_model = (
        "/data/local/tmp/L-shaped/models/gemma-3-270m"
        if model == "gemma270m"
        else "/data/local/tmp/L-shaped/models/qwen2.5-0.5b"
    )
    return [
        {
            "type": "android",
            "client_id": client_id,
            "client_index": idx,
            "serial": serial,
            "backend": "mft",
            "batch_size": 8,
            "max_seq_len": 64,
            "grad_accum_steps": None,
            "supports_grad_accum": False,
            "local_stage_dir": ANDROID_STAGE_DIR,
            "local_model_dir": str(local_model),
            "remote_model_dir": remote_model,
        }
        for idx, (client_id, serial) in enumerate(NOVA_DEVICES)
    ]


def make_jetson_specs(model: str) -> list[dict[str, Any]]:
    if model == "qwen05b":
        model_dir = "/home/jetson/L-shaped/models/qwen2.5-0.5b"
        remote_root = "/home/jetson/qwen05b_mixed_formal"
    else:
        model_dir = "/home/jetson/L-shaped/models/gemma-3-270m"
        remote_root = "/home/jetson/gemma3_mixed_formal"
    return [
        {
            "type": "jetson_gpu",
            "client_id": client_id,
            "client_index": 5 + idx,
            "host": host,
            "username": "jetson",
            "password": "jetson",
            "model_dir": model_dir,
            "remote_root": remote_root,
        }
        for idx, (client_id, host) in enumerate(JETSON_DEVICES)
    ]


def make_split_specs_gemma1b() -> list[dict[str, Any]]:
    return [
        {
            "type": "nano",
            "client_id": "jetson_121",
            "client_index": 0,
            "host": "10.200.20.121",
            "username": "jetson",
            "password": "jetson",
            "backend": "mft",
            "remote_root": "/home/jetson/L-shaped",
            "model_dir": "/home/jetson/L-shaped/models/gemma-3-1b-pt-split0-embedding",
            "batch_size": 8,
            "max_seq_len": 64,
            "max_rounds": 1,
        },
        {
            "type": "nano",
            "client_id": "jetson_88",
            "client_index": 1,
            "host": "10.200.21.88",
            "username": "jetson",
            "password": "jetson",
            "backend": "mft",
            "remote_root": "/home/jetson/L-shaped",
            "model_dir": "/home/jetson/L-shaped/models/gemma-3-1b-pt-split0-embedding",
            "batch_size": 8,
            "max_seq_len": 64,
            "max_rounds": 1,
        },
        *[
            {
                "type": "android",
                "client_id": client_id,
                "client_index": 2 + idx,
                "serial": serial,
                "backend": "mft",
                "batch_size": 8,
                "max_seq_len": 64,
                "max_rounds": 1,
                "local_stage_dir": str((LROOT / "outputs" / "android_client" / "arm64-v8a" / "mft").resolve()),
                "local_model_dir": str(GEMMA1B_EMBED_LOCAL_MODEL),
                "remote_model_dir": "/data/local/tmp/L-shaped/models/gemma-3-1b-pt-split0-embedding",
            }
            for idx, (client_id, serial) in enumerate(NOVA_DEVICES)
        ],
    ]


def configure_classic(
    *,
    template: Path,
    out: Path,
    run_name: str,
    ds: DatasetSpec,
    port: int,
    model: str,
    algorithm: str,
) -> Path:
    cfg = load_yaml(template)
    cfg["runtime"]["run_name"] = run_name
    cfg["runtime"]["output_dir"] = f"outputs/{run_name}"
    cfg["flower"].update(
        {
            "server_address": f"0.0.0.0:{port}",
            "num_rounds": 1,
            "min_available_clients": len(CLIENT_IDS),
            "min_fit_clients": len(CLIENT_IDS),
            "sample_clients": len(CLIENT_IDS),
            "round_timeout": 14400,
            "eval_every_rounds": 0,
            "client_wait_timeout": 7200,
        }
    )
    cfg["federated"].update(
        {
            "algorithm": algorithm,
            "aggregate_by_num_examples": True,
            "local_steps": 1,
            "local_epochs": 0,
        }
    )
    if algorithm == "flexlora":
        cfg["federated"]["client_lora_ranks"] = dict(FLEXLORA_RANKS)
    else:
        cfg["federated"].pop("client_lora_ranks", None)
    cfg["dataset"].update(
        {
            **dataset_paths(ds),
            "source": "mmlu_csv",
            "split": "train",
            "eval_split": "validation",
            "num_clients": len(CLIENT_IDS),
            "client_ids": list(CLIENT_IDS),
            "batch_size": 8,
            "jetson_batch_size": 1,
            "max_seq_len": 64,
            "partition_mode": "round_robin",
            "dirichlet_alpha": 0.8,
            "smoke_test_examples": 0,
        }
    )
    cfg.setdefault("model", {})["model_name_or_path"] = "qwen2.5-0.5b" if model == "qwen05b" else "gemma-3-270m"
    cfg.setdefault("client", {})["target_mode"] = "attn"
    write_yaml(out, cfg)
    return out


def configure_split_gemma1b(*, template: Path, out: Path, run_name: str, ds: DatasetSpec, port: int) -> Path:
    cfg = load_yaml(template)
    cfg["runtime"]["run_name"] = run_name
    cfg["runtime"]["output_dir"] = f"outputs/runs/{run_name}/server"
    cfg["flower"].update(
        {
            "server_address": f"0.0.0.0:{port}",
            "num_rounds": 1,
            "min_available_clients": len(CLIENT_IDS),
            "min_fit_clients": len(CLIENT_IDS),
            "sample_clients": len(CLIENT_IDS),
            "round_timeout": 14400,
            "client_wait_timeout": 7200,
            "eval_every_rounds": 0,
        }
    )
    cfg["federated"].update(
        {
            "algorithm": "splitlora",
            "local_steps": 1,
            "local_epochs": 0,
            "aggregate_by_num_examples": False,
            "accept_failures": False,
        }
    )
    cfg["dataset"].update(
        {
            "source": "mmlu_csv",
            "source_path": dataset_paths(ds)["source_path"],
            "split": "train",
            "eval_split": "validation",
            "num_clients": len(CLIENT_IDS),
            "client_ids": list(CLIENT_IDS),
            "batch_size": 8,
            "max_seq_len": 64,
            "partition_mode": "round_robin",
            "dirichlet_alpha": 0.8,
            "smoke_test_examples": 0,
        }
    )
    cfg.setdefault("model", {})["model_name_or_path"] = GEMMA1B_SERVER_MODEL
    write_yaml(out, cfg)
    return out


def generate_configs(batch_id: str) -> dict[str, Any]:
    config_dir = LROOT / "configs" / "required_reruns_nova5_jetson2"
    split_config_dir = LROOT / "legacy_split" / "configs" / "required_reruns_nova5_jetson2"
    android_gemma = config_dir / "nova5_android_mft_gemma270m_batch8_seq64.json"
    android_qwen = config_dir / "nova5_android_mft_qwen05b_batch8_seq64.json"
    jetson_gemma = config_dir / "jetson2_gemma270m_gpu.json"
    jetson_qwen = config_dir / "jetson2_qwen05b_gpu.json"
    split_specs = split_config_dir / "nova5_jetson2_split_mft_gemma1b_embedding_batch8_seq64.json"
    write_json(android_gemma, make_android_specs("gemma270m"))
    write_json(android_qwen, make_android_specs("qwen05b"))
    write_json(jetson_gemma, make_jetson_specs("gemma270m"))
    write_json(jetson_qwen, make_jetson_specs("qwen05b"))
    write_json(split_specs, make_split_specs_gemma1b())

    classic_configs: dict[str, Path] = {}
    fedavg_template = LROOT / "configs" / "classic_fl_gemma3_fedavg_lora_eight_client_boolq_seq64_b8_r1_l3.yaml"
    gemma_flex_template = LROOT / "configs" / "classic_fl_gemma3_flexlora_eight_client_boolq_seq64_b8_r1_l3.yaml"
    qwen_flex_template = LROOT / "configs" / "classic_fl_qwen05b_flexlora_eight_client_boolq_seq64_b8_r1_l3.yaml"
    split_template = LROOT / "legacy_split" / "configs" / "splitlora_gemma1b_seven_client_boolq_seq64_b8_r1_l3.yaml"

    run_name = f"{batch_id}_gemma270m_boolq_fedavg_nova5_jetson2_seq64_b8_r1_l1"
    classic_configs["gemma270m:boolq:fedavg"] = configure_classic(
        template=fedavg_template,
        out=config_dir / f"{run_name}.yaml",
        run_name=run_name,
        ds=GEMMA_DATASETS[0],
        port=19710,
        model="gemma270m",
        algorithm="fedavg_lora",
    )
    for idx, ds in enumerate(GEMMA_DATASETS):
        run_name = f"{batch_id}_gemma270m_{ds.key}_flexlora_nova5_jetson2_seq64_b8_r1_l1"
        classic_configs[f"gemma270m:{ds.key}:flexlora"] = configure_classic(
            template=gemma_flex_template,
            out=config_dir / f"{run_name}.yaml",
            run_name=run_name,
            ds=ds,
            port=19720 + idx,
            model="gemma270m",
            algorithm="flexlora",
        )
    for idx, ds in enumerate(QWEN_DATASETS):
        run_name = f"{batch_id}_qwen05b_{ds.key}_flexlora_nova5_jetson2_seq64_b8_r1_l1"
        classic_configs[f"qwen05b:{ds.key}:flexlora"] = configure_classic(
            template=qwen_flex_template,
            out=config_dir / f"{run_name}.yaml",
            run_name=run_name,
            ds=ds,
            port=19740 + idx,
            model="qwen05b",
            algorithm="flexlora",
        )

    split_configs: dict[str, Path] = {}
    for idx, ds in enumerate(GEMMA1B_SPLIT_DATASETS):
        run_name = f"{batch_id}_gemma1b_{ds.key}_splitlora_nova5_jetson2_seq64_b8_r1_l1"
        split_configs[ds.key] = configure_split_gemma1b(
            template=split_template,
            out=split_config_dir / f"{run_name}.yaml",
            run_name=run_name,
            ds=ds,
            port=19760 + idx,
        )
    return {
        "android_gemma": android_gemma,
        "android_qwen": android_qwen,
        "jetson_gemma": jetson_gemma,
        "jetson_qwen": jetson_qwen,
        "split_specs": split_specs,
        "classic_configs": classic_configs,
        "split_configs": split_configs,
    }


def check_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")


def check_dir(path: Path) -> None:
    if not path.is_dir():
        raise RuntimeError(f"missing dir: {path}")


def adb_devices() -> set[str]:
    proc = subprocess.run(
        [str(ADB), "devices"],
        cwd=str(ROOT),
        text=True,
        capture_output=True,
        timeout=30,
        check=False,
    )
    devices: set[str] = set()
    for line in proc.stdout.splitlines()[1:]:
        parts = line.split()
        if len(parts) >= 2 and parts[1] == "device":
            devices.add(parts[0])
    return devices


def check_jetson(host: str) -> None:
    with socket.create_connection((host, 22), timeout=5):
        pass
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(host, username="jetson", password="jetson", timeout=8, banner_timeout=8, auth_timeout=8)
    try:
        _, stdout, stderr = client.exec_command("echo ok", timeout=8)
        if stdout.read().decode("utf-8", "replace").strip() != "ok":
            raise RuntimeError(stderr.read().decode("utf-8", "replace"))
    finally:
        client.close()


def preflight() -> None:
    check_file(ADB)
    check_file(CLASSIC_MEASUREMENT)
    check_file(SPLIT_MEASUREMENT)
    check_dir(GEMMA270M_LOCAL_MODEL)
    check_dir(QWEN05B_LOCAL_MODEL)
    check_dir(GEMMA1B_EMBED_LOCAL_MODEL)
    for model_dir in (GEMMA270M_LOCAL_MODEL, QWEN05B_LOCAL_MODEL, GEMMA1B_EMBED_LOCAL_MODEL):
        for name in ("config.json", "model.safetensors", "tokenizer.json"):
            check_file(model_dir / name)
    connected = adb_devices()
    missing = [serial for _, serial in NOVA_DEVICES if serial not in connected]
    if missing:
        raise RuntimeError(f"missing adb devices: {missing}; connected={sorted(connected)}")
    for _, host in JETSON_DEVICES:
        check_jetson(host)
    for ds in {item.key: item for item in GEMMA_DATASETS + QWEN_DATASETS + GEMMA1B_SPLIT_DATASETS}.values():
        check_file(ROOT / ds.prepare_script)
        check_file(ROOT / dataset_paths(ds)["source_path"])


def run_cmd(cmd: list[str], log_path: Path, timeout: int = 21600) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def validate_run(run_dir: Path, *, method: str) -> dict[str, Any]:
    client_path = run_dir / "server" / "round1_client_summary_clean.csv"
    summary_path = run_dir / "server" / "summary_rounds_clean.csv"
    power_path = run_dir / "server" / "power_summary.csv"
    for path in (client_path, summary_path, power_path):
        check_file(path)
    client_rows = read_csv(client_path)
    summary_rows = read_csv(summary_path)
    power_rows = read_csv(power_path)
    failures: list[str] = []
    client_ids = {row.get("client_id", "") for row in client_rows}
    if client_ids != set(CLIENT_IDS):
        failures.append(f"client ids mismatch: {sorted(client_ids)}")
    if len(summary_rows) != 1:
        failures.append(f"summary_rows={len(summary_rows)}")
    if len(power_rows) != len(CLIENT_IDS):
        failures.append(f"power_rows={len(power_rows)}")
    power_index = {row.get("client_id", ""): row for row in power_rows}
    for row in client_rows:
        cid = row.get("client_id", "")
        if to_int(row.get("steps_completed")) != 1:
            failures.append(f"{cid} steps_completed={row.get('steps_completed')}")
        try:
            step_times = json.loads(row.get("step_times_sec_json") or "[]")
        except json.JSONDecodeError:
            step_times = []
        if len(step_times) != 1 or any(float(item) <= 0.0 for item in step_times):
            failures.append(f"{cid} invalid step_times={row.get('step_times_sec_json')}")
        for field in ("download_time_sec", "upload_time_sec", "download_bytes", "upload_bytes", "transmitted_bytes"):
            if to_float(row.get(field)) <= 0.0:
                failures.append(f"{cid} {field}={row.get(field)}")
        if method == "flexlora" and to_float(row.get("download_bytes")) <= 0.0:
            failures.append(f"{cid} flexlora round1 download_bytes={row.get('download_bytes')}")
        for field in ("avg_rss_mb", "peak_rss_mb"):
            if to_float(row.get(field)) <= 0.0:
                failures.append(f"{cid} {field}={row.get(field)}")
        power = power_index.get(cid, {})
        if to_int(power.get("power_samples")) <= 0:
            failures.append(f"{cid} power_samples={power.get('power_samples')}")
        if to_float(power.get("avg_power_w")) <= 0.0:
            failures.append(f"{cid} avg_power_w={power.get('avg_power_w')}")
        raw_power = run_dir / "clients" / cid / "power_samples.csv"
        if not raw_power.is_file():
            failures.append(f"{cid} missing raw power_samples.csv")
        elif not read_csv(raw_power):
            failures.append(f"{cid} empty raw power_samples.csv")
    if summary_rows and to_float(summary_rows[0].get("aggregation_time_sec"), -1.0) < 0.0:
        failures.append("negative aggregation_time_sec")
    return {"valid": not failures, "failures": "; ".join(failures), "client_rows": len(client_rows)}


def classic_cmd(
    *,
    config: Path,
    android_specs: Path,
    jetson_specs: Path,
    ds: DatasetSpec,
    run_id: str,
    run_label: str,
    model: str,
    port: int,
) -> list[str]:
    local_model = QWEN05B_LOCAL_MODEL if model == "qwen05b" else GEMMA270M_LOCAL_MODEL
    server_model = QWEN05B_SERVER_MODEL if model == "qwen05b" else GEMMA270M_SERVER_MODEL
    return [
        sys.executable,
        str(CLASSIC_MEASUREMENT),
        "--base-config",
        str(config),
        "--android-client-specs-json",
        str(android_specs),
        "--jetson-client-specs-json",
        str(jetson_specs),
        "--prepare-boolq-script",
        ds.prepare_script,
        "--prepare-script-output-dir",
        ds.data_dir,
        "--prepare-script-model-dir",
        str(local_model),
        "--prepare-script-seq-len",
        "64",
        "--server-ssh-host",
        SERVER_HOST,
        "--server-ssh-username",
        SERVER_USER,
        "--server-remote-root",
        CLASSIC_SERVER_ROOT,
        "--server-python",
        SERVER_PYTHON,
        "--server-model-dir",
        server_model,
        "--server-cuda-visible-devices",
        SERVER_CUDA_VISIBLE_DEVICES,
        "--local-model-dir",
        str(local_model),
        "--server-port",
        str(port),
        "--run-id",
        run_id,
        "--run-label",
        run_label,
        "--adb-path",
        str(ADB),
        "--startup-wait-sec",
        "20",
        "--poll-interval-sec",
        "5",
        "--timeout-sec",
        "21600",
    ]


def split_cmd(*, config: Path, specs: Path, run_id: str, run_label: str) -> list[str]:
    return [
        sys.executable,
        str(SPLIT_MEASUREMENT),
        "--base-config",
        str(config),
        "--client-specs-json",
        str(specs),
        "--run-id",
        run_id,
        "--run-label",
        run_label,
        "--skip-prepare-script",
        "--adb-path",
        str(ADB),
        "--server-ssh-host",
        SERVER_HOST,
        "--server-ssh-username",
        SERVER_USER,
        "--server-remote-root",
        SPLIT_SERVER_ROOT,
        "--server-address-host",
        SERVER_HOST,
        "--server-python",
        SERVER_PYTHON,
        "--cuda-visible-devices",
        SERVER_CUDA_VISIBLE_DEVICES,
        "--server-wait-timeout",
        "900",
        "--server-exit-timeout",
        "14400",
        "--client-exit-timeout",
        "14400",
        "--timeout-sec",
        "21600",
    ]


def build_jobs(batch_id: str, configs: dict[str, Any]) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    ds = GEMMA_DATASETS[0]
    jobs.append(
        {
            "model": "gemma270m",
            "dataset": ds,
            "method": "fedavg",
            "port": 19710,
            "run_id": f"{batch_id}_gemma270m_boolq_fedavg_nova5_jetson2",
            "cmd": classic_cmd(
                config=configs["classic_configs"]["gemma270m:boolq:fedavg"],
                android_specs=configs["android_gemma"],
                jetson_specs=configs["jetson_gemma"],
                ds=ds,
                run_id=f"{batch_id}_gemma270m_boolq_fedavg_nova5_jetson2",
                run_label="gemma270m_boolq_fedavg_nova5_jetson2",
                model="gemma270m",
                port=19710,
            ),
            "run_dir": LROOT / "outputs" / "runs" / f"{batch_id}_gemma270m_boolq_fedavg_nova5_jetson2",
        }
    )
    for idx, ds in enumerate(GEMMA_DATASETS):
        run_id = f"{batch_id}_gemma270m_{ds.key}_flexlora_nova5_jetson2"
        jobs.append(
            {
                "model": "gemma270m",
                "dataset": ds,
                "method": "flexlora",
                "port": 19720 + idx,
                "run_id": run_id,
                "cmd": classic_cmd(
                    config=configs["classic_configs"][f"gemma270m:{ds.key}:flexlora"],
                    android_specs=configs["android_gemma"],
                    jetson_specs=configs["jetson_gemma"],
                    ds=ds,
                    run_id=run_id,
                    run_label=f"gemma270m_{ds.key}_flexlora_nova5_jetson2",
                    model="gemma270m",
                    port=19720 + idx,
                ),
                "run_dir": LROOT / "outputs" / "runs" / run_id,
            }
        )
    for idx, ds in enumerate(QWEN_DATASETS):
        run_id = f"{batch_id}_qwen05b_{ds.key}_flexlora_nova5_jetson2"
        jobs.append(
            {
                "model": "qwen05b",
                "dataset": ds,
                "method": "flexlora",
                "port": 19740 + idx,
                "run_id": run_id,
                "cmd": classic_cmd(
                    config=configs["classic_configs"][f"qwen05b:{ds.key}:flexlora"],
                    android_specs=configs["android_qwen"],
                    jetson_specs=configs["jetson_qwen"],
                    ds=ds,
                    run_id=run_id,
                    run_label=f"qwen05b_{ds.key}_flexlora_nova5_jetson2",
                    model="qwen05b",
                    port=19740 + idx,
                ),
                "run_dir": LROOT / "outputs" / "runs" / run_id,
            }
        )
    for ds in GEMMA1B_SPLIT_DATASETS:
        run_id = f"{batch_id}_gemma1b_{ds.key}_splitlora_nova5_jetson2"
        jobs.append(
            {
                "model": "gemma1b",
                "dataset": ds,
                "method": "splitlora",
                "port": 19760 + GEMMA1B_SPLIT_DATASETS.index(ds),
                "run_id": run_id,
                "cmd": split_cmd(
                    config=configs["split_configs"][ds.key],
                    specs=configs["split_specs"],
                    run_id=run_id,
                    run_label=f"gemma1b_{ds.key}_splitlora_nova5_jetson2",
                ),
                "run_dir": LROOT / "legacy_split" / "outputs" / "runs" / run_id,
            }
        )
    return jobs


def build_final_tables(deliverable_dir: Path, status_rows: list[dict[str, Any]]) -> None:
    summary_rows: list[dict[str, Any]] = []
    client_rows: list[dict[str, Any]] = []
    power_rows: list[dict[str, Any]] = []
    for row in status_rows:
        if str(row.get("valid", "")).lower() != "true":
            continue
        run_dir = Path(str(row["run_dir"]))
        summary = read_csv(run_dir / "server" / "summary_rounds_clean.csv")[0]
        summary_rows.append({**{k: row[k] for k in ("model", "dataset_key", "dataset", "method", "run_id")}, **summary})
        for client in read_csv(run_dir / "server" / "round1_client_summary_clean.csv"):
            client_rows.append({**{k: row[k] for k in ("model", "dataset_key", "dataset", "method", "run_id")}, **client})
        for power in read_csv(run_dir / "server" / "power_summary.csv"):
            power_rows.append({**{k: row[k] for k in ("model", "dataset_key", "dataset", "method", "run_id")}, **power})
    write_rows(deliverable_dir / "rerun_round1_summary.csv", summary_rows)
    write_rows(deliverable_dir / "rerun_client_round1_detail.csv", client_rows)
    write_rows(deliverable_dir / "rerun_power_detail.csv", power_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--only", nargs="*", default=[], help="Filters like gemma270m:boolq:fedavg")
    parser.add_argument("--skip-preflight", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    deliverable_dir = ROOT / "deliverables" / f"required_reruns_nova5_jetson2_{args.batch_id}"
    logs_dir = deliverable_dir / "logs"
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    if not args.skip_preflight:
        preflight()
    configs = generate_configs(args.batch_id)
    jobs = build_jobs(args.batch_id, configs)
    filters = set(args.only)
    if filters:
        jobs = [
            job
            for job in jobs
            if f"{job['model']}:{job['dataset'].key}:{job['method']}" in filters
            or f"{job['dataset'].key}:{job['method']}" in filters
        ]
    write_json(
        deliverable_dir / "run_manifest.json",
        {
            "batch_id": args.batch_id,
            "clients": CLIENT_IDS,
            "nova_devices": NOVA_DEVICES,
            "jetson_devices": JETSON_DEVICES,
            "jobs": [
                {
                    "model": job["model"],
                    "dataset": job["dataset"].key,
                    "method": job["method"],
                    "run_id": job["run_id"],
                    "port": job["port"],
                    "run_dir": str(job["run_dir"]),
                }
                for job in jobs
            ],
            "configs": {key: str(value) for key, value in configs.items() if isinstance(value, Path)},
        },
    )
    if args.generate_only:
        print(deliverable_dir)
        return 0

    status_rows: list[dict[str, Any]] = []
    for index, job in enumerate(jobs, start=1):
        ds: DatasetSpec = job["dataset"]
        log_path = logs_dir / f"{index:02d}_{job['model']}_{ds.key}_{job['method']}.log"
        started = datetime.now().isoformat(timespec="seconds")
        print(f"[rerun] {index}/{len(jobs)} start {job['model']} {ds.key} {job['method']} run_id={job['run_id']}", flush=True)
        t0 = time.time()
        status: dict[str, Any] = {
            "model": job["model"],
            "dataset_key": ds.key,
            "dataset": ds.label,
            "method": job["method"],
            "run_id": job["run_id"],
            "run_dir": str(job["run_dir"]),
            "log_path": str(log_path),
            "started_at": started,
            "elapsed_sec": "",
            "valid": False,
            "failures": "",
        }
        try:
            run_cmd(job["cmd"], log_path)
            validation = validate_run(job["run_dir"], method=job["method"])
            status.update(validation)
            status["elapsed_sec"] = round(time.time() - t0, 3)
            if not validation["valid"]:
                raise RuntimeError(validation["failures"])
            append_csv(deliverable_dir / "measurement_validation.csv", status)
            status_rows.append(status)
            build_final_tables(deliverable_dir, status_rows)
        except Exception as exc:
            status["elapsed_sec"] = round(time.time() - t0, 3)
            status["failures"] = str(exc)
            append_csv(deliverable_dir / "measurement_validation.csv", status)
            status_rows.append(status)
            build_final_tables(deliverable_dir, status_rows)
            print(f"[rerun] failed {job['run_id']}: {exc}", flush=True)
            return 1
        print(f"[rerun] done {job['run_id']} elapsed={status['elapsed_sec']}s", flush=True)

    readme = [
        "# Required Reruns: nova5 + jetson2",
        "",
        f"- Batch id: `{args.batch_id}`",
        "- Clients: `nova_78`, `nova_252`, `nova_19`, `nova_72`, `nova_49`, `jetson_121`, `jetson_88`.",
        "- Parameters: `num_rounds=1`, `local_steps=1`, `batch_size=8`, `max_seq_len=64`.",
        "- Required metrics are validated in `measurement_validation.csv`; any zero/missing upload/download/RSS/power field fails the run.",
        "",
        "## Files",
        "",
        "- `measurement_validation.csv`: per-run validation.",
        "- `rerun_round1_summary.csv`: server round summary.",
        "- `rerun_client_round1_detail.csv`: per-client time/RSS/communication/power detail.",
        "- `rerun_power_detail.csv`: per-client power summary.",
        "- `logs/`: one launcher log per run.",
    ]
    (deliverable_dir / "README.md").write_text("\n".join(readme) + "\n", encoding="utf-8")
    print(deliverable_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
