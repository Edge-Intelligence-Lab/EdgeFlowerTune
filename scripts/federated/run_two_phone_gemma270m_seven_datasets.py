from __future__ import annotations

import argparse
import atexit
import csv
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
LROOT = ROOT / "L-shaped_code_docs_backup"
LOCAL_MODEL_DIR = Path("${MODEL_ROOT}/gemma-3-270m")
ADB = Path("${ANDROID_HOME}/platform-tools/adb")
SERVER_HOST = "10.200.14.82"
SERVER_USER = "AndyLu666"
SERVER_PYTHON = "/home/AndyLu666/gemma3_server_eval_env/bin/python"
CLASSIC_SERVER_ROOT = "/home/AndyLu666/gemma3_mixed_formal"
CLASSIC_SERVER_MODEL = "/home/AndyLu666/gemma3_mixed_formal/models/gemma-3-270m"
SPLIT_SERVER_ROOT = "/home/AndyLu666/L-shaped-run-classic"
SERVER_CUDA_VISIBLE_DEVICES = "3"
ANDROID_STAGE_DIR = LROOT / "outputs" / "android_client" / "arm64-v8a" / "mft"
CLIENT_IDS = ["jad_24", "mate20_144"]
DEVICE_PROFILE = "two_phone"
CONFIG_DEVICE_TAG = "two_phone"
RUN_DEVICE_TAG = "2phone"
SUMMARY_PREFIX = "two_phone_gemma270m"
DELIVERABLE_PREFIX = "gemma270m_two_phone_seven_datasets"
README_TITLE = "Gemma 3 270M Two-Phone Federated Measurements"

DEVICE_PROFILES: dict[str, list[dict[str, Any]]] = {
    "two_phone": [
        {
            "client_id": "jad_24",
            "client_index": 0,
            "serial": "JAD_ADB_SERIAL",
            "grad_accum_steps": 8,
            "mlp_chunk_size": 16,
        },
        {
            "client_id": "mate20_144",
            "client_index": 1,
            "serial": "MATE20_USB_SERIAL",
            "grad_accum_steps": 8,
            "mlp_chunk_size": 16,
        },
    ],
    "vivo": [
        {
            "client_id": "vivo_73",
            "client_index": 0,
            "serial": "VIVO_ADB_SERIAL",
            "grad_accum_steps": 8,
            "mlp_chunk_size": 16,
            "local_stage_dir": str((LROOT / "outputs" / "android_client" / "arm64-v8a" / "mft_jad_0958").resolve()),
            "enable_fixed_performance": True,
        },
    ],
}

PROFILE_FLEX_LORA_RANKS: dict[str, dict[str, int]] = {
    "two_phone": {"jad_24": 4, "mate20_144": 8},
    "vivo": {"vivo_73": 4},
}


def configure_device_profile(profile: str) -> None:
    global CLIENT_IDS, DEVICE_PROFILE, CONFIG_DEVICE_TAG, RUN_DEVICE_TAG
    global SUMMARY_PREFIX, DELIVERABLE_PREFIX, README_TITLE
    if profile not in DEVICE_PROFILES:
        raise RuntimeError(f"unknown device profile: {profile}")
    DEVICE_PROFILE = profile
    CLIENT_IDS = [item["client_id"] for item in DEVICE_PROFILES[profile]]
    if profile == "two_phone":
        CONFIG_DEVICE_TAG = "two_phone"
        RUN_DEVICE_TAG = "2phone"
        SUMMARY_PREFIX = "two_phone_gemma270m"
        DELIVERABLE_PREFIX = "gemma270m_two_phone_seven_datasets"
        README_TITLE = "Gemma 3 270M Two-Phone Federated Measurements"
    else:
        CONFIG_DEVICE_TAG = profile
        RUN_DEVICE_TAG = profile
        SUMMARY_PREFIX = f"{profile}_gemma270m"
        DELIVERABLE_PREFIX = f"gemma270m_{profile}_seven_datasets"
        README_TITLE = f"Gemma 3 270M {profile} Federated Measurements"


def num_clients() -> int:
    return len(CLIENT_IDS)


@dataclass(frozen=True)
class DatasetSpec:
    key: str
    label: str
    data_dir: str
    prefix: str
    prepare_script: str


DATASETS = [
    DatasetSpec("boolq", "BoolQ", "data/boolq", "boolq", "L-shaped_code_docs_backup/scripts/prepare_boolq_mmlu_csv.py"),
    DatasetSpec("qnli", "QNLI", "data/qnli", "qnli", "L-shaped_code_docs_backup/scripts/prepare_qnli_mmlu_csv.py"),
    DatasetSpec("piqa", "PIQA", "data/piqa", "piqa", "L-shaped_code_docs_backup/scripts/prepare_piqa_mmlu_csv.py"),
    DatasetSpec(
        "hellaswag",
        "HellaSwag",
        "data/hellaswag",
        "hellaswag",
        "L-shaped_code_docs_backup/scripts/prepare_hellaswag_mmlu_csv.py",
    ),
    DatasetSpec(
        "socialqa",
        "SocialQA",
        "data/socialqa",
        "socialqa",
        "L-shaped_code_docs_backup/scripts/prepare_socialqa_mmlu_csv.py",
    ),
    DatasetSpec("arce", "ARC-E", "data/arc", "arce", "L-shaped_code_docs_backup/scripts/prepare_arce_mmlu_csv.py"),
    DatasetSpec(
        "winogrande",
        "WinoGrande",
        "data/winogrande",
        "winogrande",
        "L-shaped_code_docs_backup/scripts/prepare_winogrande_mmlu_csv.py",
    ),
]

CLASSIC_METHODS = {
    "fedavg": {
        "label": "FedAvg + LoRA",
        "algorithm": "fedavg_lora",
        "template": "L-shaped_code_docs_backup/configs/classic_fl_gemma3_fedavg_lora_eight_client_boolq_seq64_b8_r1_l3.yaml",
        "port": 19331,
        "prox_mu": 0.0,
    },
    "fedprox": {
        "label": "FedProx + LoRA",
        "algorithm": "fedprox_lora",
        "template": "L-shaped_code_docs_backup/configs/classic_fl_gemma3_fedprox_lora_eight_client_boolq_seq64_b8_r1_l3.yaml",
        "port": 19332,
        "prox_mu": 0.01,
    },
    "flexlora": {
        "label": "FlexLoRA",
        "algorithm": "flexlora",
        "template": "L-shaped_code_docs_backup/configs/classic_fl_gemma3_flexlora_eight_client_boolq_seq64_b8_r1_l3.yaml",
        "port": 19333,
        "prox_mu": 0.0,
        "client_lora_ranks": {"jad_24": 4, "mate20_144": 8},
    },
}

SPLIT_METHOD = {
    "label": "SplitLoRA",
    "algorithm": "splitlora",
    "template": "L-shaped_code_docs_backup/legacy_split/configs/splitlora_gemma270m_eight_client_boolq_seq64_b8_r1_l3.yaml",
    "port": 19334,
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def dataset_paths(ds: DatasetSpec) -> dict[str, str]:
    return {
        "source_path": f"{ds.data_dir}/{ds.prefix}_train_mmlu.csv",
        "eval_source_path": f"{ds.data_dir}/{ds.prefix}_validation_mmlu.csv",
        "final_test_source_path": f"{ds.data_dir}/{ds.prefix}_test_mmlu.csv",
    }


def make_android_specs() -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    for item in DEVICE_PROFILES[DEVICE_PROFILE]:
        specs.append(
            {
                "type": "android",
                "client_id": item["client_id"],
                "client_index": item["client_index"],
                "serial": item["serial"],
                "backend": "mft",
                "batch_size": 8,
                "max_seq_len": 64,
                "grad_accum_steps": item.get("grad_accum_steps", 8),
                "supports_grad_accum": True,
                "device_root": "/data/local/tmp/L-shaped",
                "local_stage_dir": item.get("local_stage_dir", str(ANDROID_STAGE_DIR.resolve())),
                "local_model_dir": str(LOCAL_MODEL_DIR),
                "checkpoint_every": 1,
                "checkpoint_mlp": True,
                "mlp_chunk_size": item.get("mlp_chunk_size", 16),
            }
        )
    return specs


def make_split_specs() -> list[dict[str, Any]]:
    specs = []
    for item in make_android_specs():
        split_item = {
            "type": "android",
            "client_id": item["client_id"],
            "client_index": item["client_index"],
            "serial": item["serial"],
            "backend": "mft",
            "batch_size": 8,
            "max_seq_len": 64,
            "max_rounds": 1,
            "device_root": item["device_root"],
            "local_stage_dir": str((ROOT / item["local_stage_dir"]).resolve()),
            "local_model_dir": item["local_model_dir"],
        }
        specs.append(split_item)
    return specs


def generate_configs() -> dict[str, Any]:
    config_root = LROOT / "configs" / f"{CONFIG_DEVICE_TAG}_gemma270m"
    split_config_root = LROOT / "legacy_split" / "configs" / f"{CONFIG_DEVICE_TAG}_gemma270m"
    android_specs_path = config_root / f"{CONFIG_DEVICE_TAG}_android_mft_gemma270m_batch8_seq64.json"
    jetson_specs_path = config_root / "empty_jetson_specs.json"
    split_specs_path = split_config_root / f"{CONFIG_DEVICE_TAG}_mixed_split_mft_gemma270m_batch8_seq64.json"

    write_json(android_specs_path, make_android_specs())
    write_json(jetson_specs_path, [])
    write_json(split_specs_path, make_split_specs())

    classic_configs: dict[tuple[str, str], Path] = {}
    split_configs: dict[str, Path] = {}
    for ds in DATASETS:
        paths = dataset_paths(ds)
        for method, meta in CLASSIC_METHODS.items():
            cfg = load_yaml(ROOT / str(meta["template"]))
            run_name = f"classic_fl_gemma270m_{method}_lora_{CONFIG_DEVICE_TAG}_{ds.key}_seq64_b8_r1_l3"
            cfg["runtime"]["run_name"] = run_name
            cfg["runtime"]["output_dir"] = f"outputs/{run_name}"
            cfg["flower"].update(
                {
                    "server_address": f"0.0.0.0:{meta['port']}",
                    "num_rounds": 1,
                    "min_available_clients": num_clients(),
                    "min_fit_clients": num_clients(),
                    "sample_clients": num_clients(),
                    "round_timeout": 10800,
                    "eval_every_rounds": 0,
                    "client_wait_timeout": 3600,
                }
            )
            cfg["federated"].update(
                {
                    "algorithm": meta["algorithm"],
                    "aggregate_by_num_examples": True,
                    "local_steps": 3,
                    "local_epochs": 0,
                    "grad_accum_steps": 8,
                    "jetson_grad_accum_steps": 8,
                    "prox_mu": meta["prox_mu"],
                }
            )
            if "client_lora_ranks" in meta:
                cfg["federated"]["client_lora_ranks"] = dict(PROFILE_FLEX_LORA_RANKS[DEVICE_PROFILE])
            else:
                cfg["federated"].pop("client_lora_ranks", None)
            cfg["dataset"].update(
                {
                    **paths,
                    "source": "mmlu_csv",
                    "split": "train",
                    "eval_split": "validation",
                    "num_clients": num_clients(),
                    "client_ids": CLIENT_IDS,
                    "batch_size": 8,
                    "jetson_batch_size": 1,
                    "max_seq_len": 64,
                    "partition_mode": "round_robin",
                    "dirichlet_alpha": 0.8,
                    "smoke_test_examples": 0,
                }
            )
            cfg["model"].update(
                {
                    "model_name_or_path": "gemma-3-270m",
                    "device": "cuda",
                    "dtype": "float32",
                    "target_embedding_mode": "unused_classic_fl",
                    "freeze_input_embeddings": True,
                    "grad_clip_norm": 1.0,
                    "learning_rate": 2.0e-4,
                    "weight_decay": 0.0,
                    "training_mode": "lora",
                    "lora_r": 8,
                    "lora_alpha": 32.0,
                    "lora_dropout": 0.0,
                    "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                    "use_bf16_activations": True,
                    "checkpoint_every": 1,
                    "checkpoint_mlp": True,
                    "mlp_chunk_size": 16,
                }
            )
            cfg["client"].update(
                {
                    "backend": "mobilefinetuner_cpp",
                    "client_mode": "classic_lora",
                    "upload_dtype": "float32",
                    "pad_to_max_seq_len": False,
                    "target_mode": "attn",
                    "use_bf16_activations": True,
                    "checkpoint_every": 1,
                    "checkpoint_mlp": True,
                    "mlp_chunk_size": 16,
                    "shard_max_resident_mb": 384,
                    "shard_quantize_fp16_on_disk": True,
                    "shard_quant_mode": "int8",
                    "shard_offload_dir": "__memory__",
                }
            )
            path = config_root / f"{run_name}.yaml"
            write_yaml(path, cfg)
            classic_configs[(ds.key, method)] = path

        cfg = load_yaml(ROOT / SPLIT_METHOD["template"])
        run_name = f"splitlora_gemma270m_{CONFIG_DEVICE_TAG}_{ds.key}_seq64_b8_r1_l3"
        cfg["runtime"]["run_name"] = run_name
        cfg["runtime"]["output_dir"] = f"outputs/runs/{run_name}/server"
        cfg["flower"].update(
            {
                "server_address": f"0.0.0.0:{SPLIT_METHOD['port']}",
                "num_rounds": 1,
                "min_available_clients": num_clients(),
                "min_fit_clients": num_clients(),
                "sample_clients": num_clients(),
                "round_timeout": 7200,
                "eval_every_rounds": 0,
                "client_wait_timeout": 3600,
            }
        )
        cfg["federated"].update(
            {
                "algorithm": "splitlora",
                "local_steps": 3,
                "aggregate_by_num_examples": False,
                "accept_failures": False,
            }
        )
        cfg["dataset"].pop("eval_source_path", None)
        cfg["dataset"].pop("final_test_source_path", None)
        cfg["dataset"].update(
            {
                "source_path": paths["source_path"],
                "source": "mmlu_csv",
                "split": "train",
                "eval_split": "validation",
                "num_clients": num_clients(),
                "client_ids": CLIENT_IDS,
                "batch_size": 8,
                "max_seq_len": 64,
                "partition_mode": "round_robin",
                "dirichlet_alpha": 0.8,
                "smoke_test_examples": 0,
            }
        )
        cfg.setdefault("model", {})["model_name_or_path"] = CLASSIC_SERVER_MODEL
        path = split_config_root / f"{run_name}.yaml"
        write_yaml(path, cfg)
        split_configs[ds.key] = path

    return {
        "android_specs": android_specs_path,
        "jetson_specs": jetson_specs_path,
        "split_specs": split_specs_path,
        "classic_configs": classic_configs,
        "split_configs": split_configs,
    }


def check_file(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"missing file: {path}")


def preflight() -> None:
    check_file(ADB)
    if not LOCAL_MODEL_DIR.is_dir():
        raise RuntimeError(f"missing model dir: {LOCAL_MODEL_DIR}")
    for name in ("model.safetensors", "tokenizer.json", "config.json"):
        check_file(LOCAL_MODEL_DIR / name)
    for item in DEVICE_PROFILES[DEVICE_PROFILE]:
        stage_dir = Path(item.get("local_stage_dir", str(ANDROID_STAGE_DIR.resolve())))
        if not stage_dir.is_absolute():
            stage_dir = ROOT / stage_dir
        check_file(stage_dir / "lshaped_flower_client")
        check_file(stage_dir / "libc++_shared.so")
    for ds in DATASETS:
        for split in ("train", "validation", "test"):
            check_file(ROOT / ds.data_dir / f"{ds.prefix}_{split}_mmlu.csv")
        check_file(ROOT / ds.prepare_script)


def prepare_android_power_state() -> None:
    for item in DEVICE_PROFILES[DEVICE_PROFILE]:
        commands = [
            "cmd power set-mode 0 >/dev/null 2>&1 || true",
            "cmd power set-adaptive-power-saver-enabled false >/dev/null 2>&1 || true",
            "input keyevent 224 >/dev/null 2>&1 || true",
            "svc power stayon true >/dev/null 2>&1 || true",
        ]
        if item.get("enable_fixed_performance"):
            commands.append("cmd power set-fixed-performance-mode-enabled true >/dev/null 2>&1 || true")
        subprocess.run(
            [str(ADB), "-s", item["serial"], "shell", "; ".join(commands)],
            cwd=str(ROOT),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=60,
            check=False,
        )


def start_android_power_keepalive() -> subprocess.Popen[str] | None:
    serials = [item["serial"] for item in DEVICE_PROFILES[DEVICE_PROFILE] if item.get("enable_fixed_performance")]
    if not serials:
        return None
    quoted_serials = " ".join(serials)
    commands = (
        "cmd power set-mode 0 >/dev/null 2>&1 || true; "
        "cmd power set-adaptive-power-saver-enabled false >/dev/null 2>&1 || true; "
        "cmd power set-fixed-performance-mode-enabled true >/dev/null 2>&1 || true; "
        "svc power stayon true >/dev/null 2>&1 || true"
    )
    script = (
        f"while kill -0 {os.getpid()} >/dev/null 2>&1; do "
        f"for serial in {quoted_serials}; do "
        f"{shlex_quote(str(ADB))} -s \"$serial\" shell {shlex_quote(commands)} >/dev/null 2>&1 || true; "
        "done; "
        "sleep 20; "
        "done"
    )
    return subprocess.Popen(
        ["bash", "-lc", script],
        cwd=str(ROOT),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        text=True,
    )


def shlex_quote(value: str) -> str:
    import shlex

    return shlex.quote(value)


def run_cmd(cmd: list[str], log_path: Path, timeout: int = 10800) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        log.write("$ " + " ".join(cmd) + "\n")
        log.flush()
        proc = subprocess.run(cmd, cwd=str(ROOT), stdout=log, stderr=subprocess.STDOUT, text=True, timeout=timeout)
    if proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}); see {log_path}")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_run(run_dir: Path) -> dict[str, Any]:
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
        failures.append(f"expected 1 summary row, got {len(summary_rows)}")
    if len(power_rows) != num_clients():
        failures.append(f"expected {num_clients()} power rows, got {len(power_rows)}")
    power_index = {row.get("client_id", ""): row for row in power_rows}
    for row in client_rows:
        cid = row.get("client_id", "")
        steps = int(float(row.get("steps_completed") or 0))
        if steps != 3:
            failures.append(f"{cid} steps_completed={steps}")
        try:
            step_times = json.loads(row.get("step_times_sec_json") or "[]")
        except json.JSONDecodeError:
            step_times = []
        if len(step_times) != 3 or any(float(v) <= 0.0 for v in step_times):
            failures.append(f"{cid} invalid step_times={row.get('step_times_sec_json')}")
        for field in ("download_time_sec", "upload_time_sec", "download_bytes", "upload_bytes", "transmitted_bytes"):
            if float(row.get(field) or 0) <= 0:
                failures.append(f"{cid} {field}={row.get(field)}")
        for field in ("avg_rss_mb", "peak_rss_mb"):
            if float(row.get(field) or 0) <= 0:
                failures.append(f"{cid} {field}={row.get(field)}")
        power = power_index.get(cid, {})
        if int(float(power.get("power_samples") or 0)) <= 0:
            failures.append(f"{cid} power_samples={power.get('power_samples')}")
        if power.get("avg_power_w", "") == "":
            failures.append(f"{cid} missing avg_power_w")
        elif float(power.get("avg_power_w") or 0.0) <= 0.0:
            failures.append(f"{cid} avg_power_w={power.get('avg_power_w')}")
        raw_power_path = run_dir / "clients" / cid / "power_samples.csv"
        if not raw_power_path.is_file():
            failures.append(f"{cid} missing raw power_samples.csv")
        else:
            raw_power_rows = read_csv(raw_power_path)
            if not raw_power_rows:
                failures.append(f"{cid} empty raw power_samples.csv")
            else:
                raw_power = raw_power_rows[0]
                if not raw_power.get("power_source"):
                    failures.append(f"{cid} missing power_source")
                if float(raw_power.get("avg_power_w") or 0.0) <= 0.0:
                    failures.append(f"{cid} raw avg_power_w={raw_power.get('avg_power_w')}")
    if summary_rows:
        summary = summary_rows[0]
        if float(summary.get("aggregation_time_sec") or 0) < 0:
            failures.append("negative aggregation_time_sec")
    return {
        "valid": not failures,
        "failures": "; ".join(failures),
        "client_rows": len(client_rows),
        "summary_rows": len(summary_rows),
        "power_rows": len(power_rows),
    }


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.is_file()
    with path.open("a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_rows(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_one(
    *,
    batch_id: str,
    ds: DatasetSpec,
    method: str,
    configs: dict[str, Any],
    deliverable_dir: Path,
    logs_dir: Path,
) -> dict[str, Any]:
    run_id = f"{batch_id}_gemma270m_{ds.key}_{method}_{RUN_DEVICE_TAG}"
    log_path = logs_dir / f"{ds.key}_{method}.log"
    if method in CLASSIC_METHODS:
        meta = CLASSIC_METHODS[method]
        run_dir = LROOT / "outputs" / "runs" / run_id
        cmd = [
            sys.executable,
            str(ROOT / "smoke" / "jetson_gemma3_fl_fedavg_20260415" / "run_boolq_fedavg_measurement.py"),
            "--base-config",
            str(configs["classic_configs"][(ds.key, method)]),
            "--android-client-specs-json",
            str(configs["android_specs"]),
            "--jetson-client-specs-json",
            str(configs["jetson_specs"]),
            "--prepare-boolq-script",
            ds.prepare_script,
            "--prepare-script-output-dir",
            ds.data_dir,
            "--prepare-script-model-dir",
            str(LOCAL_MODEL_DIR),
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
            CLASSIC_SERVER_MODEL,
            "--server-cuda-visible-devices",
            SERVER_CUDA_VISIBLE_DEVICES,
            "--local-model-dir",
            str(LOCAL_MODEL_DIR),
            "--server-port",
            str(meta["port"]),
            "--run-id",
            run_id,
            "--run-label",
            f"gemma270m_{ds.key}_{method}_{RUN_DEVICE_TAG}",
            "--adb-path",
            str(ADB),
            "--startup-wait-sec",
            "15",
            "--poll-interval-sec",
            "5",
            "--timeout-sec",
            "7200",
        ]
    else:
        run_dir = LROOT / "legacy_split" / "outputs" / "runs" / run_id
        cmd = [
            sys.executable,
            str(LROOT / "legacy_split" / "scripts" / "run_boolq_split_measurement.py"),
            "--base-config",
            str(configs["split_configs"][ds.key]),
            "--client-specs-json",
            str(configs["split_specs"]),
            "--prepare-boolq-script",
            ds.prepare_script,
            "--prepare-script-output-dir",
            ds.data_dir,
            "--prepare-script-model-dir",
            str(LOCAL_MODEL_DIR),
            "--prepare-script-seq-len",
            "64",
            "--run-id",
            run_id,
            "--run-label",
            f"gemma270m_{ds.key}_splitlora_{RUN_DEVICE_TAG}",
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
            "600",
            "--server-exit-timeout",
            "7200",
            "--client-exit-timeout",
            "7200",
            "--timeout-sec",
            "10800",
        ]

    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.time()
    run_cmd(cmd, log_path, timeout=10800)
    elapsed = time.time() - t0
    validation = validate_run(run_dir)
    row = {
        "dataset": ds.label,
        "dataset_key": ds.key,
        "method": method,
        "run_id": run_id,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "started_at": started,
        "elapsed_sec": round(elapsed, 3),
        **validation,
    }
    append_csv(deliverable_dir / "measurement_validation.csv", row)
    if not validation["valid"]:
        raise RuntimeError(f"validation failed for {run_id}: {validation['failures']}")
    return row


def build_final_tables(batch_id: str, deliverable_dir: Path) -> None:
    validation_path = deliverable_dir / "measurement_validation.csv"
    if not validation_path.is_file():
        return
    rows = read_csv(validation_path)
    summary_rows = []
    server_detail_rows = []
    client_detail_rows = []
    power_detail_rows = []
    for row in rows:
        run_dir = Path(row["run_dir"])
        summary = read_csv(run_dir / "server" / "summary_rounds_clean.csv")[0]
        clients = read_csv(run_dir / "server" / "round1_client_summary_clean.csv")
        power_rows = read_csv(run_dir / "server" / "power_summary.csv")
        summary_rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "run_id": row["run_id"],
                "valid": row["valid"],
                "num_clients": len(clients),
                "mean_step_time_sec": summary.get("mean_client_step_time_sec", summary.get("mean_step_time_sec", "")),
                "aggregation_time_sec": summary.get("aggregation_time_sec", ""),
                "mean_upload_time_sec": summary.get("mean_client_upload_time_sec", summary.get("mean_upload_time_sec", "")),
                "mean_download_time_sec": summary.get("mean_client_download_time_sec", summary.get("mean_download_time_sec", "")),
                "mean_avg_rss_mb": summary.get("mean_client_avg_rss_mb", summary.get("mean_avg_rss_mb", "")),
                "mean_peak_rss_mb": summary.get("mean_client_peak_rss_mb", summary.get("mean_peak_rss_mb", "")),
                "mean_client_power_w": summary.get("mean_client_power_w", ""),
                "total_transmitted_bytes": summary.get("total_transmitted_bytes", summary.get("mean_transmitted_bytes", "")),
                "eval_loss": summary.get("eval_loss", ""),
                "eval_accuracy": summary.get("eval_accuracy", ""),
            }
        )
        server_detail_rows.append({"dataset": row["dataset"], "method": row["method"], "run_id": row["run_id"], **summary})
        for client in clients:
            client_detail_rows.append({"dataset": row["dataset"], "method": row["method"], "run_id": row["run_id"], **client})
        for power in power_rows:
            raw_power_path = run_dir / "clients" / power.get("client_id", "") / "power_samples.csv"
            raw_power = read_csv(raw_power_path)[0] if raw_power_path.is_file() and read_csv(raw_power_path) else {}
            power_detail_rows.append(
                {
                    "dataset": row["dataset"],
                    "method": row["method"],
                    "run_id": row["run_id"],
                    **power,
                    **{f"raw_{key}": value for key, value in raw_power.items()},
                }
            )

    write_rows(
        deliverable_dir / f"{SUMMARY_PREFIX}_round1_summary.csv",
        summary_rows,
        [
            "dataset",
            "method",
            "run_id",
            "valid",
            "num_clients",
            "mean_step_time_sec",
            "aggregation_time_sec",
            "mean_upload_time_sec",
            "mean_download_time_sec",
            "mean_avg_rss_mb",
            "mean_peak_rss_mb",
            "mean_client_power_w",
            "total_transmitted_bytes",
        ],
    )
    write_rows(
        deliverable_dir / f"{SUMMARY_PREFIX}_server_round1_detail.csv",
        server_detail_rows,
        ["dataset", "method", "run_id", "round", "num_clients", "aggregation_time_sec"],
    )
    write_rows(
        deliverable_dir / f"{SUMMARY_PREFIX}_client_round1_detail.csv",
        client_detail_rows,
        [
            "dataset",
            "method",
            "run_id",
            "client_id",
            "steps_completed",
            "step_times_sec_json",
            "download_time_sec",
            "upload_time_sec",
            "download_bytes",
            "upload_bytes",
            "transmitted_bytes",
            "avg_rss_mb",
            "peak_rss_mb",
            "avg_power_w",
        ],
    )
    write_rows(
        deliverable_dir / f"{SUMMARY_PREFIX}_power_detail.csv",
        power_detail_rows,
        [
            "dataset",
            "method",
            "run_id",
            "client_id",
            "avg_power_w",
            "power_samples",
            "raw_power_source",
            "raw_duration_sec",
            "raw_avg_voltage_v",
            "raw_battery_capacity_mah",
            "raw_computed_drain_mah",
            "raw_checkin_total_mah",
            "raw_checkin_shell_uid_mah",
            "raw_power_source_mah",
            "raw_power_quality",
            "raw_power_flags",
        ],
    )

    readme = [
        f"# {README_TITLE}",
        "",
        f"- Batch id: `{batch_id}`",
        "- Devices: "
        + ", ".join(f"`{item['serial']}` (`{item['client_id']}`)" for item in DEVICE_PROFILES[DEVICE_PROFILE]),
        "- Model: Gemma 3 270M, MobileFinetuner on Android",
        "- Per run: `batch_size=8`, `max_seq_len=64`, `num_rounds=1`, `local_steps=3`",
        f"- Methods: {', '.join(sorted({row['method'] for row in rows}))}",
        "- Datasets: BoolQ, QNLI, PIQA, HellaSwag, SocialQA, ARC-E, WinoGrande",
        "",
        "## Files",
        "",
        "- `measurement_validation.csv`: per-run validation for all required fields.",
        f"- `{SUMMARY_PREFIX}_round1_summary.csv`: compact cross-run summary.",
        f"- `{SUMMARY_PREFIX}_server_round1_detail.csv`: per-round server metrics, including aggregation time.",
        f"- `{SUMMARY_PREFIX}_client_round1_detail.csv`: per-client step time, communication, RSS, loss/accuracy, and power.",
        f"- `{SUMMARY_PREFIX}_power_detail.csv`: per-client adb/batterystats power source and raw drain fields.",
        "- `logs/`: stdout/stderr logs for each run.",
        "",
        "Each run directory keeps `server/round1_client_summary_clean.csv`, `server/summary_rounds_clean.csv`, and `server/power_summary.csv`.",
        "",
    ]
    (deliverable_dir / "README.md").write_text("\n".join(readme), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-id", default=datetime.now().strftime("%Y%m%d_%H%M%S"))
    parser.add_argument("--device-profile", choices=sorted(DEVICE_PROFILES), default="two_phone")
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument("--only", nargs="*", default=[], help="Optional filters like boolq:fedavg or qnli:splitlora")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_device_profile(args.device_profile)
    preflight()
    configs = generate_configs()
    deliverable_dir = ROOT / "deliverables" / f"{DELIVERABLE_PREFIX}_{args.batch_id}"
    logs_dir = deliverable_dir / "logs"
    deliverable_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        deliverable_dir / "run_manifest.json",
        {
            "batch_id": args.batch_id,
            "devices": make_android_specs(),
            "datasets": [ds.__dict__ for ds in DATASETS],
            "classic_methods": CLASSIC_METHODS,
            "split_method": SPLIT_METHOD,
            "configs": {
                "android_specs": str(configs["android_specs"]),
                "jetson_specs": str(configs["jetson_specs"]),
                "split_specs": str(configs["split_specs"]),
            },
        },
    )
    if args.generate_only:
        print(f"generated configs under {LROOT / 'configs' / f'{CONFIG_DEVICE_TAG}_gemma270m'}")
        print(f"deliverable_dir={deliverable_dir}")
        return

    filters = set(args.only)
    keepalive_proc = start_android_power_keepalive()
    if keepalive_proc is not None:
        atexit.register(lambda: keepalive_proc.terminate())
    try:
        for ds in DATASETS:
            for method in ("fedavg", "fedprox", "flexlora", "splitlora"):
                key = f"{ds.key}:{method}"
                if filters and key not in filters:
                    continue
                prepare_android_power_state()
                print(f"[{DEVICE_PROFILE}] start {key}", flush=True)
                row = run_one(
                    batch_id=args.batch_id,
                    ds=ds,
                    method=method,
                    configs=configs,
                    deliverable_dir=deliverable_dir,
                    logs_dir=logs_dir,
                )
                print(f"[{DEVICE_PROFILE}] done {key} elapsed={row['elapsed_sec']}s", flush=True)
                build_final_tables(args.batch_id, deliverable_dir)
        build_final_tables(args.batch_id, deliverable_dir)
    finally:
        if keepalive_proc is not None:
            keepalive_proc.terminate()
    print(f"deliverable_dir={deliverable_dir}")


if __name__ == "__main__":
    main()
