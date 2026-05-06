from __future__ import annotations

import argparse
import hashlib
import json
import os
import posixpath
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass
from pathlib import Path

import yaml

ADB_RETRY_ATTEMPTS = 5
ADB_RETRY_DELAY_SEC = 2.0


@dataclass
class AndroidSpec:
    client_id: str
    client_index: int
    serial: str
    backend: str
    dataset_format: str
    device_root: str
    batch_size: int
    max_seq_len: int
    max_rounds: int
    synthetic_samples: int
    mock_hidden_size: int
    answer_prefix: str
    local_stage_dir: str
    local_model_dir: str
    local_dataset_csv: str = ""
    local_dataset_train_path: str = ""
    local_dataset_valid_path: str = ""
    local_dataset_test_path: str = ""
    remote_model_dir: str = ""
    lora_r: int | None = None
    grad_accum_steps: int | None = None
    supports_grad_accum: bool = True
    use_bf16_activations: bool | None = None
    checkpoint_every: int | None = None
    checkpoint_mlp: bool | None = None
    mlp_chunk_size: int | None = None
    shard_max_resident_mb: int | None = None
    shard_quantize_fp16_on_disk: bool | None = None
    shard_quant_mode: str | None = None
    shard_offload_dir: str | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch Android-only L-shaped clients against an already-running server")
    parser.add_argument("--base-config", default="")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--server-address", default="")
    parser.add_argument("--client-specs-json", required=True)
    parser.add_argument("--shared-client-dataset-local-csv", default="")
    parser.add_argument("--shared-client-dataset-local-train-path", default="")
    parser.add_argument("--shared-client-dataset-local-valid-path", default="")
    parser.add_argument("--shared-client-dataset-local-test-path", default="")
    parser.add_argument("--default-dataset-format", default="mmlu_csv")
    parser.add_argument("--total-num-clients", type=int, required=True)
    parser.add_argument("--default-android-device-root", default="/data/local/tmp/L-shaped")
    parser.add_argument("--default-android-stage-local-dir", default="")
    parser.add_argument("--default-android-model-local-dir", default="${MODEL_ROOT}/gemma-3-270m")
    parser.add_argument("--default-client-batch-size", type=int, default=2)
    parser.add_argument("--default-client-max-seq-len", type=int, default=128)
    parser.add_argument("--default-client-max-rounds", type=int, default=-1)
    parser.add_argument("--default-client-synthetic-samples", type=int, default=32)
    parser.add_argument("--default-client-mock-hidden-size", type=int, default=128)
    parser.add_argument("--default-client-answer-prefix", default=" ")
    parser.add_argument("--default-connect-max-attempts", type=int, default=8)
    parser.add_argument("--default-connect-ready-timeout-ms", type=int, default=15000)
    parser.add_argument("--default-connect-retry-delay-ms", type=int, default=5000)
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--client-exit-timeout", type=int, default=1800)
    parser.add_argument("--skip-binary-push", action="store_true")
    parser.add_argument("--skip-model-push", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_local_path(root: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def resolve_adb_path(args: argparse.Namespace) -> str:
    if args.adb_path:
        return args.adb_path
    env_adb = os.environ.get("ADB_PATH", "")
    if env_adb:
        return env_adb
    common = Path(r"C:\Program Files\ADB\platform-tools\adb.exe")
    if common.is_file():
        return str(common)
    return "adb"


def run_local(cmd: list[str], *, timeout: int = 1200, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    return proc


def is_adb_retryable(proc: subprocess.CompletedProcess[str]) -> bool:
    stderr = proc.stderr.lower()
    stdout = proc.stdout.lower()
    haystack = f"{stdout}\n{stderr}"
    return any(
        token in haystack
        for token in (
            "device offline",
            "device not found",
            "no devices/emulators found",
            "closed",
            "broken pipe",
        )
    )


def adb_reconnect(args: argparse.Namespace, serial: str) -> None:
    adb_bin = resolve_adb_path(args)
    try:
        run_local([adb_bin, "disconnect", serial], timeout=15, check=False)
    except subprocess.TimeoutExpired:
        print(f"[run_android_clients_only] adb disconnect timeout for {serial}", flush=True)
    try:
        if ":" in serial:
            run_local([adb_bin, "connect", serial], timeout=15, check=False)
        else:
            run_local([adb_bin, "start-server"], timeout=15, check=False)
    except subprocess.TimeoutExpired:
        print(f"[run_android_clients_only] adb reconnect timeout for {serial}", flush=True)


def adb(args: argparse.Namespace, serial: str, adb_args: list[str], *, timeout: int = 1200, check: bool = True) -> str:
    cmd = [resolve_adb_path(args), "-s", serial, *adb_args]
    last_proc: subprocess.CompletedProcess[str] | None = None
    for attempt in range(1, ADB_RETRY_ATTEMPTS + 1):
        try:
            proc = run_local(cmd, timeout=timeout, check=False)
        except subprocess.TimeoutExpired as exc:
            if attempt == ADB_RETRY_ATTEMPTS:
                if check:
                    raise RuntimeError(
                        f"Command timed out after {timeout}s: {' '.join(cmd)}\n"
                        f"stdout:\n{(exc.stdout or '')}\nstderr:\n{(exc.stderr or '')}"
                    ) from exc
                return ""
            adb_reconnect(args, serial)
            time.sleep(ADB_RETRY_DELAY_SEC)
            continue
        last_proc = proc
        if proc.returncode == 0:
            return proc.stdout.strip()
        if not is_adb_retryable(proc) or attempt == ADB_RETRY_ATTEMPTS:
            if check:
                raise RuntimeError(
                    f"Command failed ({proc.returncode}): {' '.join(cmd)}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
                )
            return proc.stdout.strip()
        adb_reconnect(args, serial)
        time.sleep(ADB_RETRY_DELAY_SEC)
    assert last_proc is not None
    if check:
        raise RuntimeError(
            f"Command failed ({last_proc.returncode}): {' '.join(cmd)}\nstdout:\n{last_proc.stdout}\nstderr:\n{last_proc.stderr}"
        )
    return last_proc.stdout.strip()


def adb_shell(args: argparse.Namespace, serial: str, command: str, *, timeout: int = 1200, check: bool = True) -> str:
    return adb(args, serial, ["shell", command], timeout=timeout, check=check)


def adb_remote_file_exists(args: argparse.Namespace, serial: str, remote_path: str) -> bool:
    out = adb_shell(
        args,
        serial,
        f"[ -f {shlex.quote(remote_path)} ] && echo YES || echo NO",
        timeout=60,
    )
    return out.splitlines()[-1].strip() == "YES"


def adb_remote_file_size(args: argparse.Namespace, serial: str, remote_path: str) -> int:
    out = adb_shell(
        args,
        serial,
        (
            f"if [ -f {shlex.quote(remote_path)} ]; then "
            f"wc -c < {shlex.quote(remote_path)}; "
            f"else echo -1; fi"
        ),
        timeout=120,
        check=False,
    )
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return -1
    try:
        return int(lines[-1])
    except ValueError:
        return -1


def adb_pull_if_exists(args: argparse.Namespace, serial: str, remote_path: str, local_path: Path) -> bool:
    exists = adb_shell(
        args,
        serial,
        f"[ -e {shlex.quote(remote_path)} ] && echo YES || echo NO",
        timeout=60,
    )
    if exists.splitlines()[-1].strip() != "YES":
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    adb(args, serial, ["pull", remote_path, str(local_path)], timeout=600)
    return True


def local_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def adb_sha256(args: argparse.Namespace, serial: str, remote_path: str) -> str:
    command = (
        f"if command -v sha256sum >/dev/null 2>&1; then "
        f"sha256sum {shlex.quote(remote_path)}; "
        f"elif command -v toybox >/dev/null 2>&1; then "
        f"toybox sha256sum {shlex.quote(remote_path)}; "
        f"else echo NO_SHA256; fi"
    )
    out = adb_shell(args, serial, command, timeout=120, check=False)
    lines = [line.strip() for line in out.splitlines() if line.strip()]
    if not lines:
        return ""
    first = lines[-1]
    if first == "NO_SHA256":
        return ""
    return first.split()[0]


def adb_push_dir_files(args: argparse.Namespace, serial: str, local_dir: Path, remote_dir: str, *, timeout: int = 3600) -> None:
    assert local_dir.is_dir(), f"Missing local directory: {local_dir}"
    for child in sorted(local_dir.iterdir()):
        if child.name == ".sync_complete":
            continue
        if child.is_dir():
            continue
        adb(args, serial, ["push", str(child), posixpath.join(remote_dir, child.name)], timeout=timeout)


def wait_android_exit(args: argparse.Namespace, serial: str, pid: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        try:
            out = adb_shell(
                args,
                serial,
                f"if kill -0 {shlex.quote(pid)} >/dev/null 2>&1; then echo RUNNING; else echo DONE; fi",
                timeout=30,
            )
        except RuntimeError:
            time.sleep(ADB_RETRY_DELAY_SEC)
            continue
        if "DONE" in out:
            return True
        time.sleep(1.0)
    return False


def build_client_command(
    *,
    binary_path: str,
    server_address: str,
    client_id: str,
    backend: str,
    client_index: int,
    num_clients: int,
    batch_size: int,
    max_seq_len: int,
    max_rounds: int,
    run_mode: str,
    split_layer: int,
    local_steps: int,
    local_epochs: int,
    grad_accum_steps: int | None,
    fedprox_mu: float,
    learning_rate: float,
    grad_clip_norm: float,
    weight_decay: float,
    logging_steps: int,
    target_mode: str,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
    lora_targets: str,
    answer_prefix: str,
    synthetic_samples: int,
    mock_hidden_size: int,
    connect_max_attempts: int,
    connect_ready_timeout_ms: int,
    connect_retry_delay_ms: int,
    metrics_path: str,
    dataset_format: str,
    dataset_csv: str,
    dataset_train_path: str,
    dataset_valid_path: str,
    dataset_test_path: str,
    model_dir: str = "",
    use_bf16_activations: bool = False,
    checkpoint_every: int = 0,
    checkpoint_mlp: bool = False,
    mlp_chunk_size: int = 0,
    shard_max_resident_mb: int = 0,
    shard_quantize_fp16_on_disk: bool = True,
    shard_quant_mode: str = "",
    shard_offload_dir: str = "",
) -> list[str]:
    parts = [
        binary_path,
        "--server_address",
        server_address,
        "--client_id",
        client_id,
        "--backend",
        backend,
        "--client_index",
        str(client_index),
        "--num_clients",
        str(num_clients),
        "--batch_size",
        str(batch_size),
        "--max_seq_len",
        str(max_seq_len),
        "--max_rounds",
        str(max_rounds),
        "--run_mode",
        run_mode,
        "--split_layer",
        str(split_layer),
        "--answer_prefix",
        answer_prefix,
        "--synthetic_samples",
        str(synthetic_samples),
        "--mock_hidden_size",
        str(mock_hidden_size),
        "--connect_max_attempts",
        str(connect_max_attempts),
        "--connect_ready_timeout_ms",
        str(connect_ready_timeout_ms),
        "--connect_retry_delay_ms",
        str(connect_retry_delay_ms),
        "--metrics_path",
        metrics_path,
        "--dataset_format",
        dataset_format,
        "--local_steps",
        str(local_steps),
        "--local_epochs",
        str(local_epochs),
        "--fedprox_mu",
        str(fedprox_mu),
        "--learning_rate",
        str(learning_rate),
        "--grad_clip_norm",
        str(grad_clip_norm),
        "--weight_decay",
        str(weight_decay),
        "--logging_steps",
        str(logging_steps),
        "--target_mode",
        target_mode,
        "--lora_r",
        str(lora_r),
        "--lora_alpha",
        str(lora_alpha),
        "--lora_dropout",
        str(lora_dropout),
        "--lora_targets",
        lora_targets,
    ]
    if grad_accum_steps is not None:
        parts.extend(["--grad_accum_steps", str(grad_accum_steps)])
    if model_dir:
        parts.extend(["--model_dir", model_dir])
    if use_bf16_activations:
        parts.extend(["--use_bf16_activations", "true"])
    if checkpoint_every > 0:
        parts.extend(["--checkpoint_every", str(checkpoint_every)])
    if checkpoint_mlp:
        parts.extend(["--checkpoint_mlp", "true"])
    if mlp_chunk_size > 0:
        parts.extend(["--mlp_chunk_size", str(mlp_chunk_size)])
    if shard_max_resident_mb > 0:
        parts.extend(["--shard_max_resident_mb", str(shard_max_resident_mb)])
        parts.extend(["--shard_quantize_fp16_on_disk", "true" if shard_quantize_fp16_on_disk else "false"])
        if shard_quant_mode:
            parts.extend(["--shard_quant_mode", shard_quant_mode])
        if shard_offload_dir:
            parts.extend(["--shard_offload_dir", shard_offload_dir])
    if dataset_csv:
        parts.extend(["--dataset_csv", dataset_csv])
    if dataset_train_path:
        parts.extend(["--dataset_train_path", dataset_train_path])
    if dataset_valid_path:
        parts.extend(["--dataset_valid_path", dataset_valid_path])
    if dataset_test_path:
        parts.extend(["--dataset_test_path", dataset_test_path])
    return parts


def load_specs(args: argparse.Namespace) -> list[AndroidSpec]:
    with open(args.client_specs_json, "r", encoding="utf-8") as handle:
        raw = json.load(handle)
    specs: list[AndroidSpec] = []
    for item in raw:
        if str(item.get("type", "android")) != "android":
            continue
        specs.append(
            AndroidSpec(
                client_id=str(item["client_id"]),
                client_index=int(item["client_index"]),
                serial=str(item["serial"]),
                backend=str(item.get("backend", "mock")),
                dataset_format=str(item.get("dataset_format", args.default_dataset_format)),
                device_root=str(item.get("device_root", args.default_android_device_root)),
                batch_size=int(item.get("batch_size", args.default_client_batch_size)),
                max_seq_len=int(item.get("max_seq_len", args.default_client_max_seq_len)),
                max_rounds=int(item.get("max_rounds", args.default_client_max_rounds)),
                synthetic_samples=int(item.get("synthetic_samples", args.default_client_synthetic_samples)),
                mock_hidden_size=int(item.get("mock_hidden_size", args.default_client_mock_hidden_size)),
                answer_prefix=str(item.get("answer_prefix", args.default_client_answer_prefix)),
                local_stage_dir=str(item.get("local_stage_dir", args.default_android_stage_local_dir)),
                local_model_dir=str(item.get("local_model_dir", args.default_android_model_local_dir)),
                local_dataset_csv=str(item.get("local_dataset_csv", "")),
                local_dataset_train_path=str(item.get("local_dataset_train_path", "")),
                local_dataset_valid_path=str(item.get("local_dataset_valid_path", "")),
                local_dataset_test_path=str(item.get("local_dataset_test_path", "")),
                remote_model_dir=str(item.get("remote_model_dir", "")),
                lora_r=int(item["lora_r"]) if item.get("lora_r") is not None else None,
                grad_accum_steps=(
                    int(item["grad_accum_steps"]) if item.get("grad_accum_steps") is not None else None
                ),
                supports_grad_accum=bool(item.get("supports_grad_accum", True)),
                use_bf16_activations=(
                    bool(item["use_bf16_activations"]) if item.get("use_bf16_activations") is not None else None
                ),
                checkpoint_every=(
                    int(item["checkpoint_every"]) if item.get("checkpoint_every") is not None else None
                ),
                checkpoint_mlp=bool(item["checkpoint_mlp"]) if item.get("checkpoint_mlp") is not None else None,
                mlp_chunk_size=int(item["mlp_chunk_size"]) if item.get("mlp_chunk_size") is not None else None,
                shard_max_resident_mb=(
                    int(item["shard_max_resident_mb"]) if item.get("shard_max_resident_mb") is not None else None
                ),
                shard_quantize_fp16_on_disk=(
                    bool(item["shard_quantize_fp16_on_disk"])
                    if item.get("shard_quantize_fp16_on_disk") is not None
                    else None
                ),
                shard_quant_mode=(
                    str(item["shard_quant_mode"]) if item.get("shard_quant_mode") is not None else None
                ),
                shard_offload_dir=(
                    str(item["shard_offload_dir"]) if item.get("shard_offload_dir") is not None else None
                ),
            )
        )
    if not specs:
        raise RuntimeError("No android specs found")
    return specs


def main() -> None:
    args = parse_args()
    if not args.prepare_only and not args.server_address:
        raise RuntimeError("--server-address is required unless --prepare-only is set")
    root = repo_root()
    run_dir = root / "outputs" / "runs" / args.run_id
    clients_dir = run_dir / "clients"
    clients_dir.mkdir(parents=True, exist_ok=True)
    specs = load_specs(args)
    base_config_path = Path(args.base_config).resolve() if args.base_config else run_dir / "resolved_config.yaml"
    base_cfg = {}
    if base_config_path.is_file():
        with base_config_path.open("r", encoding="utf-8") as handle:
            base_cfg = yaml.safe_load(handle) or {}
    federated_cfg = base_cfg.get("federated", {})
    model_cfg = base_cfg.get("model", {})
    dataset_cfg = base_cfg.get("dataset", {})
    logging_cfg = base_cfg.get("logging", {})
    client_cfg = base_cfg.get("client", {})
    default_local_steps = int(federated_cfg.get("local_steps", 1))
    default_local_epochs = int(federated_cfg.get("local_epochs", 0))
    default_grad_accum_steps = int(federated_cfg.get("grad_accum_steps", 1))
    default_fedprox_mu = float(federated_cfg.get("prox_mu", 0.0))
    default_learning_rate = float(model_cfg.get("learning_rate", 2e-4))
    default_grad_clip_norm = float(model_cfg.get("grad_clip_norm", 1.0))
    default_weight_decay = float(model_cfg.get("weight_decay", 0.0))
    default_logging_steps = int(logging_cfg.get("log_every_rounds", 1))
    default_client_mode = str(client_cfg.get("client_mode", "classic_lora"))
    default_run_mode = "split" if default_client_mode == "split_lora" else "train"
    default_split_layer = int(client_cfg.get("split_layer", 0))
    default_target_mode = str(client_cfg.get("target_mode", model_cfg.get("target_mode", "attn")))
    default_lora_targets = ",".join(model_cfg.get("lora_target_modules", []))
    default_lora_r = int(model_cfg.get("lora_r", 8))
    client_lora_ranks = {
        str(client_id): int(rank) for client_id, rank in federated_cfg.get("client_lora_ranks", {}).items()
    }
    default_lora_alpha = float(model_cfg.get("lora_alpha", 16.0))
    default_lora_dropout = float(model_cfg.get("lora_dropout", 0.0))
    default_use_bf16_activations = bool(
        client_cfg.get("use_bf16_activations", model_cfg.get("use_bf16_activations", False))
    )
    default_checkpoint_every = int(
        client_cfg.get("checkpoint_every", model_cfg.get("checkpoint_every", 0))
    )
    default_checkpoint_mlp = bool(
        client_cfg.get("checkpoint_mlp", model_cfg.get("checkpoint_mlp", False))
    )
    default_mlp_chunk_size = int(
        client_cfg.get("mlp_chunk_size", model_cfg.get("mlp_chunk_size", 0))
    )
    default_shard_max_resident_mb = int(
        client_cfg.get("shard_max_resident_mb", model_cfg.get("shard_max_resident_mb", 0))
    )
    default_shard_quantize_fp16_on_disk = bool(
        client_cfg.get(
            "shard_quantize_fp16_on_disk",
            model_cfg.get("shard_quantize_fp16_on_disk", True),
        )
    )
    default_shard_quant_mode = str(
        client_cfg.get("shard_quant_mode", model_cfg.get("shard_quant_mode", ""))
    )
    default_shard_offload_dir = str(
        client_cfg.get("shard_offload_dir", model_cfg.get("shard_offload_dir", ""))
    )
    sessions: list[dict[str, object]] = []
    try:
        for spec in specs:
            client_dir = clients_dir / spec.client_id
            client_dir.mkdir(parents=True, exist_ok=True)
            stage_dir: Path | None = None
            if not args.skip_binary_push:
                stage_dir = resolve_local_path(root, spec.local_stage_dir)
                if not (stage_dir / "lshaped_flower_client").is_file():
                    raise RuntimeError(f"Missing Android binary in {stage_dir}")
            model_dir = resolve_local_path(root, spec.local_model_dir) if spec.local_model_dir else None
            if spec.backend == "mft" and model_dir is None and not args.skip_model_push:
                raise RuntimeError(f"Missing local_model_dir for {spec.client_id}")

            adb_shell(args, spec.serial, f"mkdir -p {shlex.quote(spec.device_root)}")
            remote_bin_dir = posixpath.join(spec.device_root, "bin", spec.backend)
            remote_model_parent = posixpath.join(spec.device_root, "models")
            remote_run_client_dir = posixpath.join(spec.device_root, "outputs", "runs", args.run_id, spec.client_id)
            remote_metrics = posixpath.join(remote_run_client_dir, "client_metrics.csv")
            remote_log = posixpath.join(remote_run_client_dir, "client.log")
            remote_launch_script = posixpath.join(remote_run_client_dir, "launch_client.sh")

            # Clear stale Android client processes before launching a new run.
            adb_shell(
                args,
                spec.serial,
                textwrap.dedent(
                    """
                    for pid in $(pidof lshaped_flower_client 2>/dev/null || true); do
                      kill "$pid" >/dev/null 2>&1 || true
                    done
                    sleep 1
                    for pid in $(ps -A | grep '[l]shaped_flower_client' | awk '{print $2}'); do
                      kill -9 "$pid" >/dev/null 2>&1 || true
                    done
                    """
                ).strip(),
                timeout=120,
                check=False,
            )

            adb_shell(
                args,
                spec.serial,
                f"mkdir -p {shlex.quote(remote_bin_dir)} {shlex.quote(remote_model_parent)} {shlex.quote(remote_run_client_dir)}",
            )
            remote_binary = posixpath.join(remote_bin_dir, "lshaped_flower_client")
            if not args.skip_binary_push:
                assert stage_dir is not None
                local_binary = stage_dir / "lshaped_flower_client"
                push_binary = True
                if adb_remote_file_exists(args, spec.serial, remote_binary):
                    remote_hash = adb_sha256(args, spec.serial, remote_binary)
                    local_hash = local_sha256(local_binary)
                    push_binary = remote_hash != local_hash or not remote_hash
                if push_binary:
                    adb(args, spec.serial, ["push", str(local_binary), remote_binary], timeout=1800)
            else:
                assert adb_remote_file_exists(args, spec.serial, remote_binary), (
                    f"Missing remote binary for {spec.client_id}: {remote_binary}"
                )
            if stage_dir is not None and (stage_dir / "libc++_shared.so").is_file():
                remote_libcxx = posixpath.join(remote_bin_dir, "libc++_shared.so")
                if not args.skip_binary_push:
                    local_libcxx = stage_dir / "libc++_shared.so"
                    push_libcxx = True
                    if adb_remote_file_exists(args, spec.serial, remote_libcxx):
                        remote_hash = adb_sha256(args, spec.serial, remote_libcxx)
                        local_hash = local_sha256(local_libcxx)
                        push_libcxx = remote_hash != local_hash or not remote_hash
                    if push_libcxx:
                        adb(args, spec.serial, ["push", str(local_libcxx), remote_libcxx], timeout=1800)
                elif args.skip_binary_push:
                    assert adb_remote_file_exists(args, spec.serial, remote_libcxx), (
                        f"Missing remote runtime library for {spec.client_id}: {remote_libcxx}"
                    )
            remote_model_dir = spec.remote_model_dir
            if spec.backend == "mft":
                if not remote_model_dir:
                    model_name = model_dir.name if model_dir is not None else "gemma-3-270m"
                    remote_model_dir = posixpath.join(remote_model_parent, model_name)
                adb_shell(args, spec.serial, f"mkdir -p {shlex.quote(remote_model_dir)}")
                remote_model_file = posixpath.join(remote_model_dir, "model.safetensors")
                remote_tokenizer_file = posixpath.join(remote_model_dir, "tokenizer.json")
                remote_config_file = posixpath.join(remote_model_dir, "config.json")
                local_model_size = model_dir.joinpath("model.safetensors").stat().st_size if model_dir is not None else -1
                model_ready = (
                    adb_remote_file_exists(args, spec.serial, remote_model_file)
                    and adb_remote_file_size(args, spec.serial, remote_model_file) == local_model_size
                    and adb_remote_file_exists(args, spec.serial, remote_tokenizer_file)
                    and adb_remote_file_exists(args, spec.serial, remote_config_file)
                )
                if not args.skip_model_push and not model_ready:
                    assert model_dir is not None
                    adb_push_dir_files(args, spec.serial, model_dir, remote_model_dir, timeout=3600)
                elif args.skip_model_push:
                    assert model_ready, (
                        f"Missing remote model for {spec.client_id}: {remote_model_file}"
                    )
            adb_shell(args, spec.serial, f"chmod 755 {shlex.quote(remote_binary)}")

            dataset_csv_local = None
            dataset_train_local = None
            dataset_valid_local = None
            dataset_test_local = None
            remote_dataset_csv = ""
            remote_dataset_train = ""
            remote_dataset_valid = ""
            remote_dataset_test = ""
            if spec.dataset_format == "mmlu_csv":
                dataset_csv_raw = spec.local_dataset_csv or args.shared_client_dataset_local_csv
                if not dataset_csv_raw:
                    raise RuntimeError(f"Missing dataset csv for {spec.client_id}")
                dataset_csv_local = resolve_local_path(root, dataset_csv_raw)
                if not dataset_csv_local.is_file():
                    raise RuntimeError(f"Missing dataset csv: {dataset_csv_local}")
                remote_dataset_csv = posixpath.join(remote_run_client_dir, dataset_csv_local.name)
                adb(args, spec.serial, ["push", str(dataset_csv_local), remote_dataset_csv], timeout=600)
            elif spec.dataset_format == "wikitext_raw":
                train_raw = spec.local_dataset_train_path or args.shared_client_dataset_local_train_path
                valid_raw = spec.local_dataset_valid_path or args.shared_client_dataset_local_valid_path
                test_raw = spec.local_dataset_test_path or args.shared_client_dataset_local_test_path
                if not train_raw or not valid_raw or not test_raw:
                    raise RuntimeError(f"Missing WikiText paths for {spec.client_id}")
                dataset_train_local = resolve_local_path(root, train_raw)
                dataset_valid_local = resolve_local_path(root, valid_raw)
                dataset_test_local = resolve_local_path(root, test_raw)
                for local_path in (dataset_train_local, dataset_valid_local, dataset_test_local):
                    if not local_path.is_file():
                        raise RuntimeError(f"Missing WikiText file: {local_path}")
                remote_dataset_train = posixpath.join(remote_run_client_dir, dataset_train_local.name)
                remote_dataset_valid = posixpath.join(remote_run_client_dir, dataset_valid_local.name)
                remote_dataset_test = posixpath.join(remote_run_client_dir, dataset_test_local.name)
                adb(args, spec.serial, ["push", str(dataset_train_local), remote_dataset_train], timeout=600)
                adb(args, spec.serial, ["push", str(dataset_valid_local), remote_dataset_valid], timeout=600)
                adb(args, spec.serial, ["push", str(dataset_test_local), remote_dataset_test], timeout=600)
            else:
                raise RuntimeError(f"Unsupported dataset_format for {spec.client_id}: {spec.dataset_format}")

            if args.prepare_only:
                (client_dir / "staged_paths.json").write_text(
                    json.dumps(
                        {
                            "client_id": spec.client_id,
                            "serial": spec.serial,
                            "remote_binary": remote_binary,
                            "remote_model_dir": remote_model_dir,
                            "remote_dataset_csv": remote_dataset_csv,
                            "remote_dataset_train": remote_dataset_train,
                            "remote_dataset_valid": remote_dataset_valid,
                            "remote_dataset_test": remote_dataset_test,
                            "remote_run_client_dir": remote_run_client_dir,
                        },
                        indent=2,
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
                continue

            command = " ".join(
                shlex.quote(part)
                for part in build_client_command(
                    binary_path=remote_binary,
                    server_address=args.server_address,
                    client_id=spec.client_id,
                    backend=spec.backend,
                    client_index=spec.client_index,
                    num_clients=args.total_num_clients,
                    batch_size=spec.batch_size,
                    max_seq_len=spec.max_seq_len,
                    max_rounds=spec.max_rounds,
                    run_mode=default_run_mode,
                    split_layer=default_split_layer,
                    local_steps=default_local_steps,
                    local_epochs=default_local_epochs,
                    grad_accum_steps=(
                        None
                        if not spec.supports_grad_accum
                        else spec.grad_accum_steps
                        if spec.grad_accum_steps is not None
                        else default_grad_accum_steps
                    ),
                    fedprox_mu=default_fedprox_mu,
                    learning_rate=default_learning_rate,
                    grad_clip_norm=default_grad_clip_norm,
                    weight_decay=default_weight_decay,
                    logging_steps=default_logging_steps,
                    target_mode=default_target_mode,
                    lora_r=(
                        spec.lora_r
                        if spec.lora_r is not None
                        else client_lora_ranks.get(spec.client_id, default_lora_r)
                    ),
                    lora_alpha=default_lora_alpha,
                    lora_dropout=default_lora_dropout,
                    lora_targets=default_lora_targets,
                    answer_prefix=spec.answer_prefix,
                    synthetic_samples=spec.synthetic_samples,
                    mock_hidden_size=spec.mock_hidden_size,
                    connect_max_attempts=args.default_connect_max_attempts,
                    connect_ready_timeout_ms=args.default_connect_ready_timeout_ms,
                    connect_retry_delay_ms=args.default_connect_retry_delay_ms,
                    metrics_path=remote_metrics,
                    dataset_format=spec.dataset_format,
                    dataset_csv=remote_dataset_csv,
                    dataset_train_path=remote_dataset_train,
                    dataset_valid_path=remote_dataset_valid,
                    dataset_test_path=remote_dataset_test,
                    model_dir=remote_model_dir,
                    use_bf16_activations=(
                        spec.use_bf16_activations
                        if spec.use_bf16_activations is not None
                        else default_use_bf16_activations
                    ),
                    checkpoint_every=(
                        spec.checkpoint_every
                        if spec.checkpoint_every is not None
                        else default_checkpoint_every
                    ),
                    checkpoint_mlp=(
                        spec.checkpoint_mlp
                        if spec.checkpoint_mlp is not None
                        else default_checkpoint_mlp
                    ),
                    mlp_chunk_size=(
                        spec.mlp_chunk_size
                        if spec.mlp_chunk_size is not None
                        else default_mlp_chunk_size
                    ),
                    shard_max_resident_mb=(
                        spec.shard_max_resident_mb
                        if spec.shard_max_resident_mb is not None
                        else default_shard_max_resident_mb
                    ),
                    shard_quantize_fp16_on_disk=(
                        spec.shard_quantize_fp16_on_disk
                        if spec.shard_quantize_fp16_on_disk is not None
                        else default_shard_quantize_fp16_on_disk
                    ),
                    shard_quant_mode=(
                        spec.shard_quant_mode
                        if spec.shard_quant_mode is not None
                        else default_shard_quant_mode
                    ),
                    shard_offload_dir=(
                        spec.shard_offload_dir
                        if spec.shard_offload_dir is not None
                        else default_shard_offload_dir
                        if default_shard_offload_dir
                        else posixpath.join(remote_run_client_dir, "parameter_shards")
                    ),
                )
            )
            launch_script = "\n".join(
                [
                    "#!/system/bin/sh",
                    "set -eu",
                    f"cd {shlex.quote(remote_run_client_dir)}",
                    f"mkdir -p {shlex.quote(posixpath.join(remote_run_client_dir, 'tmp'))}",
                    f"export TMPDIR={shlex.quote(posixpath.join(remote_run_client_dir, 'tmp'))}",
                    f"export LD_LIBRARY_PATH={shlex.quote(remote_bin_dir)}:${{LD_LIBRARY_PATH:-}}",
                    "export OPS_GEMMA_TRACE_PROFILE=1",
                    "export OPS_GEMMA_TRACE_LIMIT=24",
                    f"exec {command} > {shlex.quote(remote_log)} 2>&1",
                    "",
                ]
            )
            local_launch_script = client_dir / "launch_client.sh"
            with local_launch_script.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(launch_script)
            adb(args, spec.serial, ["push", str(local_launch_script), remote_launch_script], timeout=600)
            adb_shell(args, spec.serial, f"chmod 755 {shlex.quote(remote_launch_script)}")
            launch_result = adb(
                args,
                spec.serial,
                [
                    "shell",
                    (
                        f"nohup /system/bin/sh {shlex.quote(remote_launch_script)} "
                        "> /dev/null 2>&1 < /dev/null & echo $!"
                    ),
                ],
                timeout=120,
            )
            pid = launch_result.splitlines()[-1].strip()
            if not pid:
                raise RuntimeError(f"Failed to start Android client {spec.client_id}: missing PID")
            (client_dir / "adb_session.log").write_text(
                json.dumps(
                    {
                        "client_id": spec.client_id,
                        "serial": spec.serial,
                        "pid": pid,
                        "remote_launch_script": remote_launch_script,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            sessions.append(
                {
                    "spec": spec,
                    "client_dir": client_dir,
                    "remote_log": remote_log,
                    "remote_metrics": remote_metrics,
                    "pid": pid,
                }
            )

        missing: list[str] = []
        if args.prepare_only:
            manifest = {
                "run_id": args.run_id,
                "mode": "prepare_only",
                "total_num_clients": args.total_num_clients,
                "clients": [
                    {
                        "client_id": spec.client_id,
                        "serial": spec.serial,
                        "backend": spec.backend,
                        "client_index": spec.client_index,
                    }
                    for spec in specs
                ],
                "missing_files": missing,
            }
            (run_dir / "android_manifest.json").write_text(
                json.dumps(manifest, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            print(f"run_id={args.run_id}")
            print(f"run_dir={run_dir}")
            print("prepare_only=true")
            return
        for session in sessions:
            spec = session["spec"]
            assert isinstance(spec, AndroidSpec)
            pid = str(session["pid"])
            if not wait_android_exit(args, spec.serial, pid, args.client_exit_timeout):
                missing.append(f"{spec.client_id}:client_timeout")
                try:
                    adb_shell(args, spec.serial, f"kill {shlex.quote(pid)} >/dev/null 2>&1 || true", timeout=30, check=False)
                except RuntimeError:
                    pass
            client_dir = Path(session["client_dir"])
            try:
                if not adb_pull_if_exists(args, spec.serial, str(session["remote_log"]), client_dir / "client.log"):
                    missing.append(f"{spec.client_id}:client.log")
            except RuntimeError:
                missing.append(f"{spec.client_id}:client.log")
            try:
                if not adb_pull_if_exists(args, spec.serial, str(session["remote_metrics"]), client_dir / "client_metrics.csv"):
                    missing.append(f"{spec.client_id}:client_metrics.csv")
            except RuntimeError:
                missing.append(f"{spec.client_id}:client_metrics.csv")

        manifest = {
            "run_id": args.run_id,
            "server_address": args.server_address,
            "total_num_clients": args.total_num_clients,
            "clients": [
                {
                    "client_id": spec.client_id,
                    "serial": spec.serial,
                    "backend": spec.backend,
                    "client_index": spec.client_index,
                }
                for spec in specs
            ],
            "missing_files": missing,
        }
        (run_dir / "android_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"run_id={args.run_id}")
        print(f"run_dir={run_dir}")
        if missing:
            print(f"missing_files={','.join(missing)}")
    except Exception as exc:
        print(f"[run_android_clients_only] {exc}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
