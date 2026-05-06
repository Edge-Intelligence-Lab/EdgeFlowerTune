from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko
import yaml


@dataclass
class NanoSpec:
    client_id: str
    client_index: int
    host: str
    username: str
    password: str
    backend: str
    remote_root: str
    model_dir: str
    batch_size: int
    max_seq_len: int
    max_rounds: int
    synthetic_samples: int
    mock_hidden_size: int
    answer_prefix: str


@dataclass
class AndroidSpec:
    client_id: str
    client_index: int
    serial: str
    backend: str
    device_root: str
    batch_size: int
    max_seq_len: int
    max_rounds: int
    synthetic_samples: int
    mock_hidden_size: int
    answer_prefix: str
    local_stage_dir: str
    remote_stage_dir: str
    local_model_dir: str
    remote_model_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one remote server model with a mix of Nano SSH clients and Android adb clients",
    )
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--client-specs-json", required=True)
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--cuda-visible-devices", default="4")
    parser.add_argument("--server-address-host", default="10.200.14.82")
    parser.add_argument("--server-ssh-host", default="GPU_3")
    parser.add_argument("--server-ssh-username", default="BESTTOOLBOX")
    parser.add_argument("--server-remote-root", default="/datapool/BESTTOOLBOX/L-shaped")
    parser.add_argument("--server-python", default="/home/BESTTOOLBOX/anaconda3/envs/oneshotLLM/bin/python")
    parser.add_argument("--shared-client-dataset-local-csv", default="")
    parser.add_argument("--shared-client-dataset-local-train-path", default="")
    parser.add_argument("--shared-client-dataset-local-valid-path", default="")
    parser.add_argument("--shared-client-dataset-local-test-path", default="")
    parser.add_argument(
        "--shared-client-dataset-remote-csv",
        default="/datapool/BESTTOOLBOX/L-shaped/data/mmlu/official_mmlu_test_100.csv",
    )
    parser.add_argument(
        "--shared-client-dataset-remote-train-path",
        default="/datapool/BESTTOOLBOX/L-shaped/data/wikitext2/wikitext-2-raw/wiki.train.raw",
    )
    parser.add_argument(
        "--shared-client-dataset-remote-valid-path",
        default="/datapool/BESTTOOLBOX/L-shaped/data/wikitext2/wikitext-2-raw/wiki.valid.raw",
    )
    parser.add_argument(
        "--shared-client-dataset-remote-test-path",
        default="/datapool/BESTTOOLBOX/L-shaped/data/wikitext2/wikitext-2-raw/wiki.test.raw",
    )
    parser.add_argument("--default-nano-username", default="jetson")
    parser.add_argument("--default-nano-password", default="")
    parser.add_argument("--default-nano-password-env", default="NANO_PASSWORD")
    parser.add_argument("--default-nano-remote-root", default="")
    parser.add_argument("--default-client-backend", choices=("mock", "mft"), default="mft")
    parser.add_argument("--default-client-batch-size", type=int, default=2)
    parser.add_argument("--default-client-max-seq-len", type=int, default=128)
    parser.add_argument("--default-client-max-rounds", type=int, default=-1)
    parser.add_argument("--default-client-synthetic-samples", type=int, default=32)
    parser.add_argument("--default-client-mock-hidden-size", type=int, default=128)
    parser.add_argument("--default-client-answer-prefix", default=" ")
    parser.add_argument("--default-android-device-root", default="/data/local/tmp/L-shaped")
    parser.add_argument("--default-android-stage-local-dir", default="")
    parser.add_argument(
        "--default-android-stage-remote-dir",
        default="/datapool/BESTTOOLBOX/L-shaped/outputs/android_client/arm64-v8a/mock",
    )
    parser.add_argument("--default-android-model-local-dir", default="")
    parser.add_argument(
        "--default-android-model-remote-dir",
        default="/datapool/BESTTOOLBOX/L-shaped/artifacts/client_bundles/gemma-3-270m",
    )
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--server-wait-timeout", type=int, default=180)
    parser.add_argument("--server-exit-timeout", type=int, default=1800)
    parser.add_argument("--client-exit-timeout", type=int, default=180)
    parser.add_argument("--connect-max-attempts", type=int, default=60)
    parser.add_argument("--connect-ready-timeout-ms", type=int, default=15000)
    parser.add_argument("--connect-retry-delay-ms", type=int, default=5000)
    parser.add_argument("--skip-android-binary-push", action="store_true")
    parser.add_argument("--skip-android-model-push", action="store_true")
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_existing_file(*candidates: Path) -> Path:
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise RuntimeError(
        "Missing local file: " + " | ".join(str(candidate) for candidate in candidates)
    )


def resolve_nano_password(args: argparse.Namespace) -> str:
    return args.default_nano_password or os.environ.get(args.default_nano_password_env, "")


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


def wait_for_port(host: str, port: int, timeout_sec: int) -> None:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(2.0)
            try:
                sock.connect((host, port))
                return
            except OSError:
                time.sleep(1.0)
    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def resolve_ssh_target(host: str, username: str) -> dict[str, Any]:
    resolved: dict[str, Any] = {
        "hostname": host,
        "username": username,
        "port": 22,
        "key_filenames": [],
    }
    ssh_config_path = Path.home() / ".ssh" / "config"
    if ssh_config_path.is_file():
        ssh_config = paramiko.SSHConfig()
        with ssh_config_path.open("r", encoding="utf-8", errors="ignore") as handle:
            ssh_config.parse(handle)
        entry = ssh_config.lookup(host)
        resolved["hostname"] = str(entry.get("hostname", resolved["hostname"]))
        resolved["username"] = str(entry.get("user", resolved["username"]))
        if "port" in entry:
            resolved["port"] = int(entry["port"])
        identity_files = entry.get("identityfile", [])
        if isinstance(identity_files, str):
            identity_files = [identity_files]
        for item in identity_files:
            candidate = Path(os.path.expandvars(os.path.expanduser(str(item).strip('"'))))
            if candidate.is_file():
                resolved["key_filenames"].append(str(candidate))
    default_key = Path.home() / ".ssh" / "id_rsa"
    if default_key.is_file() and str(default_key) not in resolved["key_filenames"]:
        resolved["key_filenames"].append(str(default_key))
    return resolved


def connect_ssh(host: str, username: str) -> paramiko.SSHClient:
    target = resolve_ssh_target(host, username)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        target["hostname"],
        username=target["username"],
        port=target["port"],
        timeout=20,
        key_filename=target["key_filenames"] or None,
        look_for_keys=True,
        allow_agent=True,
    )
    return client


def connect_nano(host: str, username: str, password: str) -> paramiko.SSHClient:
    if not password:
        return connect_ssh(host, username)
    last_error: Exception | None = None
    for attempt in range(5):
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(
                host,
                username=username,
                password=password,
                timeout=20,
                banner_timeout=30,
                auth_timeout=20,
                look_for_keys=False,
                allow_agent=False,
            )
            return client
        except (paramiko.SSHException, socket.timeout, OSError) as exc:
            last_error = exc
            client.close()
            if attempt == 4:
                break
            time.sleep(3)
    assert last_error is not None
    raise last_error


def run_remote(client: paramiko.SSHClient, command: str, *, get_pty: bool = False) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, get_pty=get_pty)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    parts = [part for part in remote_dir.strip("/").split("/") if part]
    current = ""
    for part in parts:
        current = f"{current}/{part}" if current else f"/{part}"
        try:
            sftp.stat(current)
        except FileNotFoundError:
            sftp.mkdir(current)


def upload_file_sftp(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    assert local_path.is_file(), f"Missing local file: {local_path}"
    sftp_mkdirs(sftp, posixpath.dirname(remote_path))
    sftp.put(str(local_path), remote_path)


def upload_repo_file_to_remote_root(
    sftp: paramiko.SFTPClient,
    remote_root: str,
    repo_root_path: Path,
    local_path: Path,
) -> None:
    local_path = local_path.resolve()
    repo_candidates = [repo_root_path.resolve(), repo_root_path.resolve().parent]
    relative = None
    for candidate in repo_candidates:
        try:
            relative = local_path.relative_to(candidate)
            break
        except ValueError:
            continue
    if relative is None:
        raise RuntimeError(f"Could not map local file into remote root: {local_path}")
    remote_path = posixpath.join(remote_root, relative.as_posix())
    upload_file_sftp(sftp, local_path, remote_path)


def fetch_file_sftp(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))


def download_tree_sftp(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = posixpath.join(remote_dir, entry.filename)
        local_path = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            download_tree_sftp(sftp, remote_path, local_path)
        else:
            fetch_file_sftp(sftp, remote_path, local_path)


def ensure_local_file_from_remote(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> Path:
    if not local_path.is_file():
        fetch_file_sftp(sftp, remote_path, local_path)
    return local_path


def ensure_local_dir_from_remote(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> Path:
    manifest = local_dir / ".sync_complete"
    if manifest.is_file():
        return local_dir
    if local_dir.exists():
        for child in sorted(local_dir.rglob("*"), reverse=True):
            if child.is_file():
                child.unlink()
            elif child.is_dir():
                child.rmdir()
    local_dir.mkdir(parents=True, exist_ok=True)
    download_tree_sftp(sftp, remote_dir, local_dir)
    manifest.write_text(remote_dir, encoding="utf-8")
    return local_dir


def adb(args: argparse.Namespace, serial: str, adb_args: list[str], *, timeout: int = 1200, check: bool = True) -> str:
    cmd = [resolve_adb_path(args), "-s", serial, *adb_args]
    proc = run_local(cmd, timeout=timeout, check=check)
    return proc.stdout.strip()


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


def adb_push_dir_files(args: argparse.Namespace, serial: str, local_dir: Path, remote_dir: str, *, timeout: int = 3600) -> None:
    assert local_dir.is_dir(), f"Missing local directory: {local_dir}"
    for child in sorted(local_dir.iterdir()):
        if child.name == ".sync_complete":
            continue
        if child.is_dir():
            raise RuntimeError(f"Nested directories are not supported for adb push: {child}")
        adb(args, serial, ["push", str(child), posixpath.join(remote_dir, child.name)], timeout=timeout)


def wait_android_exit(args: argparse.Namespace, serial: str, pid: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        out = adb_shell(
            args,
            serial,
            f"if kill -0 {shlex.quote(pid)} >/dev/null 2>&1; then echo RUNNING; else echo DONE; fi",
            timeout=30,
        )
        if "DONE" in out:
            return True
        time.sleep(1.0)
    return False


def wait_remote_exit(client: paramiko.SSHClient, pid: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        code, out, _ = run_remote(
            client,
            f"if ps -p {shlex.quote(pid)} >/dev/null 2>&1; then echo RUNNING; else echo DONE; fi",
        )
        if code == 0 and "DONE" in out:
            return True
        time.sleep(1.0)
    return False


def load_client_specs(args: argparse.Namespace) -> list[NanoSpec | AndroidSpec]:
    raw_specs = json.loads(Path(args.client_specs_json).read_text(encoding="utf-8"))
    assert isinstance(raw_specs, list) and raw_specs, "client-specs-json must be a non-empty JSON list"
    default_password = resolve_nano_password(args)
    specs: list[NanoSpec | AndroidSpec] = []
    for item in raw_specs:
        assert isinstance(item, dict), "Each client spec must be an object"
        kind = str(item.get("type", "")).strip().lower()
        backend = str(item.get("backend", args.default_client_backend))
        common: dict[str, Any] = {
            "client_id": str(item["client_id"]),
            "client_index": int(item["client_index"]),
            "backend": backend,
            "batch_size": int(item.get("batch_size", args.default_client_batch_size)),
            "max_seq_len": int(item.get("max_seq_len", args.default_client_max_seq_len)),
            "max_rounds": int(item.get("max_rounds", args.default_client_max_rounds)),
            "synthetic_samples": int(item.get("synthetic_samples", args.default_client_synthetic_samples)),
            "mock_hidden_size": int(item.get("mock_hidden_size", args.default_client_mock_hidden_size)),
            "answer_prefix": str(item.get("answer_prefix", args.default_client_answer_prefix)),
        }
        if kind == "nano":
            spec = NanoSpec(
                **common,
                host=str(item["host"]),
                username=str(item.get("username", args.default_nano_username)),
                password=str(item.get("password", default_password)),
                remote_root=str(item.get("remote_root", args.default_nano_remote_root or f"/home/{args.default_nano_username}/L-shaped")),
                model_dir=str(item.get("model_dir", "")),
            )
            if spec.backend == "mft":
                assert spec.model_dir, f"Missing model_dir for nano client {spec.client_id}"
            specs.append(spec)
            continue
        if kind == "android":
            spec = AndroidSpec(
                **common,
                serial=str(item["serial"]),
                device_root=str(item.get("device_root", args.default_android_device_root)),
                local_stage_dir=str(item.get("local_stage_dir", args.default_android_stage_local_dir)),
                remote_stage_dir=str(item.get("remote_stage_dir", args.default_android_stage_remote_dir)),
                local_model_dir=str(item.get("local_model_dir", args.default_android_model_local_dir)),
                remote_model_dir=str(item.get("remote_model_dir", args.default_android_model_remote_dir)),
            )
            if spec.backend == "mft":
                assert spec.local_model_dir or spec.remote_model_dir, f"Missing model dir for android client {spec.client_id}"
            specs.append(spec)
            continue
        raise RuntimeError(f"Unsupported client spec type: {kind}")
    return specs


def build_client_command(
    server_address: str,
    client_id: str,
    backend: str,
    dataset_format: str,
    client_index: int,
    num_clients: int,
    batch_size: int,
    max_seq_len: int,
    max_rounds: int,
    answer_prefix: str,
    synthetic_samples: int,
    mock_hidden_size: int,
    metrics_path: str,
    connect_max_attempts: int,
    connect_ready_timeout_ms: int,
    connect_retry_delay_ms: int,
    *,
    binary_path: str,
    model_dir: str = "",
    dataset_csv: str = "",
    dataset_train_path: str = "",
    dataset_valid_path: str = "",
    dataset_test_path: str = "",
) -> list[str]:
    parts = [
        binary_path,
        "--server_address",
        server_address,
        "--client_id",
        client_id,
        "--backend",
        backend,
        "--dataset_format",
        dataset_format,
        "--run_mode",
        "split",
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
        "--answer_prefix",
        answer_prefix,
        "--synthetic_samples",
        str(synthetic_samples),
        "--mock_hidden_size",
        str(mock_hidden_size),
        "--metrics_path",
        metrics_path,
        "--connect_max_attempts",
        str(connect_max_attempts),
        "--connect_ready_timeout_ms",
        str(connect_ready_timeout_ms),
        "--connect_retry_delay_ms",
        str(connect_retry_delay_ms),
    ]
    if backend == "mft":
        assert model_dir, "mft client requires model_dir"
        parts.extend(["--model_dir", model_dir])
    if dataset_csv:
        parts.extend(["--dataset_csv", dataset_csv])
    if dataset_train_path:
        parts.extend(["--dataset_train_path", dataset_train_path])
    if dataset_valid_path:
        parts.extend(["--dataset_valid_path", dataset_valid_path])
    if dataset_test_path:
        parts.extend(["--dataset_test_path", dataset_test_path])
    return parts


def main() -> None:
    args = parse_args()
    root = repo_root()
    base_config = (root / args.base_config).resolve() if not Path(args.base_config).is_absolute() else Path(args.base_config)
    assert base_config.is_file(), f"Missing base config: {base_config}"

    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8"))
    specs = load_client_specs(args)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or Path(base_config).stem
    run_id = args.run_id.strip() or f"{timestamp}_{label}"
    run_dir = root / "outputs" / "runs" / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    server_local_dir = run_dir / "server"
    clients_local_dir = run_dir / "clients"
    assets_local_dir = run_dir / "_cached_assets"
    server_local_dir.mkdir(parents=True, exist_ok=True)
    clients_local_dir.mkdir(parents=True, exist_ok=True)
    assets_local_dir.mkdir(parents=True, exist_ok=True)

    remote_run_root = posixpath.join(args.server_remote_root, "outputs", "runs", run_id)
    remote_server_dir = posixpath.join(remote_run_root, "server")
    cfg["runtime"]["run_name"] = run_id
    cfg["runtime"]["output_dir"] = remote_server_dir
    resolved_local = run_dir / "resolved_config.yaml"
    resolved_local.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")
    dataset_source = str(cfg.get("dataset", {}).get("source", "mmlu_csv")).strip().lower()

    server_address = str(cfg["flower"]["server_address"])
    assert ":" in server_address, f"Invalid server_address: {server_address}"
    server_port = int(server_address.rsplit(":", 1)[1])
    server_public_address = server_address.replace("0.0.0.0", args.server_address_host)

    server_client = connect_ssh(args.server_ssh_host, args.server_ssh_username)
    server_sftp = server_client.open_sftp()
    nano_sessions: list[dict[str, Any]] = []
    android_sessions: list[dict[str, Any]] = []
    server_pid = ""

    try:
        run_remote(server_client, f"rm -rf {shlex.quote(remote_run_root)}")
        sftp_mkdirs(server_sftp, remote_server_dir)
        remote_resolved = posixpath.join(remote_run_root, "resolved_config.yaml")
        upload_file_sftp(server_sftp, resolved_local, remote_resolved)
        for local_path in (
            resolve_existing_file(
                root.parent / "src" / "lshaped" / "common" / "protocol.py",
                root / "src" / "lshaped" / "common" / "protocol.py",
            ),
            resolve_existing_file(
                root.parent / "src" / "lshaped" / "server" / "gemma_suffix_trainer.py",
                root / "src" / "lshaped" / "server" / "gemma_suffix_trainer.py",
            ),
            resolve_existing_file(
                root.parent / "src" / "lshaped" / "server" / "activation_loss.py",
                root / "src" / "lshaped" / "server" / "activation_loss.py",
            ),
            resolve_existing_file(
                root.parent / "src" / "lshaped" / "server" / "split_strategy.py",
                root / "src" / "lshaped" / "server" / "split_strategy.py",
            ),
            resolve_existing_file(
                root.parent / "src" / "lshaped" / "server" / "run_server.py",
                root / "src" / "lshaped" / "server" / "run_server.py",
            ),
        ):
            upload_repo_file_to_remote_root(server_sftp, args.server_remote_root, root, local_path)

        remote_server_stdout = posixpath.join(remote_server_dir, "server_stdout.log")
        remote_server_stderr = posixpath.join(remote_server_dir, "server_stderr.log")
        server_cmd = (
            "set -e; "
            f"mkdir -p {shlex.quote(remote_server_dir)}; "
            f"cd {shlex.quote(args.server_remote_root)}; "
            "nohup env "
            f"CUDA_VISIBLE_DEVICES={shlex.quote(args.cuda_visible_devices)} "
            "PYTHONPATH=src "
            f"{shlex.quote(args.server_python)} -m lshaped.server.run_server --config {shlex.quote(remote_resolved)} "
            f"> {shlex.quote(remote_server_stdout)} 2> {shlex.quote(remote_server_stderr)} < /dev/null & "
            "echo $!"
        )
        code, out, err = run_remote(server_client, server_cmd)
        if code != 0:
            raise RuntimeError(f"Failed to start remote server:\nstdout:\n{out}\nstderr:\n{err}")
        server_pid = out.strip().splitlines()[-1].strip()
        assert server_pid, "Remote server did not return a PID"
        wait_for_port(args.server_address_host, server_port, args.server_wait_timeout)

        dataset_local: Path | None = None
        dataset_train_local: Path | None = None
        dataset_valid_local: Path | None = None
        dataset_test_local: Path | None = None
        if dataset_source == "wikitext_raw":
            dataset_train_local = (
                Path(args.shared_client_dataset_local_train_path).resolve()
                if args.shared_client_dataset_local_train_path
                else None
            )
            dataset_valid_local = (
                Path(args.shared_client_dataset_local_valid_path).resolve()
                if args.shared_client_dataset_local_valid_path
                else None
            )
            dataset_test_local = (
                Path(args.shared_client_dataset_local_test_path).resolve()
                if args.shared_client_dataset_local_test_path
                else None
            )
            if dataset_train_local is None:
                dataset_train_local = ensure_local_file_from_remote(
                    server_sftp,
                    args.shared_client_dataset_remote_train_path,
                    assets_local_dir / Path(args.shared_client_dataset_remote_train_path).name,
                )
            if dataset_valid_local is None:
                dataset_valid_local = ensure_local_file_from_remote(
                    server_sftp,
                    args.shared_client_dataset_remote_valid_path,
                    assets_local_dir / Path(args.shared_client_dataset_remote_valid_path).name,
                )
            if dataset_test_local is None:
                dataset_test_local = ensure_local_file_from_remote(
                    server_sftp,
                    args.shared_client_dataset_remote_test_path,
                    assets_local_dir / Path(args.shared_client_dataset_remote_test_path).name,
                )
            assert dataset_train_local.is_file(), f"Missing shared train raw file: {dataset_train_local}"
            assert dataset_valid_local.is_file(), f"Missing shared valid raw file: {dataset_valid_local}"
            assert dataset_test_local.is_file(), f"Missing shared test raw file: {dataset_test_local}"
        else:
            dataset_local = Path(args.shared_client_dataset_local_csv).resolve() if args.shared_client_dataset_local_csv else None
            if dataset_local is None:
                dataset_local = ensure_local_file_from_remote(
                    server_sftp,
                    args.shared_client_dataset_remote_csv,
                    assets_local_dir / Path(args.shared_client_dataset_remote_csv).name,
                )
            assert dataset_local.is_file(), f"Missing shared dataset CSV: {dataset_local}"

        android_stage_cache: dict[str, Path] = {}
        android_model_cache: dict[str, Path] = {}

        for spec in specs:
            local_client_dir = clients_local_dir / spec.client_id
            local_client_dir.mkdir(parents=True, exist_ok=True)
            if isinstance(spec, NanoSpec):
                client = connect_nano(spec.host, spec.username, spec.password)
                sftp = client.open_sftp()
                remote_run_client_dir = posixpath.join(spec.remote_root, "outputs", "runs", run_id, spec.client_id)
                remote_metrics = posixpath.join(remote_run_client_dir, "client_metrics.csv")
                remote_log = posixpath.join(remote_run_client_dir, "client.log")
                remote_pid_path = posixpath.join(remote_run_client_dir, "client.pid")
                remote_dataset = ""
                remote_dataset_train = ""
                remote_dataset_valid = ""
                remote_dataset_test = ""
                if dataset_source == "wikitext_raw":
                    assert dataset_train_local is not None and dataset_valid_local is not None and dataset_test_local is not None
                    remote_dataset_train = posixpath.join(remote_run_client_dir, dataset_train_local.name)
                    remote_dataset_valid = posixpath.join(remote_run_client_dir, dataset_valid_local.name)
                    remote_dataset_test = posixpath.join(remote_run_client_dir, dataset_test_local.name)
                    upload_file_sftp(sftp, dataset_train_local, remote_dataset_train)
                    upload_file_sftp(sftp, dataset_valid_local, remote_dataset_valid)
                    upload_file_sftp(sftp, dataset_test_local, remote_dataset_test)
                else:
                    assert dataset_local is not None
                    remote_dataset = posixpath.join(remote_run_client_dir, dataset_local.name)
                    upload_file_sftp(sftp, dataset_local, remote_dataset)
                binary = posixpath.join(
                    spec.remote_root,
                    "build",
                    "cpp_client_mft" if spec.backend == "mft" else "cpp_client_mock",
                    "lshaped_flower_client",
                )
                parts = [
                    shlex.quote(x)
                    for x in build_client_command(
                        server_address=server_public_address,
                        client_id=spec.client_id,
                        backend=spec.backend,
                        dataset_format=dataset_source,
                        client_index=spec.client_index,
                        num_clients=len(specs),
                        batch_size=spec.batch_size,
                        max_seq_len=spec.max_seq_len,
                        max_rounds=spec.max_rounds,
                        answer_prefix=spec.answer_prefix,
                        synthetic_samples=spec.synthetic_samples,
                        mock_hidden_size=spec.mock_hidden_size,
                        metrics_path=remote_metrics,
                        connect_max_attempts=args.connect_max_attempts,
                        connect_ready_timeout_ms=args.connect_ready_timeout_ms,
                        connect_retry_delay_ms=args.connect_retry_delay_ms,
                        binary_path=binary,
                        model_dir=spec.model_dir,
                        dataset_csv=remote_dataset,
                        dataset_train_path=remote_dataset_train,
                        dataset_valid_path=remote_dataset_valid,
                        dataset_test_path=remote_dataset_test,
                    )
                ]
                remote_cmd = (
                    "set -e; "
                    f"mkdir -p {shlex.quote(remote_run_client_dir)}; "
                    f"cd {shlex.quote(spec.remote_root)}; "
                    "nohup "
                    + " ".join(parts)
                    + f" > {shlex.quote(remote_log)} 2>&1 < /dev/null & echo $! | tee {shlex.quote(remote_pid_path)}"
                )
                code, out, err = run_remote(client, remote_cmd)
                (local_client_dir / "launch_stdout.txt").write_text(out, encoding="utf-8")
                (local_client_dir / "launch_stderr.txt").write_text(err, encoding="utf-8")
                if code != 0:
                    raise RuntimeError(f"Failed to launch nano client {spec.client_id}:\n{out}\n{err}")
                pid = out.strip().splitlines()[-1].strip()
                assert pid, f"Nano client {spec.client_id} did not return pid"
                nano_sessions.append(
                    {
                        "spec": spec,
                        "client": client,
                        "sftp": sftp,
                        "local_dir": local_client_dir,
                        "remote_log": remote_log,
                        "remote_metrics": remote_metrics,
                        "remote_pid_path": remote_pid_path,
                        "pid": pid,
                    }
                )
                continue

            assert isinstance(spec, AndroidSpec)
            stage_dir: Path | None = None
            if not args.skip_android_binary_push:
                if spec.local_stage_dir:
                    stage_dir = Path(spec.local_stage_dir).resolve()
                else:
                    stage_dir = android_stage_cache.get(spec.remote_stage_dir)
                    if stage_dir is None:
                        local_cache = assets_local_dir / "android_stage" / spec.backend
                        stage_dir = ensure_local_dir_from_remote(server_sftp, spec.remote_stage_dir, local_cache)
                        android_stage_cache[spec.remote_stage_dir] = stage_dir
                assert (stage_dir / "lshaped_flower_client").is_file(), f"Missing Android binary in {stage_dir}"

            model_dir: Path | None = None
            if spec.backend == "mft":
                if not args.skip_android_model_push:
                    if spec.local_model_dir:
                        model_dir = Path(spec.local_model_dir).resolve()
                    else:
                        model_dir = android_model_cache.get(spec.remote_model_dir)
                        if model_dir is None:
                            local_cache = assets_local_dir / "android_model" / Path(spec.remote_model_dir).name
                            model_dir = ensure_local_dir_from_remote(server_sftp, spec.remote_model_dir, local_cache)
                            android_model_cache[spec.remote_model_dir] = model_dir
                    assert (model_dir / "model.safetensors").is_file(), f"Missing Android model bundle in {model_dir}"

            adb_shell(args, spec.serial, f"mkdir -p {shlex.quote(spec.device_root)}")
            remote_bin_dir = posixpath.join(spec.device_root, "bin", spec.backend)
            remote_model_parent = posixpath.join(spec.device_root, "models")
            remote_run_client_dir = posixpath.join(spec.device_root, "outputs", "runs", run_id, spec.client_id)
            remote_dataset = ""
            remote_dataset_train = ""
            remote_dataset_valid = ""
            remote_dataset_test = ""
            remote_metrics = posixpath.join(remote_run_client_dir, "client_metrics.csv")
            remote_log = posixpath.join(remote_run_client_dir, "client.log")
            remote_pid_path = posixpath.join(remote_run_client_dir, "client.pid")

            adb_shell(args, spec.serial, f"rm -rf {shlex.quote(remote_run_client_dir)}")
            adb_shell(
                args,
                spec.serial,
                f"mkdir -p {shlex.quote(remote_bin_dir)} {shlex.quote(remote_model_parent)} {shlex.quote(remote_run_client_dir)}",
            )
            remote_binary = f"{remote_bin_dir}/lshaped_flower_client"
            if not args.skip_android_binary_push:
                assert stage_dir is not None
                local_binary = stage_dir / "lshaped_flower_client"
                adb(args, spec.serial, ["push", str(local_binary), remote_binary], timeout=1800)
            else:
                assert adb_remote_file_exists(args, spec.serial, remote_binary), (
                    f"Missing remote Android binary for {spec.client_id}: {remote_binary}"
                )
            if stage_dir is not None and (stage_dir / "libc++_shared.so").is_file():
                remote_libcxx = f"{remote_bin_dir}/libc++_shared.so"
                local_libcxx = stage_dir / "libc++_shared.so"
                if not args.skip_android_binary_push:
                    adb(args, spec.serial, ["push", str(local_libcxx), remote_libcxx], timeout=1800)
                elif args.skip_android_binary_push:
                    assert adb_remote_file_exists(args, spec.serial, remote_libcxx), (
                        f"Missing remote Android runtime library for {spec.client_id}: {remote_libcxx}"
                    )
            remote_model_dir = ""
            if spec.backend == "mft":
                model_name = model_dir.name if model_dir is not None else Path(spec.remote_model_dir).name
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
                if not args.skip_android_model_push and not model_ready:
                    assert model_dir is not None
                    adb_push_dir_files(args, spec.serial, model_dir, remote_model_dir, timeout=3600)
                elif args.skip_android_model_push:
                    assert model_ready, f"Missing remote Android model for {spec.client_id}: {remote_model_file}"
            if dataset_source == "wikitext_raw":
                assert dataset_train_local is not None and dataset_valid_local is not None and dataset_test_local is not None
                remote_dataset_train = posixpath.join(remote_run_client_dir, dataset_train_local.name)
                remote_dataset_valid = posixpath.join(remote_run_client_dir, dataset_valid_local.name)
                remote_dataset_test = posixpath.join(remote_run_client_dir, dataset_test_local.name)
                adb(args, spec.serial, ["push", str(dataset_train_local), remote_dataset_train], timeout=1200)
                adb(args, spec.serial, ["push", str(dataset_valid_local), remote_dataset_valid], timeout=1200)
                adb(args, spec.serial, ["push", str(dataset_test_local), remote_dataset_test], timeout=1200)
            else:
                assert dataset_local is not None
                remote_dataset = posixpath.join(remote_run_client_dir, dataset_local.name)
                adb(args, spec.serial, ["push", str(dataset_local), remote_dataset], timeout=600)
            adb_shell(
                args,
                spec.serial,
                f"chmod 755 {shlex.quote(posixpath.join(remote_bin_dir, 'lshaped_flower_client'))}",
            )

            cmd_list = build_client_command(
                server_address=server_public_address,
                client_id=spec.client_id,
                backend=spec.backend,
                dataset_format=dataset_source,
                client_index=spec.client_index,
                num_clients=len(specs),
                batch_size=spec.batch_size,
                max_seq_len=spec.max_seq_len,
                max_rounds=spec.max_rounds,
                answer_prefix=spec.answer_prefix,
                synthetic_samples=spec.synthetic_samples,
                mock_hidden_size=spec.mock_hidden_size,
                metrics_path=remote_metrics,
                connect_max_attempts=args.connect_max_attempts,
                connect_ready_timeout_ms=args.connect_ready_timeout_ms,
                connect_retry_delay_ms=args.connect_retry_delay_ms,
                binary_path=posixpath.join(remote_bin_dir, "lshaped_flower_client"),
                model_dir=remote_model_dir,
                dataset_csv=remote_dataset,
                dataset_train_path=remote_dataset_train,
                dataset_valid_path=remote_dataset_valid,
                dataset_test_path=remote_dataset_test,
            )
            command = " ".join(shlex.quote(x) for x in cmd_list)
            remote_launch_script = posixpath.join(remote_run_client_dir, "launch_client.sh")
            local_launch_script = local_client_dir / "launch_client.sh"
            launch_script = "\n".join(
                [
                    "#!/system/bin/sh",
                    "set -eu",
                    f"cd {shlex.quote(remote_run_client_dir)}",
                    f"mkdir -p {shlex.quote(posixpath.join(remote_run_client_dir, 'tmp'))}",
                    f"export TMPDIR={shlex.quote(posixpath.join(remote_run_client_dir, 'tmp'))}",
                    f"export LD_LIBRARY_PATH={shlex.quote(remote_bin_dir)}:${{LD_LIBRARY_PATH:-}}",
                    f"{command} > {shlex.quote(remote_log)} 2>&1 < /dev/null &",
                    f"echo $! | tee {shlex.quote(remote_pid_path)}",
                    "",
                ]
            )
            with local_launch_script.open("w", encoding="utf-8", newline="\n") as handle:
                handle.write(launch_script)
            adb(args, spec.serial, ["push", str(local_launch_script), remote_launch_script], timeout=600)
            adb_shell(args, spec.serial, f"chmod 755 {shlex.quote(remote_launch_script)}")
            pid = adb_shell(args, spec.serial, f"sh {shlex.quote(remote_launch_script)}", timeout=120)
            pid = pid.splitlines()[-1].strip()
            assert pid, f"Android client {spec.client_id} did not return pid"
            android_sessions.append(
                {
                    "spec": spec,
                    "local_dir": local_client_dir,
                    "remote_log": remote_log,
                    "remote_metrics": remote_metrics,
                    "remote_pid_path": remote_pid_path,
                    "pid": pid,
                }
            )

        deadline = time.time() + args.server_exit_timeout
        while time.time() < deadline:
            code, out, _ = run_remote(
                server_client,
                f"if ps -p {shlex.quote(server_pid)} >/dev/null 2>&1; then echo RUNNING; else echo DONE; fi",
            )
            if code == 0 and "DONE" in out:
                break
            time.sleep(2.0)
        else:
            run_remote(server_client, f"kill {shlex.quote(server_pid)} >/dev/null 2>&1 || true")
            raise TimeoutError(f"Timed out waiting for remote server pid {server_pid}")

        for session in nano_sessions:
            if not wait_remote_exit(session["client"], session["pid"], args.client_exit_timeout):
                run_remote(session["client"], f"kill {shlex.quote(session['pid'])} >/dev/null 2>&1 || true")

        for session in android_sessions:
            if not wait_android_exit(args, session["spec"].serial, session["pid"], args.client_exit_timeout):
                adb_shell(args, session["spec"].serial, f"kill {shlex.quote(session['pid'])} >/dev/null 2>&1 || true", check=False)

        for session in nano_sessions:
            fetch_file_sftp(session["sftp"], session["remote_log"], session["local_dir"] / "client.log")
            fetch_file_sftp(session["sftp"], session["remote_metrics"], session["local_dir"] / "client_metrics.csv")
            fetch_file_sftp(session["sftp"], session["remote_pid_path"], session["local_dir"] / "client.pid")

        for session in android_sessions:
            adb_pull_if_exists(args, session["spec"].serial, str(session["remote_log"]), session["local_dir"] / "client.log")
            adb_pull_if_exists(
                args,
                session["spec"].serial,
                str(session["remote_metrics"]),
                session["local_dir"] / "client_metrics.csv",
            )
            adb_pull_if_exists(args, session["spec"].serial, str(session["remote_pid_path"]), session["local_dir"] / "client.pid")

        download_tree_sftp(server_sftp, remote_run_root, run_dir / "_server_raw")
        raw_server_dir = run_dir / "_server_raw" / "server"
        if raw_server_dir.is_dir():
            for child in raw_server_dir.iterdir():
                target = server_local_dir / child.name
                if child.is_file():
                    child.replace(target)
                    continue
                target.mkdir(parents=True, exist_ok=True)
                for sub in child.rglob("*"):
                    relative = sub.relative_to(child)
                    dest = target / relative
                    if sub.is_dir():
                        dest.mkdir(parents=True, exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        sub.replace(dest)

        manifest = {
            "run_id": run_id,
            "server_remote_root": args.server_remote_root,
            "remote_run_root": remote_run_root,
            "server_pid": server_pid,
            "server_public_address": server_public_address,
            "resolved_config": str(resolved_local),
            "clients": [
                {
                    "client_id": spec.client_id,
                    "type": "nano" if isinstance(spec, NanoSpec) else "android",
                    "backend": spec.backend,
                    "client_index": spec.client_index,
                }
                for spec in specs
            ],
        }
        (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"run_id={run_id}")
        print(f"run_dir={run_dir}")
    finally:
        for session in nano_sessions:
            try:
                session["sftp"].close()
            except Exception:
                pass
            try:
                session["client"].close()
            except Exception:
                pass
        try:
            server_sftp.close()
        except Exception:
            pass
        try:
            server_client.close()
        except Exception:
            pass


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_mixed_client_experiment] {exc}", file=sys.stderr)
        raise
