from __future__ import annotations

import argparse
import json
import os
import socket
import stat
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import paramiko
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one server3 + Android-only Flower experiment")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--android-client-specs-json", required=True)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cuda-visible-devices", default="4")
    parser.add_argument("--server-address-host", default="10.200.14.82")
    parser.add_argument("--server-port", type=int, default=19080)
    parser.add_argument("--server-ssh-host", default="10.200.14.82")
    parser.add_argument("--server-ssh-username", default="AndyLu666")
    parser.add_argument("--server-remote-root", default="/home/AndyLu666/L-shaped-run-classic")
    parser.add_argument("--server-python", default="/home/AndyLu666/L-shaped-run-classic/.venv/bin/python3")
    parser.add_argument("--shared-client-dataset-local-csv", required=True)
    parser.add_argument(
        "--shared-client-dataset-remote-csv",
        default="/home/AndyLu666/L-shaped-run-classic/data/mmlu/official_mmlu_test_100.csv",
    )
    parser.add_argument("--total-num-clients", type=int, default=5)
    parser.add_argument("--default-android-stage-local-dir", default="outputs/android_client/arm64-v8a/mft")
    parser.add_argument("--default-android-model-local-dir", default="${MODEL_ROOT}/gemma-3-270m")
    parser.add_argument("--adb-path", default="")
    parser.add_argument("--skip-android-binary-push", action="store_true")
    parser.add_argument("--skip-android-model-push", action="store_true")
    parser.add_argument("--connect-max-attempts", type=int, default=8)
    parser.add_argument("--connect-ready-timeout-ms", type=int, default=15000)
    parser.add_argument("--connect-retry-delay-ms", type=int, default=5000)
    parser.add_argument("--server-wait-timeout", type=int, default=300)
    parser.add_argument("--server-exit-timeout", type=int, default=7200)
    parser.add_argument("--client-exit-timeout", type=int, default=3600)
    parser.add_argument("--android-launch-delay-sec", type=int, default=5)
    parser.add_argument("--sync-remote-results", action="store_true", default=True)
    parser.add_argument("--summarize", action="store_true", default=True)
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


def connect_server(host: str, username: str) -> paramiko.SSHClient:
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


def format_ssh_cli_target(host: str, username: str) -> str:
    if "@" in host:
        return host
    return f"{username}@{host}"


def upload_remote_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    local_path = local_path.resolve()
    if not local_path.is_file():
        raise RuntimeError(f"Missing local file for upload: {local_path}")
    remote_parent = Path(remote_path).as_posix().rsplit("/", 1)[0]
    if remote_parent:
        parts = [part for part in remote_parent.strip("/").split("/") if part]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else f"/{part}"
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current)
    sftp.put(str(local_path), remote_path)


def download_tree(sftp: paramiko.SFTPClient, remote_dir: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path = local_dir / entry.filename
        if stat.S_ISDIR(entry.st_mode):
            download_tree(sftp, remote_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))


def ensure_android_devices_ready(args: argparse.Namespace, serials: list[str]) -> None:
    adb = resolve_adb_path(args)
    for serial in serials:
        run_local([adb, "connect", serial], timeout=30, check=False)
    for serial in serials:
        proc = run_local([adb, "-s", serial, "get-state"], timeout=30, check=False)
        if proc.stdout.strip() != "device":
            raise RuntimeError(f"ADB device not ready: {serial} state={proc.stdout.strip()}")


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


def build_run_id(base_config: str, run_label: str | None, run_id: str | None) -> str:
    if run_id:
        return run_id
    stem = Path(base_config).stem
    label = run_label or stem
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{label}"


def main() -> None:
    args = parse_args()
    root = repo_root()
    base_config = (root / args.base_config).resolve() if not Path(args.base_config).is_absolute() else Path(args.base_config)
    if not base_config.is_file():
        raise RuntimeError(f"Missing base config: {base_config}")

    android_specs = json.loads(Path(args.android_client_specs_json).read_text(encoding="utf-8"))
    if not isinstance(android_specs, list) or not android_specs:
        raise RuntimeError("android-client-specs-json must be a non-empty JSON list")
    android_serials = [str(item["serial"]) for item in android_specs]
    ensure_android_devices_ready(args, android_serials)

    run_id = build_run_id(args.base_config, args.run_label, args.run_id)
    run_dir = root / "outputs" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    server_dir = run_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)

    cfg = yaml.safe_load(base_config.read_text(encoding="utf-8")) or {}
    cfg["runtime"]["run_name"] = run_id
    cfg["runtime"]["output_dir"] = f"outputs/runs/{run_id}/server"
    resolved_config = run_dir / "resolved_config.yaml"
    resolved_config.write_text(yaml.safe_dump(cfg, sort_keys=False, allow_unicode=True), encoding="utf-8")

    remote_run_root = f"{args.server_remote_root}/outputs/runs/{run_id}"
    remote_resolved_config = f"{remote_run_root}/resolved_config.yaml"
    local_dataset = resolve_local_path(root, args.shared_client_dataset_local_csv)
    ssh_cli_target = format_ssh_cli_target(args.server_ssh_host, args.server_ssh_username)

    server_client = connect_server(args.server_ssh_host, args.server_ssh_username)
    sftp = server_client.open_sftp()
    try:
        upload_remote_file(sftp, resolved_config, remote_resolved_config)
        upload_remote_file(sftp, local_dataset, args.shared_client_dataset_remote_csv)
    finally:
        sftp.close()
        server_client.close()

    run_local(
        ["ssh", ssh_cli_target, f"mkdir -p {remote_run_root} && fuser -k {args.server_port}/tcp >/dev/null 2>&1 || true"],
        timeout=30,
        check=False,
    )

    remote_command = (
        f"cd {args.server_remote_root} && "
        f"mkdir -p outputs/runs/{run_id}/server && "
        f"env PYTHONPATH={args.server_remote_root}/src CUDA_VISIBLE_DEVICES={args.cuda_visible_devices} "
        f"{args.server_python} -m lshaped.server.run_server --config {remote_resolved_config}"
    )
    server_log_path = server_dir / "server_ssh.log"
    server_log_handle = server_log_path.open("w", encoding="utf-8", newline="\n")
    server_proc = subprocess.Popen(
        ["ssh", ssh_cli_target, remote_command],
        stdout=server_log_handle,
        stderr=subprocess.STDOUT,
        text=True,
    )

    android_log_handle = None
    android_proc: subprocess.Popen[str] | None = None
    try:
        wait_for_port(args.server_address_host, args.server_port, args.server_wait_timeout)
        if args.android_launch_delay_sec > 0:
            time.sleep(args.android_launch_delay_sec)

        android_cmd = [
            sys.executable,
            str(root / "scripts" / "run_android_clients_only.py"),
            "--base-config",
            str(resolved_config),
            "--run-id",
            run_id,
            "--server-address",
            f"{args.server_address_host}:{args.server_port}",
            "--client-specs-json",
            args.android_client_specs_json,
            "--shared-client-dataset-local-csv",
            args.shared_client_dataset_local_csv,
            "--total-num-clients",
            str(args.total_num_clients),
            "--default-connect-max-attempts",
            str(args.connect_max_attempts),
            "--default-connect-ready-timeout-ms",
            str(args.connect_ready_timeout_ms),
            "--default-connect-retry-delay-ms",
            str(args.connect_retry_delay_ms),
            "--default-android-stage-local-dir",
            args.default_android_stage_local_dir,
            "--default-android-model-local-dir",
            args.default_android_model_local_dir,
            "--client-exit-timeout",
            str(args.client_exit_timeout),
        ]
        if args.adb_path:
            android_cmd.extend(["--adb-path", args.adb_path])
        if args.skip_android_binary_push:
            android_cmd.append("--skip-binary-push")
        if args.skip_android_model_push:
            android_cmd.append("--skip-model-push")

        android_log_path = run_dir / "android_launcher.log"
        android_log_handle = android_log_path.open("w", encoding="utf-8", newline="\n")
        android_proc = subprocess.Popen(android_cmd, stdout=android_log_handle, stderr=subprocess.STDOUT, text=True)
        android_rc = android_proc.wait(timeout=args.server_exit_timeout + args.client_exit_timeout)
        if android_rc != 0:
            raise RuntimeError(f"Android launcher failed with rc={android_rc}")

        server_rc = server_proc.wait(timeout=args.server_exit_timeout)
        if server_rc != 0:
            raise RuntimeError(f"Remote server exited with rc={server_rc}")
    finally:
        if android_proc is not None and android_proc.poll() is None:
            android_proc.kill()
        if android_log_handle is not None:
            android_log_handle.close()
        if server_proc.poll() is None:
            server_proc.kill()
        server_log_handle.close()
        run_local(
            ["ssh", ssh_cli_target, f"fuser -k {args.server_port}/tcp >/dev/null 2>&1 || true"],
            timeout=30,
            check=False,
        )

    if args.sync_remote_results:
        server_client = connect_server(args.server_ssh_host, args.server_ssh_username)
        sftp = server_client.open_sftp()
        try:
            for item in ("server", "resolved_config.yaml"):
                remote_path = f"{remote_run_root}/{item}"
                local_path = run_dir / item
                try:
                    attr = sftp.stat(remote_path)
                except FileNotFoundError:
                    continue
                if stat.S_ISDIR(attr.st_mode):
                    download_tree(sftp, remote_path, local_path)
                else:
                    local_path.parent.mkdir(parents=True, exist_ok=True)
                    sftp.get(remote_path, str(local_path))
        finally:
            sftp.close()
            server_client.close()

    if args.summarize and (run_dir / "server" / "metrics.csv").is_file():
        run_local(
            [sys.executable, str(root / "scripts" / "summarize_run_metrics.py"), "--run-dir", str(run_dir)],
            timeout=300,
            check=True,
        )

    manifest = {
        "run_id": run_id,
        "base_config": str(base_config),
        "resolved_config": str(resolved_config),
        "android_client_specs_json": args.android_client_specs_json,
        "server_address": f"{args.server_address_host}:{args.server_port}",
        "server_remote_root": args.server_remote_root,
        "server_python": args.server_python,
    }
    (run_dir / "android_only_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    main()
