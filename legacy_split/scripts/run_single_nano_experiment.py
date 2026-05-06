from __future__ import annotations

import argparse
import json
import os
import posixpath
import shlex
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import paramiko
import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one server + one Nano client experiment and collect logs")
    parser.add_argument("--base-config", required=True, help="Base YAML config path")
    parser.add_argument("--run-label", default=None, help="Optional suffix in the generated run id")
    parser.add_argument("--cuda-visible-devices", default="4")
    parser.add_argument("--client-host", required=True)
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--client-backend", choices=("mock", "mft"), default="mock")
    parser.add_argument("--client-model-dir", default="", help="Remote Nano model dir for backend=mft")
    parser.add_argument(
        "--client-dataset-csv",
        default="",
        help="Server-local dataset CSV to upload to the Nano and pass as --dataset_csv",
    )
    parser.add_argument(
        "--client-remote-dataset-csv",
        default="",
        help="Optional remote Nano dataset CSV path; defaults to <remote_run_root>/<filename>",
    )
    parser.add_argument("--client-username", default="jetson")
    parser.add_argument("--client-password", default=None)
    parser.add_argument("--client-password-env", default="NANO_PASSWORD")
    parser.add_argument("--client-remote-root", default=None)
    parser.add_argument("--client-index", type=int, default=None)
    parser.add_argument("--client-num-clients", type=int, default=None)
    parser.add_argument("--client-max-rounds", type=int, default=-1)
    parser.add_argument("--client-answer-prefix", default=" ")
    parser.add_argument("--client-batch-size", type=int, default=2)
    parser.add_argument("--client-max-seq-len", type=int, default=128)
    parser.add_argument("--client-synthetic-samples", type=int, default=32)
    parser.add_argument("--client-mock-hidden-size", type=int, default=128)
    parser.add_argument("--server-wait-timeout", type=int, default=120)
    parser.add_argument("--server-exit-timeout", type=int, default=120)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_password(args: argparse.Namespace) -> str:
    password = args.client_password or os.environ.get(args.client_password_env)
    if not password:
        raise RuntimeError(
            f"Missing client password: pass --client-password or set {args.client_password_env}"
        )
    return password


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
    raise TimeoutError(f"Timed out waiting for {host}:{port} to accept connections")


def connect(host: str, username: str, password: str) -> paramiko.SSHClient:
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    client.connect(
        host,
        username=username,
        password=password,
        timeout=20,
        look_for_keys=False,
        allow_agent=False,
    )
    return client


def run_remote(client: paramiko.SSHClient, command: str) -> tuple[int, str, str]:
    stdin, stdout, stderr = client.exec_command(command, get_pty=True)
    out = stdout.read().decode("utf-8", "replace")
    err = stderr.read().decode("utf-8", "replace")
    code = stdout.channel.recv_exit_status()
    return code, out, err


def fetch_remote_file(sftp: paramiko.SFTPClient, remote_path: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))


def upload_remote_file(sftp: paramiko.SFTPClient, local_path: Path, remote_path: str) -> None:
    local_path = local_path.resolve()
    if not local_path.is_file():
        raise RuntimeError(f"Missing local file for upload: {local_path}")
    remote_parent = posixpath.dirname(remote_path)
    if remote_parent:
        try:
            sftp.stat(remote_parent)
        except FileNotFoundError:
            parts = remote_parent.strip("/").split("/")
            current = ""
            for part in parts:
                current = f"{current}/{part}" if current else f"/{part}"
                try:
                    sftp.stat(current)
                except FileNotFoundError:
                    sftp.mkdir(current)
    sftp.put(str(local_path), remote_path)


def main() -> None:
    args = parse_args()
    root = repo_root()
    base_config = (root / args.base_config).resolve() if not Path(args.base_config).is_absolute() else Path(args.base_config)
    if not base_config.is_file():
        raise RuntimeError(f"Missing base config: {base_config}")

    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or Path(base_config).stem
    run_id = f"{timestamp}_{label}_{args.client_id}"
    run_dir = root / "outputs" / "runs" / run_id
    server_dir = run_dir / "server"
    client_dir = run_dir / "clients" / args.client_id
    server_dir.mkdir(parents=True, exist_ok=True)
    client_dir.mkdir(parents=True, exist_ok=True)

    cfg["runtime"]["run_name"] = run_id
    cfg["runtime"]["output_dir"] = str(server_dir)
    resolved_config = run_dir / "resolved_config.yaml"
    with open(resolved_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)

    server_address = str(cfg["flower"]["server_address"])
    if ":" not in server_address:
        raise RuntimeError(f"Invalid server_address in config: {server_address}")
    port = int(server_address.rsplit(":", 1)[1])
    wait_host = "127.0.0.1"

    server_stdout_path = server_dir / "server_stdout.log"
    server_stderr_path = server_dir / "server_stderr.log"
    client_dataset_csv = Path(args.client_dataset_csv).resolve() if args.client_dataset_csv else None
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    python_bin = "/home/BESTTOOLBOX/anaconda3/envs/oneshotLLM/bin/python"
    server_cmd = [
        python_bin,
        "-m",
        "lshaped.server.run_server",
        "--config",
        str(resolved_config),
    ]

    server_proc = None
    client = None
    try:
        with open(server_stdout_path, "w", encoding="utf-8") as stdout_f, open(
            server_stderr_path, "w", encoding="utf-8"
        ) as stderr_f:
            server_proc = subprocess.Popen(
                server_cmd,
                cwd=str(root),
                env=env,
                stdout=stdout_f,
                stderr=stderr_f,
            )

        wait_for_port(wait_host, port, args.server_wait_timeout)

        client_password = resolve_password(args)
        client_remote_root = args.client_remote_root or f"/home/{args.client_username}/L-shaped"
        remote_run_root = posixpath.join(client_remote_root, "outputs", "runs", run_id, args.client_id)
        remote_metrics_path = posixpath.join(remote_run_root, "client_metrics.csv")
        remote_log_path = posixpath.join(remote_run_root, "client.log")
        remote_dataset_csv = ""
        binary = (
            posixpath.join(client_remote_root, "build", "cpp_client_mft", "lshaped_flower_client")
            if args.client_backend == "mft"
            else posixpath.join(client_remote_root, "build", "cpp_client_mock", "lshaped_flower_client")
        )
        client_num_clients = args.client_num_clients or int(cfg.dataset.num_clients)
        if args.client_index is not None:
            client_index = args.client_index
        else:
            try:
                client_index = list(cfg.dataset.client_ids).index(args.client_id)
            except ValueError:
                client_index = 0
        command_parts = [
            shlex.quote(binary),
            "--server_address", shlex.quote(server_address.replace("0.0.0.0", "10.200.14.82")),
            "--client_id", shlex.quote(args.client_id),
            "--backend", shlex.quote(args.client_backend),
            "--client_index", str(client_index),
            "--num_clients", str(client_num_clients),
            "--batch_size", str(args.client_batch_size),
            "--max_seq_len", str(args.client_max_seq_len),
            "--split_layer", "0",
            "--max_rounds", str(args.client_max_rounds),
            "--answer_prefix", shlex.quote(args.client_answer_prefix),
            "--synthetic_samples", str(args.client_synthetic_samples),
            "--mock_hidden_size", str(args.client_mock_hidden_size),
            "--metrics_path", shlex.quote(remote_metrics_path),
        ]
        if args.client_backend == "mft":
            if not args.client_model_dir:
                raise RuntimeError("--client-model-dir is required when --client-backend=mft")
            command_parts.extend(["--model_dir", shlex.quote(args.client_model_dir)])

        client = connect(args.client_host, args.client_username, client_password)
        sftp = client.open_sftp()
        try:
            if client_dataset_csv is not None:
                remote_dataset_csv = args.client_remote_dataset_csv or posixpath.join(
                    remote_run_root, client_dataset_csv.name
                )
                upload_remote_file(sftp, client_dataset_csv, remote_dataset_csv)
            if remote_dataset_csv:
                command_parts.extend(["--dataset_csv", shlex.quote(remote_dataset_csv)])
            remote_cmd = (
                f"set -e; mkdir -p {shlex.quote(remote_run_root)}; "
                f"cd {shlex.quote(client_remote_root)}; "
                + " ".join(command_parts)
                + f" > {shlex.quote(remote_log_path)} 2>&1"
            )
            client_code, client_out, client_err = run_remote(client, remote_cmd)
            (client_dir / "client_exec_stdout.txt").write_text(client_out, encoding="utf-8")
            (client_dir / "client_exec_stderr.txt").write_text(client_err, encoding="utf-8")
            if client_code != 0:
                raise RuntimeError(f"Nano client exited with code {client_code}")
            fetch_remote_file(sftp, remote_log_path, client_dir / "client.log")
            fetch_remote_file(sftp, remote_metrics_path, client_dir / "client_metrics.csv")
        finally:
            sftp.close()

        if server_proc is not None:
            server_proc.wait(timeout=args.server_exit_timeout)
    finally:
        if client is not None:
            client.close()
        if server_proc is not None and server_proc.poll() is None:
            server_proc.send_signal(signal.SIGTERM)
            try:
                server_proc.wait(timeout=15)
            except subprocess.TimeoutExpired:
                server_proc.kill()

    manifest = {
        "run_id": run_id,
        "base_config": str(base_config),
        "resolved_config": str(resolved_config),
        "server_command": server_cmd,
        "server_stdout": str(server_stdout_path),
        "server_stderr": str(server_stderr_path),
        "client_host": args.client_host,
        "client_id": args.client_id,
        "client_backend": args.client_backend,
        "client_model_dir": args.client_model_dir,
        "client_dataset_csv": str(client_dataset_csv) if client_dataset_csv is not None else "",
        "client_remote_dataset_csv": remote_dataset_csv,
        "client_dir": str(client_dir),
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_single_nano_experiment] {exc}", file=sys.stderr)
        raise
