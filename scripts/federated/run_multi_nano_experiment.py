from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import posixpath
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import paramiko
import yaml


@dataclass
class RemoteClientSpec:
    client_id: str
    host: str
    client_index: int
    username: str = "jetson"
    password: str = ""
    backend: str = "mft"
    model_dir: str = ""
    remote_root: str = ""
    dataset_csv: str = ""
    remote_dataset_csv: str = ""
    batch_size: int = 2
    max_seq_len: int = 128
    max_rounds: int = -1
    synthetic_samples: int = 32
    mock_hidden_size: int = 128
    answer_prefix: str = " "
    checkpoint_every: int = 0
    checkpoint_mlp: bool = False
    use_bf16_activations: bool = False
    mlp_chunk_size: int = 0
    shard_max_resident_mb: int = 0
    shard_quantize_fp16_on_disk: bool = True
    shard_quant_mode: str = ""
    shard_offload_dir: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one server + multiple Nano clients and collect logs")
    parser.add_argument("--base-config", required=True)
    parser.add_argument("--client-specs-json", required=True)
    parser.add_argument("--run-label", default=None)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cuda-visible-devices", default="4")
    parser.add_argument("--server-python", default=sys.executable)
    parser.add_argument("--shared-client-dataset-csv", default="")
    parser.add_argument("--client-total-num-clients", type=int, default=0)
    parser.add_argument("--default-client-username", default="jetson")
    parser.add_argument("--default-client-password", default="")
    parser.add_argument("--default-client-password-env", default="NANO_PASSWORD")
    parser.add_argument("--default-client-backend", choices=("mock", "mft"), default="mft")
    parser.add_argument("--default-client-model-dir", default="")
    parser.add_argument("--default-client-remote-root", default="")
    parser.add_argument("--default-client-batch-size", type=int, default=2)
    parser.add_argument("--default-client-max-seq-len", type=int, default=128)
    parser.add_argument("--default-client-max-rounds", type=int, default=-1)
    parser.add_argument("--default-client-synthetic-samples", type=int, default=32)
    parser.add_argument("--default-client-mock-hidden-size", type=int, default=128)
    parser.add_argument("--default-client-answer-prefix", default=" ")
    parser.add_argument("--default-connect-max-attempts", type=int, default=8)
    parser.add_argument("--default-connect-ready-timeout-ms", type=int, default=15000)
    parser.add_argument("--default-connect-retry-delay-ms", type=int, default=5000)
    parser.add_argument("--server-wait-timeout", type=int, default=120)
    parser.add_argument("--server-exit-timeout", type=int, default=1800)
    parser.add_argument("--client-exit-timeout", type=int, default=60)
    parser.add_argument("--foreground-clients", action="store_true")
    parser.add_argument("--client-start-delay-seconds", type=int, default=0)
    return parser.parse_args()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def resolve_default_password(args: argparse.Namespace) -> str:
    return args.default_client_password or os.environ.get(args.default_client_password_env, "")


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
        parts = [part for part in remote_parent.strip("/").split("/") if part]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else f"/{part}"
            try:
                sftp.stat(current)
            except FileNotFoundError:
                sftp.mkdir(current)
    sftp.put(str(local_path), remote_path)


def load_client_specs(args: argparse.Namespace, cfg: dict) -> list[RemoteClientSpec]:
    with open(args.client_specs_json, "r", encoding="utf-8") as f:
        raw_specs = json.load(f)
    if not isinstance(raw_specs, list) or not raw_specs:
        raise RuntimeError("client-specs-json must be a non-empty JSON list")

    shared_dataset_csv = Path(args.shared_client_dataset_csv).resolve() if args.shared_client_dataset_csv else None
    default_password = resolve_default_password(args)
    specs: list[RemoteClientSpec] = []
    for item in raw_specs:
        if not isinstance(item, dict):
            raise RuntimeError("Each client spec must be a JSON object")
        client_id = str(item["client_id"])
        host = str(item["host"])
        client_index = int(item["client_index"])
        username = str(item.get("username", args.default_client_username))
        password = str(item.get("password", default_password))
        backend = str(item.get("backend", args.default_client_backend))
        model_dir = str(item.get("model_dir", args.default_client_model_dir))
        remote_root = str(item.get("remote_root", args.default_client_remote_root or f"/home/{username}/L-shaped"))
        dataset_csv = str(item.get("dataset_csv", shared_dataset_csv if shared_dataset_csv else ""))
        specs.append(
            RemoteClientSpec(
                client_id=client_id,
                host=host,
                client_index=client_index,
                username=username,
                password=password,
                backend=backend,
                model_dir=model_dir,
                remote_root=remote_root,
                dataset_csv=str(dataset_csv) if dataset_csv else "",
                remote_dataset_csv=str(item.get("remote_dataset_csv", "")),
                batch_size=int(item.get("batch_size", args.default_client_batch_size)),
                max_seq_len=int(item.get("max_seq_len", args.default_client_max_seq_len)),
                max_rounds=int(item.get("max_rounds", args.default_client_max_rounds)),
                synthetic_samples=int(item.get("synthetic_samples", args.default_client_synthetic_samples)),
                mock_hidden_size=int(item.get("mock_hidden_size", args.default_client_mock_hidden_size)),
                answer_prefix=str(item.get("answer_prefix", args.default_client_answer_prefix)),
                checkpoint_every=int(item.get("checkpoint_every", 0)),
                checkpoint_mlp=bool(item.get("checkpoint_mlp", False)),
                use_bf16_activations=bool(item.get("use_bf16_activations", False)),
                mlp_chunk_size=int(item.get("mlp_chunk_size", 0)),
                shard_max_resident_mb=int(item.get("shard_max_resident_mb", 0)),
                shard_quantize_fp16_on_disk=bool(item.get("shard_quantize_fp16_on_disk", True)),
                shard_quant_mode=str(item.get("shard_quant_mode", "")),
                shard_offload_dir=str(item.get("shard_offload_dir", "")),
            )
        )
    for spec in specs:
        if not spec.password:
            raise RuntimeError(f"Missing password for client {spec.client_id}")
        if spec.backend == "mft" and not spec.model_dir:
            raise RuntimeError(f"Missing model_dir for mft client {spec.client_id}")
    return specs


def wait_remote_exit(client: paramiko.SSHClient, pid: str, timeout_sec: int) -> bool:
    deadline = time.time() + timeout_sec
    while time.time() < deadline:
        code, out, err = run_remote(client, f"if ps -p {shlex.quote(pid)} >/dev/null 2>&1; then echo RUNNING; else echo DONE; fi")
        if "DONE" in out:
            return True
        time.sleep(1.0)
    return False


def main() -> None:
    args = parse_args()
    root = repo_root()
    base_config = (root / args.base_config).resolve() if not Path(args.base_config).is_absolute() else Path(args.base_config)
    if not base_config.is_file():
        raise RuntimeError(f"Missing base config: {base_config}")

    with open(base_config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    client_specs = load_client_specs(args, cfg)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = args.run_label or Path(base_config).stem
    run_id = args.run_id or f"{timestamp}_{label}"
    run_dir = root / "outputs" / "runs" / run_id
    server_dir = run_dir / "server"
    server_dir.mkdir(parents=True, exist_ok=True)
    for spec in client_specs:
        (run_dir / "clients" / spec.client_id).mkdir(parents=True, exist_ok=True)

    total_num_clients = args.client_total_num_clients or len(client_specs)
    if total_num_clients <= 0:
        raise RuntimeError("client-total-num-clients must be > 0")

    cfg["runtime"]["run_name"] = run_id
    cfg["runtime"]["output_dir"] = str(server_dir)
    resolved_config = run_dir / "resolved_config.yaml"
    with open(resolved_config, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
    federated_cfg = cfg.get("federated", {})
    model_cfg = cfg.get("model", {})
    dataset_cfg = cfg.get("dataset", {})
    logging_cfg = cfg.get("logging", {})
    client_cfg = cfg.get("client", {})
    default_local_steps = int(federated_cfg.get("local_steps", 1))
    default_local_epochs = int(federated_cfg.get("local_epochs", 0))
    default_grad_accum_steps = int(federated_cfg.get("grad_accum_steps", 1))
    default_batch_size = int(dataset_cfg.get("batch_size", args.default_client_batch_size))
    default_max_seq_len = int(dataset_cfg.get("max_seq_len", args.default_client_max_seq_len))
    default_learning_rate = float(model_cfg.get("learning_rate", 2e-4))
    default_weight_decay = float(model_cfg.get("weight_decay", 0.0))
    default_grad_clip_norm = float(model_cfg.get("grad_clip_norm", 1.0))
    default_logging_steps = int(logging_cfg.get("log_every_rounds", 1))
    default_target_mode = str(client_cfg.get("target_mode", model_cfg.get("target_mode", "attn")))
    default_lora_targets = ",".join(model_cfg.get("lora_target_modules", []))
    default_lora_r = int(model_cfg.get("lora_r", 8))
    default_lora_alpha = float(model_cfg.get("lora_alpha", 16.0))
    default_lora_dropout = float(model_cfg.get("lora_dropout", 0.0))

    server_address = str(cfg["flower"]["server_address"])
    if ":" not in server_address:
        raise RuntimeError(f"Invalid server_address in config: {server_address}")
    port = int(server_address.rsplit(":", 1)[1])

    server_stdout_path = server_dir / "server_stdout.log"
    server_stderr_path = server_dir / "server_stderr.log"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = str(root / "src")
    server_cmd = [args.server_python, "-m", "lshaped.server.run_server", "--config", str(resolved_config)]

    server_proc = None
    sessions: list[dict] = []
    executor: concurrent.futures.ThreadPoolExecutor | None = None
    try:
        with open(server_stdout_path, "w", encoding="utf-8") as stdout_f, open(server_stderr_path, "w", encoding="utf-8") as stderr_f:
            server_proc = subprocess.Popen(server_cmd, cwd=str(root), env=env, stdout=stdout_f, stderr=stderr_f)

        wait_for_port("127.0.0.1", port, args.server_wait_timeout)
        if args.client_start_delay_seconds > 0:
            print(f"[run_multi_nano_experiment] sleeping {args.client_start_delay_seconds}s before launching clients")
            time.sleep(args.client_start_delay_seconds)
        if args.foreground_clients:
            executor = concurrent.futures.ThreadPoolExecutor(max_workers=len(client_specs))

        for spec in client_specs:
            client_dir = run_dir / "clients" / spec.client_id
            remote_run_root = posixpath.join(spec.remote_root, "outputs", "runs", run_id, spec.client_id)
            remote_metrics_path = posixpath.join(remote_run_root, "client_metrics.csv")
            remote_log_path = posixpath.join(remote_run_root, "client.log")
            remote_pid_path = posixpath.join(remote_run_root, "client.pid")
            remote_dataset_csv = spec.remote_dataset_csv
            binary = (
                posixpath.join(spec.remote_root, "build", "cpp_client_mft", "lshaped_flower_client")
                if spec.backend == "mft"
                else posixpath.join(spec.remote_root, "build", "cpp_client_mock", "lshaped_flower_client")
            )

            client = connect(spec.host, spec.username, spec.password)
            sftp = client.open_sftp()
            if spec.dataset_csv:
                local_dataset_csv = Path(spec.dataset_csv).resolve()
                remote_dataset_csv = remote_dataset_csv or posixpath.join(remote_run_root, local_dataset_csv.name)
                upload_remote_file(sftp, local_dataset_csv, remote_dataset_csv)

            command_parts = [
                shlex.quote(binary),
                "--server_address", shlex.quote(server_address.replace("0.0.0.0", "10.200.14.82")),
                "--client_id", shlex.quote(spec.client_id),
                "--backend", shlex.quote(spec.backend),
                "--client_index", str(spec.client_index),
                "--num_clients", str(total_num_clients),
                "--batch_size", str(default_batch_size),
                "--max_seq_len", str(default_max_seq_len),
                "--max_rounds", str(spec.max_rounds),
                "--local_steps", str(default_local_steps),
                "--local_epochs", str(default_local_epochs),
                "--grad_accum_steps", str(default_grad_accum_steps),
                "--learning_rate", str(default_learning_rate),
                "--grad_clip_norm", str(default_grad_clip_norm),
                "--weight_decay", str(default_weight_decay),
                "--logging_steps", str(default_logging_steps),
                "--target_mode", shlex.quote(default_target_mode),
                "--lora_r", str(default_lora_r),
                "--lora_alpha", str(default_lora_alpha),
                "--lora_dropout", str(default_lora_dropout),
                "--answer_prefix", shlex.quote(spec.answer_prefix),
                "--synthetic_samples", str(spec.synthetic_samples),
                "--mock_hidden_size", str(spec.mock_hidden_size),
                "--connect_max_attempts", str(args.default_connect_max_attempts),
                "--connect_ready_timeout_ms", str(args.default_connect_ready_timeout_ms),
                "--connect_retry_delay_ms", str(args.default_connect_retry_delay_ms),
                "--metrics_path", shlex.quote(remote_metrics_path),
            ]
            if default_lora_targets:
                command_parts.extend(["--lora_targets", shlex.quote(default_lora_targets)])
            if spec.backend == "mft":
                command_parts.extend(["--model_dir", shlex.quote(spec.model_dir)])
                if spec.checkpoint_every > 0:
                    command_parts.extend(["--checkpoint_every", str(spec.checkpoint_every)])
                if spec.checkpoint_mlp:
                    command_parts.extend(["--checkpoint_mlp", "true"])
                if spec.use_bf16_activations:
                    command_parts.extend(["--use_bf16_activations", "true"])
                if spec.mlp_chunk_size > 0:
                    command_parts.extend(["--mlp_chunk_size", str(spec.mlp_chunk_size)])
                if spec.shard_max_resident_mb > 0:
                    command_parts.extend(["--shard_max_resident_mb", str(spec.shard_max_resident_mb)])
                    command_parts.extend([
                        "--shard_quantize_fp16_on_disk",
                        "true" if spec.shard_quantize_fp16_on_disk else "false",
                    ])
                    if spec.shard_quant_mode:
                        command_parts.extend(["--shard_quant_mode", shlex.quote(spec.shard_quant_mode)])
                    shard_offload_dir = spec.shard_offload_dir or posixpath.join(remote_run_root, "parameter_shards")
                    command_parts.extend(["--shard_offload_dir", shlex.quote(shard_offload_dir)])
            if remote_dataset_csv:
                command_parts.extend(["--dataset_csv", shlex.quote(remote_dataset_csv)])

            if args.foreground_clients:
                remote_cmd = (
                    "set -e; "
                    f"mkdir -p {shlex.quote(remote_run_root)}; "
                    f"cd {shlex.quote(spec.remote_root)}; "
                    + " ".join(command_parts)
                    + f" > {shlex.quote(remote_log_path)} 2>&1"
                )
                future = executor.submit(run_remote, client, remote_cmd)
                sessions.append(
                    {
                        "spec": spec,
                        "client": client,
                        "sftp": sftp,
                        "client_dir": client_dir,
                        "remote_run_root": remote_run_root,
                        "remote_log_path": remote_log_path,
                        "remote_metrics_path": remote_metrics_path,
                        "remote_pid_path": remote_pid_path,
                        "remote_dataset_csv": remote_dataset_csv,
                        "future": future,
                    }
                )
            else:
                remote_cmd = (
                    "set -e; "
                    f"mkdir -p {shlex.quote(remote_run_root)}; "
                    f"cd {shlex.quote(spec.remote_root)}; "
                    "nohup "
                    + " ".join(command_parts)
                    + f" > {shlex.quote(remote_log_path)} 2>&1 < /dev/null & "
                    + f"echo $! | tee {shlex.quote(remote_pid_path)}"
                )
                code, out, err = run_remote(client, remote_cmd)
                (client_dir / "launch_stdout.txt").write_text(out, encoding="utf-8")
                (client_dir / "launch_stderr.txt").write_text(err, encoding="utf-8")
                if code != 0:
                    raise RuntimeError(f"Failed to start remote client {spec.client_id} on {spec.host}")
                pid = out.strip().splitlines()[-1].strip()
                if not pid:
                    raise RuntimeError(f"Remote client {spec.client_id} did not return a PID")
                sessions.append(
                    {
                        "spec": spec,
                        "client": client,
                        "sftp": sftp,
                        "client_dir": client_dir,
                        "remote_run_root": remote_run_root,
                        "remote_log_path": remote_log_path,
                        "remote_metrics_path": remote_metrics_path,
                        "remote_pid_path": remote_pid_path,
                        "remote_dataset_csv": remote_dataset_csv,
                        "pid": pid,
                    }
                )

        if server_proc is not None:
            server_proc.wait(timeout=args.server_exit_timeout)

        if args.foreground_clients:
            for session in sessions:
                future = session["future"]
                assert isinstance(future, concurrent.futures.Future)
                try:
                    code, out, err = future.result(timeout=args.client_exit_timeout)
                except concurrent.futures.TimeoutError:
                    raise RuntimeError(f"Timed out waiting for foreground client {session['spec'].client_id}")
                (session["client_dir"] / "client_exec_stdout.txt").write_text(out, encoding="utf-8")
                (session["client_dir"] / "client_exec_stderr.txt").write_text(err, encoding="utf-8")
                if code != 0:
                    raise RuntimeError(f"Foreground client {session['spec'].client_id} exited with code {code}")
        else:
            for session in sessions:
                client = session["client"]
                pid = session["pid"]
                if not wait_remote_exit(client, pid, args.client_exit_timeout):
                    run_remote(client, f"kill {shlex.quote(pid)} >/dev/null 2>&1 || true")

        for session in sessions:
            sftp = session["sftp"]
            client_dir = session["client_dir"]
            try:
                fetch_remote_file(sftp, session["remote_log_path"], client_dir / "client.log")
                fetch_remote_file(sftp, session["remote_metrics_path"], client_dir / "client_metrics.csv")
                if not args.foreground_clients:
                    fetch_remote_file(sftp, session["remote_pid_path"], client_dir / "client.pid")
            finally:
                sftp.close()
    finally:
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
        for session in sessions:
            try:
                session["client"].close()
            except Exception:
                pass
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
        "clients": [
            {
                "client_id": session["spec"].client_id,
                "host": session["spec"].host,
                "backend": session["spec"].backend,
                "model_dir": session["spec"].model_dir,
                "remote_dataset_csv": session["remote_dataset_csv"],
                "remote_run_root": session["remote_run_root"],
            }
            for session in sessions
        ],
        "total_num_clients": total_num_clients,
    }
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"run_id={run_id}")
    print(f"run_dir={run_dir}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"[run_multi_nano_experiment] {exc}", file=sys.stderr)
        raise
