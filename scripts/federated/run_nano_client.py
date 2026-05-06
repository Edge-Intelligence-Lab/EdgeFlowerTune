from __future__ import annotations

import argparse
import os
import shlex

import paramiko


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the L-shaped C++ client on a Jetson Nano over SSH")
    parser.add_argument("--host", required=True, help="Nano host or IP")
    parser.add_argument("--username", default="jetson")
    parser.add_argument("--password", default=None, help="SSH password; defaults to env var if omitted")
    parser.add_argument("--password-env", default="NANO_PASSWORD")
    parser.add_argument("--remote-root", default=None, help="Remote repo root, default: /home/<username>/L-shaped")
    parser.add_argument("--server-address", default="10.200.14.82:19080")
    parser.add_argument("--client-id", required=True)
    parser.add_argument("--backend", choices=("mock", "mft"), default="mock")
    parser.add_argument("--model-dir", default="", help="Required when backend=mft")
    parser.add_argument("--dataset-csv", default="", help="Remote Nano dataset CSV path")
    parser.add_argument("--client-index", type=int, default=0)
    parser.add_argument("--num-clients", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=-1)
    parser.add_argument("--answer-prefix", default=" ")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--local-steps", type=int, default=1)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--logging-steps", type=int, default=1)
    parser.add_argument("--target-mode", default="attn")
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=float, default=16.0)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--synthetic-samples", type=int, default=32)
    parser.add_argument("--mock-hidden-size", type=int, default=128)
    parser.add_argument("--metrics-path", default=None)
    parser.add_argument("--detach", action="store_true", help="Run client under nohup and return immediately")
    return parser.parse_args()


def resolve_password(args: argparse.Namespace) -> str:
    password = args.password or os.environ.get(args.password_env)
    if not password:
        raise RuntimeError(
            f"Missing password: pass --password or set env var {args.password_env}"
        )
    return password


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


def main() -> None:
    args = parse_args()
    password = resolve_password(args)
    remote_root = args.remote_root or f"/home/{args.username}/L-shaped"
    binary = (
        f"{remote_root}/build/cpp_client_mft/lshaped_flower_client"
        if args.backend == "mft"
        else f"{remote_root}/build/cpp_client_mock/lshaped_flower_client"
    )
    metrics_path = args.metrics_path or f"{remote_root}/outputs/{args.client_id}_metrics.csv"
    if args.backend == "mft" and not args.model_dir:
        raise RuntimeError("--model-dir is required when --backend=mft")

    command_parts = [
        shlex.quote(binary),
        "--server_address", shlex.quote(args.server_address),
        "--client_id", shlex.quote(args.client_id),
        "--backend", shlex.quote(args.backend),
        "--client_index", str(args.client_index),
        "--num_clients", str(args.num_clients),
        "--batch_size", str(args.batch_size),
        "--max_seq_len", str(args.max_seq_len),
        "--max_rounds", str(args.max_rounds),
        "--local_steps", str(args.local_steps),
        "--local_epochs", str(args.local_epochs),
        "--grad_accum_steps", str(args.grad_accum_steps),
        "--learning_rate", str(args.learning_rate),
        "--weight_decay", str(args.weight_decay),
        "--logging_steps", str(args.logging_steps),
        "--target_mode", shlex.quote(args.target_mode),
        "--lora_r", str(args.lora_r),
        "--lora_alpha", str(args.lora_alpha),
        "--lora_dropout", str(args.lora_dropout),
        "--answer_prefix", shlex.quote(args.answer_prefix),
        "--synthetic_samples", str(args.synthetic_samples),
        "--mock_hidden_size", str(args.mock_hidden_size),
        "--metrics_path", shlex.quote(metrics_path),
    ]
    if args.backend == "mft":
        command_parts.extend(["--model_dir", shlex.quote(args.model_dir)])
    if args.dataset_csv:
        command_parts.extend(["--dataset_csv", shlex.quote(args.dataset_csv)])
    remote_cmd = (
        f"cd {shlex.quote(remote_root)} && mkdir -p outputs && " + " ".join(command_parts)
    )
    if args.detach:
        remote_cmd = f"nohup bash -lc {shlex.quote(remote_cmd)} > /tmp/{args.client_id}.log 2>&1 < /dev/null &"

    client = connect(args.host, args.username, password)
    try:
        stdin, stdout, stderr = client.exec_command(remote_cmd, get_pty=not args.detach)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        if out:
            print(out, end="" if out.endswith("\n") else "\n")
        if err:
            print(err, end="" if err.endswith("\n") else "\n")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            raise SystemExit(exit_status)
    finally:
        client.close()


if __name__ == "__main__":
    main()
