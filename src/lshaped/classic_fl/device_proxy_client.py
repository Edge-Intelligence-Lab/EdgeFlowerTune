from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import paramiko

from lshaped.classic_fl.adapter_utils import (
    adapter_bytes_to_parameters,
    parameters_to_adapter_bytes,
    read_file_bytes,
    write_file_bytes,
)


@dataclass
class DeviceSpec:
    client_id: str
    transport: str
    local_run_dir: str
    initial_adapter_path: str
    num_examples: int
    train_timeout_sec: int
    remote_train_binary: str
    remote_model_dir: str
    remote_train_jsonl: str
    remote_valid_jsonl: str
    remote_work_root: str
    adb_path: str = "adb"
    serial: str = ""
    host: str = ""
    username: str = ""
    password: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one classic edge FL proxy client")
    parser.add_argument("--server-address", required=True)
    parser.add_argument("--spec-json", required=True)
    parser.add_argument("--grpc-max-message-length", type=int, default=536870912)
    return parser.parse_args()


def load_spec(path: str | Path) -> DeviceSpec:
    raw = json.loads(Path(path).read_text())
    return DeviceSpec(**raw)


def _parse_last_loss(log_text: str) -> float:
    loss = 0.0
    for line in log_text.splitlines():
        parsed = None
        if "Loss=" in line:
            try:
                parsed = float(line.split("Loss=", 1)[1].split()[0].rstrip(","))
            except Exception:
                parsed = None
        elif " loss " in line:
            try:
                parsed = float(line.split(" loss ", 1)[1].split()[0].rstrip(","))
            except Exception:
                parsed = None
        elif "loss=" in line:
            try:
                parsed = float(line.split("loss=", 1)[1].split()[0].rstrip(","))
            except Exception:
                parsed = None
        if parsed is not None:
            loss = parsed
    return loss


class DeviceRunner:
    def __init__(self, spec: DeviceSpec) -> None:
        self.spec = spec

    def push(self, local_path: Path, remote_path: str) -> None:
        raise NotImplementedError

    def pull(self, remote_path: str, local_path: Path) -> None:
        raise NotImplementedError

    def run(self, command: str, timeout_sec: int) -> tuple[int, str, str]:
        raise NotImplementedError


class AdbRunner(DeviceRunner):
    def _base(self) -> list[str]:
        return [self.spec.adb_path, "-s", self.spec.serial]

    def push(self, local_path: Path, remote_path: str) -> None:
        subprocess.run(self._base() + ["push", str(local_path), remote_path], check=True, capture_output=True, text=True)

    def pull(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(self._base() + ["pull", remote_path, str(local_path)], check=True, capture_output=True, text=True)

    def run(self, command: str, timeout_sec: int) -> tuple[int, str, str]:
        proc = subprocess.run(
            self._base() + ["shell", command],
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        return proc.returncode, proc.stdout, proc.stderr


class SshRunner(DeviceRunner):
    def __init__(self, spec: DeviceSpec) -> None:
        super().__init__(spec)
        self.client = paramiko.SSHClient()
        self.client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self.client.connect(
            hostname=spec.host,
            username=spec.username,
            password=spec.password,
            timeout=20,
            look_for_keys=False,
            allow_agent=False,
        )
        self.sftp = self.client.open_sftp()

    def close(self) -> None:
        self.sftp.close()
        self.client.close()

    def push(self, local_path: Path, remote_path: str) -> None:
        self._ensure_remote_parent(remote_path)
        self.sftp.put(str(local_path), remote_path)

    def pull(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        self.sftp.get(remote_path, str(local_path))

    def run(self, command: str, timeout_sec: int) -> tuple[int, str, str]:
        stdin, stdout, stderr = self.client.exec_command(command, get_pty=True, timeout=timeout_sec)
        out = stdout.read().decode("utf-8", "replace")
        err = stderr.read().decode("utf-8", "replace")
        code = stdout.channel.recv_exit_status()
        return code, out, err

    def _ensure_remote_parent(self, remote_path: str) -> None:
        parent = str(Path(remote_path).parent).replace("\\", "/")
        parts = [part for part in parent.strip("/").split("/") if part]
        current = ""
        for part in parts:
            current = f"{current}/{part}" if current else f"/{part}"
            try:
                self.sftp.stat(current)
            except FileNotFoundError:
                self.sftp.mkdir(current)


class DeviceProxyNumPyClient(fl.client.NumPyClient):
    def __init__(self, spec: DeviceSpec) -> None:
        self.spec = spec
        self.local_run_dir = Path(spec.local_run_dir)
        self.local_run_dir.mkdir(parents=True, exist_ok=True)
        self.initial_adapter = read_file_bytes(spec.initial_adapter_path)
        self.runner: DeviceRunner
        if spec.transport == "adb":
            self.runner = AdbRunner(spec)
        elif spec.transport == "ssh":
            self.runner = SshRunner(spec)
        else:
            raise ValueError(f"Unsupported transport: {spec.transport}")

    def get_parameters(self, config: dict[str, Any]):  # noqa: ANN001
        return [np.frombuffer(self.initial_adapter, dtype=np.uint8).copy()]

    def fit(self, parameters, config):  # noqa: ANN001
        round_id = int(config.get("server_round", -1))
        incoming = fl.common.ndarrays_to_parameters(parameters)
        incoming_bytes = parameters_to_adapter_bytes(incoming)

        local_round_dir = self.local_run_dir / f"round_{round_id:06d}"
        local_round_dir.mkdir(parents=True, exist_ok=True)
        local_global = local_round_dir / "global.safetensors"
        local_updated = local_round_dir / "updated.safetensors"
        local_log = local_round_dir / "train.log"
        write_file_bytes(local_global, incoming_bytes)

        remote_round_root = f"{self.spec.remote_work_root}/round_{round_id:06d}"
        remote_global = f"{remote_round_root}/global.safetensors"
        remote_output_dir = f"{remote_round_root}/output"
        remote_updated = f"{remote_output_dir}/gemma_lora.safetensors"
        remote_log = f"{remote_round_root}/train.log"

        self.runner.push(local_global, remote_global)
        command = self._build_remote_command(config, remote_round_root, remote_global, remote_output_dir, remote_log, remote_updated)

        start = time.perf_counter()
        code, stdout_text, stderr_text = self.runner.run(command, timeout_sec=self.spec.train_timeout_sec)
        round_time = time.perf_counter() - start
        (local_round_dir / "remote_stdout.log").write_text(stdout_text, encoding="utf-8")
        (local_round_dir / "remote_stderr.log").write_text(stderr_text, encoding="utf-8")
        if code != 0:
            raise RuntimeError(
                f"Device training failed for {self.spec.client_id} round={round_id} code={code}\n"
                f"stdout:\n{stdout_text}\n\nstderr:\n{stderr_text}"
            )

        self.runner.pull(remote_updated, local_updated)
        try:
            self.runner.pull(remote_log, local_log)
            log_text = local_log.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            log_text = stdout_text

        updated_bytes = read_file_bytes(local_updated)
        metrics = {
            "client_id": self.spec.client_id,
            "transport": self.spec.transport,
            "round_time_sec": round_time,
            "train_loss": _parse_last_loss(log_text),
            "transmitted_bytes": len(incoming_bytes) + len(updated_bytes),
        }
        return [np.frombuffer(updated_bytes, dtype=np.uint8).copy()], self.spec.num_examples, metrics

    def evaluate(self, parameters, config):  # noqa: ANN001
        return 0.0, 0, {}

    def close(self) -> None:
        if isinstance(self.runner, SshRunner):
            self.runner.close()

    def _build_remote_command(
        self,
        config: dict[str, Any],
        remote_round_root: str,
        remote_global: str,
        remote_output_dir: str,
        remote_log: str,
        remote_updated: str,
    ) -> str:
        args = [
            shlex.quote(self.spec.remote_train_binary),
            "--model_dir", shlex.quote(self.spec.remote_model_dir),
            "--jsonl_train", shlex.quote(self.spec.remote_train_jsonl),
            "--output_dir", shlex.quote(remote_output_dir),
            "--seq_len", str(int(config.get("seq_len", 128))),
            "--batch", str(int(config.get("batch_size", 1))),
            "--grad_accum", str(int(config.get("grad_accum", 1))),
            "--lr", str(float(config.get("learning_rate", 2e-4))),
            "--weight_decay", str(float(config.get("weight_decay", 0.0))),
            "--targets", shlex.quote(str(config.get("target_mode", "full"))),
            "--lora_r", str(int(config.get("lora_r", 8))),
            "--lora_alpha", str(float(config.get("lora_alpha", 32.0))),
            "--lora_dropout", str(float(config.get("lora_dropout", 0.0))),
            "--logging_steps", str(int(config.get("logging_steps", 1))),
            "--save_every", "0",
            "--max_steps", str(int(config.get("max_steps", 1))),
            "--init_lora", shlex.quote(remote_global),
            "--export_lora", shlex.quote(remote_updated),
        ]
        if self.spec.remote_valid_jsonl:
            args.extend(["--jsonl_valid", shlex.quote(self.spec.remote_valid_jsonl)])
        args.append("--shuffle" if int(config.get("shuffle", 0)) else "--no_shuffle")

        if self.spec.transport == "adb":
            bin_dir = str(Path(self.spec.remote_train_binary).parent).replace("\\", "/")
            setup = (
                f"mkdir -p {shlex.quote(remote_round_root)} {shlex.quote(remote_output_dir)} {shlex.quote(remote_round_root + '/tmp')}; "
                f"cd {shlex.quote(remote_round_root)}; "
                f"export TMPDIR={shlex.quote(remote_round_root + '/tmp')}; "
                f"export LD_LIBRARY_PATH={shlex.quote(bin_dir)}; "
                + " ".join(args)
                + f" > {shlex.quote(remote_log)} 2>&1"
            )
            return f"sh -lc {shlex.quote(setup)}"

        setup = (
            f"mkdir -p {shlex.quote(remote_round_root)} {shlex.quote(remote_output_dir)}; "
            + " ".join(args)
            + f" > {shlex.quote(remote_log)} 2>&1"
        )
        return f"bash -lc {shlex.quote(setup)}"


def run_client(spec: DeviceSpec, server_address: str, grpc_max_message_length: int) -> None:
    client = DeviceProxyNumPyClient(spec)
    try:
        fl.client.start_numpy_client(
            server_address=server_address,
            client=client,
            grpc_max_message_length=grpc_max_message_length,
        )
    finally:
        client.close()
