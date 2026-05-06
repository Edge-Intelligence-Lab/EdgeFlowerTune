from __future__ import annotations

import argparse
import csv
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import flwr as fl
import numpy as np
import yaml
from flwr.common import FitIns, FitRes, GetParametersIns, Parameters, Scalar
from flwr.server import ServerConfig, start_server
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg

from lshaped.classic_fl.adapter_utils import (
    adapter_bytes_to_parameters,
    adapter_bytes_to_state,
    adapter_state_to_bytes,
    aggregate_adapter_states,
    parameters_to_adapter_bytes,
)


@dataclass
class RuntimeConfig:
    run_name: str
    output_dir: str
    seed: int = 7


@dataclass
class FlowerConfig:
    server_address: str
    grpc_max_message_length: int = 536870912
    num_rounds: int = 3
    min_available_clients: int = 5
    min_fit_clients: int = 5
    sample_clients: int = 5
    round_timeout: int = 1800
    client_wait_timeout: int = 300


@dataclass
class FederatedConfig:
    aggregate_by_num_examples: bool = True
    local_steps: int = 1
    local_epochs: int = 1


@dataclass
class ModelConfig:
    initial_lora_path: str = ""
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_mode: str = "attn"


@dataclass
class TrainConfig:
    seq_len: int = 128
    batch_size: int = 1
    learning_rate: float = 2e-4
    weight_decay: float = 0.0
    logging_steps: int = 1
    max_steps_per_round: int = 1
    shuffle: bool = False


@dataclass
class LoggingConfig:
    save_every_rounds: int = 1


@dataclass
class ClassicFedAvgLoraConfig:
    runtime: RuntimeConfig
    flower: FlowerConfig
    federated: FederatedConfig
    model: ModelConfig
    train: TrainConfig
    logging: LoggingConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run classic FedAvg + LoRA Flower server")
    parser.add_argument("--config", required=True, help="Path to YAML config")
    return parser.parse_args()


def load_config(path: str | Path) -> ClassicFedAvgLoraConfig:
    raw = yaml.safe_load(Path(path).read_text())
    return ClassicFedAvgLoraConfig(
        runtime=RuntimeConfig(**raw["runtime"]),
        flower=FlowerConfig(**raw["flower"]),
        federated=FederatedConfig(**raw.get("federated", {})),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
        logging=LoggingConfig(**raw.get("logging", {})),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


class ClassicFedAvgLoraStrategy(FedAvg):
    def __init__(self, cfg: ClassicFedAvgLoraConfig) -> None:
        self.cfg = cfg
        self.output_dir = Path(cfg.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_csv = self.output_dir / "metrics.csv"
        self.current_adapter_bytes = (
            Path(cfg.model.initial_lora_path).read_bytes()
            if cfg.model.initial_lora_path
            else b""
        )
        self.metadata = {
            "rank": str(cfg.model.lora_r),
            "alpha": str(cfg.model.lora_alpha),
            "dropout": str(cfg.model.lora_dropout),
            "target_mode": cfg.model.target_mode,
        }
        super().__init__(
            min_available_clients=cfg.flower.min_available_clients,
            min_fit_clients=cfg.flower.min_fit_clients,
            fraction_fit=1.0,
            fraction_evaluate=0.0,
        )

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        if self.current_adapter_bytes:
            self._save_checkpoint(0, self.current_adapter_bytes)
            return adapter_bytes_to_parameters(self.current_adapter_bytes)

        available = self._wait_for_required_clients(client_manager)
        clients = client_manager.sample(num_clients=1, min_num_clients=1)
        if not clients:
            raise RuntimeError(f"No clients available after wait (available={available})")
        get_res = clients[0].get_parameters(GetParametersIns(config={}), timeout=self.cfg.flower.round_timeout, group_id=0)
        self.current_adapter_bytes = parameters_to_adapter_bytes(get_res.parameters)
        self._save_checkpoint(0, self.current_adapter_bytes)
        print(f"[classic-fl] initialized global adapter from client {clients[0].cid}")
        return get_res.parameters

    def _wait_for_required_clients(self, client_manager: ClientManager) -> int:
        required = max(self.cfg.flower.min_available_clients, self.cfg.flower.min_fit_clients)
        available = client_manager.num_available()
        if available >= required:
            return available

        deadline = time.time() + max(1, int(self.cfg.flower.client_wait_timeout))
        print(
            f"[classic-fl] waiting for clients available={available} required={required} "
            f"timeout={self.cfg.flower.client_wait_timeout}s"
        )
        while time.time() < deadline:
            time.sleep(1.0)
            available = client_manager.num_available()
            if available >= required:
                print(f"[classic-fl] required clients available: {available}/{required}")
                return available
        raise RuntimeError(
            f"Timed out waiting for required clients: available={available} required={required}"
        )

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        available = self._wait_for_required_clients(client_manager)
        sample_size = min(self.cfg.flower.sample_clients, available)
        clients = client_manager.sample(
            num_clients=max(self.cfg.flower.min_fit_clients, sample_size),
            min_num_clients=self.cfg.flower.min_fit_clients,
        )
        fit_config: dict[str, Scalar] = {
            "server_round": server_round,
            "mode": "train",
            "seq_len": self.cfg.train.seq_len,
            "batch_size": self.cfg.train.batch_size,
            "learning_rate": self.cfg.train.learning_rate,
            "weight_decay": self.cfg.train.weight_decay,
            "logging_steps": self.cfg.train.logging_steps,
            "max_steps": self.cfg.train.max_steps_per_round,
            "local_steps": self.cfg.federated.local_steps,
            "local_epochs": self.cfg.federated.local_epochs,
            "shuffle": int(self.cfg.train.shuffle),
            "lora_r": self.cfg.model.lora_r,
            "lora_alpha": self.cfg.model.lora_alpha,
            "lora_dropout": self.cfg.model.lora_dropout,
            "target_mode": self.cfg.model.target_mode,
        }
        fit_ins = FitIns(parameters=parameters, config=fit_config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, FitRes]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if failures:
            print(f"[classic-fl] round={server_round} failures={len(failures)}")
        if not results:
            return adapter_bytes_to_parameters(self.current_adapter_bytes), {}

        states: list[dict[str, np.ndarray]] = []
        weights: list[float] = []
        rows: list[dict[str, Any]] = []

        for client_proxy, fit_res in results:
            adapter_bytes = parameters_to_adapter_bytes(fit_res.parameters)
            state = adapter_bytes_to_state(adapter_bytes)
            weight = float(fit_res.num_examples if self.cfg.federated.aggregate_by_num_examples else 1.0)
            states.append(state)
            weights.append(weight)
            rows.append(
                {
                    "round": server_round,
                    "client_id": str(fit_res.metrics.get("client_id", client_proxy.cid)),
                    "num_examples": fit_res.num_examples,
                    "weight": weight,
                    "round_time_sec": float(
                        fit_res.metrics.get(
                            "client_round_time_sec",
                            fit_res.metrics.get("round_time_sec", 0.0),
                        )
                    ),
                    "train_loss": float(fit_res.metrics.get("train_loss", 0.0)),
                    "transmitted_bytes": int(fit_res.metrics.get("transmitted_bytes", 0)),
                    "transport": str(fit_res.metrics.get("transport", "")),
                }
            )

        aggregated = aggregate_adapter_states(states, weights)
        self.current_adapter_bytes = adapter_state_to_bytes(aggregated, metadata=self.metadata)
        self._append_metrics(rows)
        self._save_checkpoint(server_round, self.current_adapter_bytes)

        mean_loss = float(sum(row["train_loss"] for row in rows) / max(1, len(rows)))
        total_bytes = int(sum(row["transmitted_bytes"] for row in rows))
        metrics: dict[str, Scalar] = {
            "mean_train_loss": mean_loss,
            "aggregated_clients": len(rows),
            "total_transmitted_bytes": total_bytes,
        }
        print(
            f"[classic-fl] round={server_round} aggregated_clients={len(rows)} "
            f"mean_train_loss={mean_loss:.4f} total_bytes={total_bytes}"
        )
        return adapter_bytes_to_parameters(self.current_adapter_bytes), metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, fl.common.EvaluateIns]]:
        return []

    def aggregate_evaluate(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: list[Any],
    ) -> tuple[float | None, dict[str, Scalar]]:
        return None, {}

    def evaluate(self, server_round: int, parameters: Parameters) -> tuple[float, dict[str, Scalar]] | None:
        return None

    def _append_metrics(self, rows: list[dict[str, Any]]) -> None:
        file_exists = self.metrics_csv.exists()
        with self.metrics_csv.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "round",
                    "client_id",
                    "num_examples",
                    "weight",
                    "round_time_sec",
                    "train_loss",
                    "transmitted_bytes",
                    "transport",
                ],
            )
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

    def _save_checkpoint(self, server_round: int, data: bytes) -> None:
        if self.cfg.logging.save_every_rounds <= 0:
            return
        if server_round % self.cfg.logging.save_every_rounds != 0:
            return
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (ckpt_dir / f"round_{server_round:06d}.safetensors").write_bytes(data)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed)
    strategy = ClassicFedAvgLoraStrategy(cfg)
    print(f"[classic-fl] starting server on {cfg.flower.server_address}")
    start_server(
        server_address=cfg.flower.server_address,
        config=ServerConfig(num_rounds=cfg.flower.num_rounds, round_timeout=cfg.flower.round_timeout),
        strategy=strategy,
        grpc_max_message_length=cfg.flower.grpc_max_message_length,
    )


if __name__ == "__main__":
    main()
