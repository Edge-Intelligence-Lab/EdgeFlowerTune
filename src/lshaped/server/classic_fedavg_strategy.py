from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from safetensors.numpy import save_file

from lshaped.common.logging_utils import append_csv, append_jsonl
from lshaped.config import AppConfig


def _to_int(value: Scalar | None, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value:
        return int(float(value))
    return default


def _to_float(value: Scalar | None, default: float = 0.0) -> float:
    if value is None:
        return default
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    if isinstance(value, str) and value:
        return float(value)
    return default


def _to_str(value: Scalar | None, default: str = "") -> str:
    if value is None:
        return default
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


class ClassicFedAvgLoraStrategy(FedAvg):
    def __init__(self, cfg: AppConfig, logger) -> None:
        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=cfg.flower.min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=cfg.flower.min_available_clients,
            accept_failures=cfg.federated.accept_failures,
            fit_metrics_aggregation_fn=self._aggregate_fit_metrics,
            initial_parameters=None,
            inplace=True,
        )
        self.cfg = cfg
        self.logger = logger
        self.metrics_csv = Path(cfg.runtime.output_dir) / "metrics.csv"
        self.metrics_jsonl = Path(cfg.runtime.output_dir) / "metrics.jsonl"
        self.checkpoint_dir = Path(cfg.runtime.output_dir) / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.parameter_names: list[str] | None = None

    def _wait_for_required_clients(self, client_manager: ClientManager) -> int:
        required = max(self.cfg.flower.min_available_clients, self.cfg.flower.min_fit_clients)
        available = client_manager.num_available()
        if available >= required:
            return available

        deadline = time.time() + max(1, int(self.cfg.flower.client_wait_timeout))
        self.logger.warning(
            "Waiting for more clients: available=%s required=%s timeout=%ss",
            available,
            required,
            self.cfg.flower.client_wait_timeout,
        )
        while time.time() < deadline:
            time.sleep(1.0)
            available = client_manager.num_available()
            if available >= required:
                self.logger.info("Required clients available: %s/%s", available, required)
                return available

        raise RuntimeError(
            f"Timed out waiting for required clients: available={available} required={required}"
        )

    def _fit_config(self, server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "batch_size": self.cfg.dataset.batch_size,
            "max_seq_len": self.cfg.dataset.max_seq_len,
            "local_steps": self.cfg.federated.local_steps,
            "local_epochs": self.cfg.federated.local_epochs,
            "prox_mu": float(self.cfg.federated.prox_mu),
            "learning_rate": float(self.cfg.model.learning_rate),
            "weight_decay": float(self.cfg.model.weight_decay),
        }

    def _aggregate_fit_metrics(self, metrics: list[tuple[int, dict[str, Scalar]]]) -> dict[str, Scalar]:
        if not metrics:
            return {}
        total_examples = sum(max(1, num_examples) for num_examples, _ in metrics)
        weighted_keys = {
            "loss",
            "objective_loss",
            "prox_term",
            "train_time_sec",
            "client_rss_mb",
            "transmitted_bytes",
        }
        aggregated: dict[str, Scalar] = {"total_examples": total_examples}
        for key in weighted_keys:
            weighted_sum = 0.0
            seen = False
            for num_examples, values in metrics:
                if key not in values:
                    continue
                weighted_sum += max(1, num_examples) * _to_float(values[key])
                seen = True
            if seen and total_examples > 0:
                aggregated[key] = weighted_sum / total_examples
        aggregated["num_clients"] = len(metrics)
        return aggregated

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        available = self._wait_for_required_clients(client_manager)
        requested = min(self.cfg.flower.sample_clients, available)
        clients = client_manager.sample(
            num_clients=max(self.cfg.flower.min_fit_clients, requested),
            min_num_clients=self.cfg.flower.min_fit_clients,
        )
        fit_ins = FitIns(parameters=parameters, config=self._fit_config(server_round))
        return [(client, fit_ins) for client in clients]

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, Any]]:
        return []

    def _record_client_metrics(self, server_round: int, results: list[tuple[ClientProxy, Any]]) -> None:
        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            row = {
                "round": server_round,
                "client_id": _to_str(metrics.get("client_id"), client_proxy.cid),
                "backend": _to_str(metrics.get("backend"), self.cfg.client.backend),
                "num_examples": int(fit_res.num_examples),
                "loss": _to_float(metrics.get("loss")),
                "steps_completed": _to_int(metrics.get("local_steps")),
                "epochs_completed": _to_int(metrics.get("local_epochs")),
                "train_time_sec": _to_float(metrics.get("train_time_sec")),
                "client_rss_mb": _to_float(metrics.get("client_rss_mb"), -1.0),
                "transmitted_bytes": _to_int(metrics.get("transmitted_bytes")),
                "objective_loss": _to_float(metrics.get("objective_loss")),
                "prox_term": _to_float(metrics.get("prox_term")),
                "parameter_count": _to_int(metrics.get("parameter_count")),
                "target_mode": _to_str(metrics.get("target_mode")),
                "lora_r": _to_int(metrics.get("lora_r")),
                "lora_alpha": _to_float(metrics.get("lora_alpha")),
                "lora_dropout": _to_float(metrics.get("lora_dropout")),
                "prox_mu": _to_float(metrics.get("prox_mu")),
            }
            append_csv(self.metrics_csv, row)
            append_jsonl(self.metrics_jsonl, row)

            parameter_names = _to_str(metrics.get("parameter_names"))
            if parameter_names and self.parameter_names is None:
                self.parameter_names = [name for name in parameter_names.split(",") if name]

    def _save_checkpoint(
        self,
        server_round: int,
        parameters: Parameters,
        aggregated_metrics: dict[str, Scalar],
    ) -> None:
        arrays = parameters_to_ndarrays(parameters)
        if not arrays:
            return

        if self.parameter_names is None or len(self.parameter_names) != len(arrays):
            self.parameter_names = [f"adapter_{index:04d}" for index in range(len(arrays))]

        tensor_map = {
            name: np.asarray(array, dtype=np.float32)
            for name, array in zip(self.parameter_names, arrays, strict=True)
        }
        metadata = {
            "round": str(server_round),
            "algorithm": self.cfg.federated.algorithm,
            "lora_r": str(self.cfg.model.lora_r),
            "lora_alpha": str(self.cfg.model.lora_alpha),
            "target_modules": ",".join(self.cfg.model.lora_target_modules),
            "num_clients": str(aggregated_metrics.get("num_clients", self.cfg.flower.min_fit_clients)),
        }
        checkpoint_path = self.checkpoint_dir / f"round_{server_round:06d}.safetensors"
        save_file(tensor_map, str(checkpoint_path), metadata=metadata)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if failures:
            self.logger.warning("Round %s failures: %s", server_round, len(failures))

        self._record_client_metrics(server_round, results)
        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        aggregated_metrics = aggregated_metrics or {}

        if aggregated_parameters is not None:
            self._save_checkpoint(server_round, aggregated_parameters, aggregated_metrics)

        self.logger.info(
            "round=%s clients=%s total_examples=%s mean_loss=%.6f mean_objective_loss=%.6f mean_prox_term=%.6f mean_train_time_sec=%.3f bytes=%.0f",
            server_round,
            aggregated_metrics.get("num_clients", len(results)),
            aggregated_metrics.get("total_examples", 0),
            _to_float(aggregated_metrics.get("loss")),
            _to_float(aggregated_metrics.get("objective_loss")),
            _to_float(aggregated_metrics.get("prox_term")),
            _to_float(aggregated_metrics.get("train_time_sec")),
            _to_float(aggregated_metrics.get("transmitted_bytes")),
        )
        return aggregated_parameters, aggregated_metrics
