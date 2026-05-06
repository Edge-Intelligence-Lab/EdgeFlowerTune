from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import FitIns, GetParametersIns, GetPropertiesIns, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from safetensors.numpy import save_file

from lshaped.config import AppConfig
from lshaped.server.classic_fedavg_strategy import (
    ClassicFedAvgLoraStrategy,
    _to_float,
    _to_str,
)


def _empty_parameters() -> Parameters:
    return ndarrays_to_parameters([])


def _parse_parameter_names(raw_value: Scalar | None) -> list[str]:
    value = _to_str(raw_value)
    if not value:
        return []
    return [name for name in value.split(",") if name]


def _base_name(name: str) -> tuple[str, str]:
    if name.endswith(".lora_A"):
        return name[:-7], "A"
    if name.endswith(".lora_B"):
        return name[:-7], "B"
    raise ValueError(f"Unsupported FlexLoRA tensor name: {name}")


def _compose_dense_delta(array_a: np.ndarray, array_b: np.ndarray, alpha: float) -> np.ndarray:
    a = np.asarray(array_a, dtype=np.float32)
    b = np.asarray(array_b, dtype=np.float32)
    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("FlexLoRA expects 2D LoRA tensors")

    if a.shape[0] == b.shape[1]:
        rank = a.shape[0]
        scale = alpha / float(max(1, rank))
        return np.asarray((a.T @ b.T) * scale, dtype=np.float32)

    if a.shape[1] == b.shape[0]:
        rank = a.shape[1]
        scale = alpha / float(max(1, rank))
        return np.asarray((a @ b) * scale, dtype=np.float32)

    raise ValueError(f"Incompatible LoRA tensor shapes for FlexLoRA: A={a.shape}, B={b.shape}")


def _factorize_dense_delta(delta: np.ndarray, rank: int, alpha: float) -> tuple[np.ndarray, np.ndarray]:
    dense = np.asarray(delta, dtype=np.float32)
    if dense.ndim != 2:
        raise ValueError(f"FlexLoRA dense delta must be 2D, got {dense.ndim}D")

    in_dim, out_dim = dense.shape
    rank = max(1, int(rank))
    a = np.zeros((rank, in_dim), dtype=np.float32)
    b = np.zeros((out_dim, rank), dtype=np.float32)

    if not np.any(dense):
        return a, b

    u, singular_values, vh = np.linalg.svd(dense, full_matrices=False)
    keep = min(rank, singular_values.shape[0])
    if keep <= 0:
        return a, b

    sqrt_s = np.sqrt(singular_values[:keep]).astype(np.float32)
    scale = float(alpha) / float(rank)
    if scale == 0.0:
        raise ValueError("FlexLoRA requires non-zero alpha")

    a[:keep, :] = (u[:, :keep] * sqrt_s[np.newaxis, :]).T.astype(np.float32)
    b[:, :keep] = (vh[:keep, :].T * ((sqrt_s / scale)[np.newaxis, :])).astype(np.float32)
    return a, b


class ClassicFlexLoraStrategy(ClassicFedAvgLoraStrategy):
    def __init__(self, cfg: AppConfig, logger) -> None:
        super().__init__(cfg=cfg, logger=logger)
        self.client_rank_by_id = {
            client_id: int(rank) for client_id, rank in cfg.federated.client_lora_ranks.items()
        }
        self.client_id_by_cid: dict[str, str] = {}
        self.base_order: list[str] = []
        self.global_dense_by_base: dict[str, np.ndarray] = {}
        self.max_rank = max(self.client_rank_by_id.values())

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        del client_manager
        return _empty_parameters()

    def _client_id_for_proxy(self, client: ClientProxy, server_round: int) -> str:
        cached = self.client_id_by_cid.get(client.cid)
        if cached:
            return cached
        try:
            props = client.get_properties(
                GetPropertiesIns(config={}),
                timeout=self.cfg.flower.round_timeout,
                group_id=server_round,
            )
            client_id = _to_str(props.properties.get("client_id"), "")
        except Exception as exc:  # pragma: no cover - best effort for mixed clients
            self.logger.warning("FlexLoRA failed to query client properties for cid=%s: %s", client.cid, exc)
            client_id = ""
        if client_id:
            self.client_id_by_cid[client.cid] = client_id
        return client_id

    def _initial_parameters_from_client(self, client: ClientProxy, server_round: int) -> Parameters:
        try:
            return client.get_parameters(
                GetParametersIns(config={}),
                timeout=self.cfg.flower.round_timeout,
                group_id=server_round,
            ).parameters
        except Exception as exc:  # pragma: no cover - best effort for mixed clients
            self.logger.warning("FlexLoRA failed to query initial parameters for cid=%s: %s", client.cid, exc)
            return _empty_parameters()

    def _fit_config(self, server_round: int) -> dict[str, Scalar]:
        return {
            "server_round": server_round,
            "batch_size": self.cfg.dataset.batch_size,
            "max_seq_len": self.cfg.dataset.max_seq_len,
            "local_steps": self.cfg.federated.local_steps,
            "local_epochs": self.cfg.federated.local_epochs,
            "prox_mu": 0.0,
            "learning_rate": float(self.cfg.model.learning_rate),
            "weight_decay": float(self.cfg.model.weight_decay),
        }

    def _dense_from_client_parameters(
        self,
        parameters: Parameters,
        parameter_names: list[str],
        alpha: float,
    ) -> dict[str, np.ndarray]:
        arrays = parameters_to_ndarrays(parameters)
        if len(arrays) != len(parameter_names):
            raise ValueError(
                f"FlexLoRA parameter count mismatch: arrays={len(arrays)} names={len(parameter_names)}"
            )

        grouped: dict[str, dict[str, np.ndarray]] = {}
        for name, array in zip(parameter_names, arrays, strict=True):
            base, part = _base_name(name)
            grouped.setdefault(base, {})[part] = np.asarray(array, dtype=np.float32)

        dense_by_base: dict[str, np.ndarray] = {}
        for base in self.base_order:
            pair = grouped.get(base)
            if not pair or "A" not in pair or "B" not in pair:
                raise ValueError(f"FlexLoRA missing A/B pair for base {base}")
            dense_by_base[base] = _compose_dense_delta(pair["A"], pair["B"], alpha)
        return dense_by_base

    def _parameters_for_rank(self, rank: int) -> Parameters:
        if not self.global_dense_by_base:
            return _empty_parameters()

        arrays: list[np.ndarray] = []
        parameter_names: list[str] = []
        for base in self.base_order:
            dense = self.global_dense_by_base[base]
            a, b = _factorize_dense_delta(dense, rank=rank, alpha=self.cfg.model.lora_alpha)
            arrays.extend([a, b])
            parameter_names.extend([base + ".lora_A", base + ".lora_B"])
        self.parameter_names = parameter_names
        return ndarrays_to_parameters(arrays)

    def _parameters_for_client(self, client_id: str) -> Parameters:
        rank = self.client_rank_by_id.get(client_id)
        if rank is None:
            self.logger.warning("FlexLoRA missing rank config for client_id=%s, sending empty adapter", client_id)
            return _empty_parameters()
        return self._parameters_for_rank(rank)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> list[tuple[ClientProxy, FitIns]]:
        del parameters
        available = self._wait_for_required_clients(client_manager)
        requested = min(self.cfg.flower.sample_clients, available)
        clients = client_manager.sample(
            num_clients=max(self.cfg.flower.min_fit_clients, requested),
            min_num_clients=self.cfg.flower.min_fit_clients,
        )

        fit_config = self._fit_config(server_round)
        scheduled: list[tuple[ClientProxy, FitIns]] = []
        for client in clients:
            client_id = self._client_id_for_proxy(client, int(server_round))
            if self.global_dense_by_base and client_id:
                personalized_parameters = self._parameters_for_client(client_id)
            else:
                personalized_parameters = self._initial_parameters_from_client(client, int(server_round))
            scheduled.append((client, FitIns(parameters=personalized_parameters, config=fit_config)))
        return scheduled

    def _save_checkpoint(
        self,
        server_round: int,
        parameters: Parameters,
        aggregated_metrics: dict[str, Scalar],
    ) -> None:
        del parameters
        if not self.global_dense_by_base:
            return

        arrays = parameters_to_ndarrays(self._parameters_for_rank(self.max_rank))
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
            "server_rank": str(self.max_rank),
            "lora_alpha": str(self.cfg.model.lora_alpha),
            "target_modules": ",".join(self.cfg.model.lora_target_modules),
            "num_clients": str(aggregated_metrics.get("num_clients", self.cfg.flower.min_fit_clients)),
            "client_ranks": ",".join(
                f"{client_id}:{self.client_rank_by_id[client_id]}" for client_id in self.cfg.dataset.client_ids
            ),
        }
        checkpoint_path = Path(self.cfg.runtime.output_dir) / "checkpoints" / f"round_{server_round:06d}.safetensors"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        save_file(tensor_map, str(checkpoint_path), metadata=metadata)

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if failures:
            self.logger.warning("Round %s failures: %s", server_round, len(failures))
        if not results:
            return _empty_parameters(), {}

        self._record_client_metrics(server_round, results)

        weighted_dense: dict[str, np.ndarray] = {}
        total_weight = 0.0
        fit_metrics: list[tuple[int, dict[str, Scalar]]] = []

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            fit_metrics.append((int(fit_res.num_examples), metrics))

            client_id = _to_str(metrics.get("client_id"), client_proxy.cid)
            self.client_id_by_cid[client_proxy.cid] = client_id

            parameter_names = _parse_parameter_names(metrics.get("parameter_names"))
            if not parameter_names:
                raise ValueError(f"FlexLoRA requires parameter_names metrics from client {client_id}")

            if not self.base_order:
                self.base_order = [
                    name[:-7] for name in parameter_names if name.endswith(".lora_A")
                ]
                self.parameter_names = [
                    name
                    for base in self.base_order
                    for name in (base + ".lora_A", base + ".lora_B")
                ]

            alpha = _to_float(metrics.get("lora_alpha"), self.cfg.model.lora_alpha)
            client_dense = self._dense_from_client_parameters(fit_res.parameters, parameter_names, alpha)

            weight = float(max(1, int(fit_res.num_examples))) if self.cfg.federated.aggregate_by_num_examples else 1.0
            total_weight += weight
            for base, delta in client_dense.items():
                accumulator = weighted_dense.get(base)
                if accumulator is None:
                    weighted_dense[base] = np.asarray(delta, dtype=np.float32) * weight
                else:
                    accumulator += np.asarray(delta, dtype=np.float32) * weight

        if total_weight <= 0.0:
            return _empty_parameters(), {}

        self.global_dense_by_base = {
            base: np.asarray(delta / total_weight, dtype=np.float32)
            for base, delta in weighted_dense.items()
        }

        aggregated_metrics = self._aggregate_fit_metrics(fit_metrics) or {}
        aggregated_parameters = self._parameters_for_rank(self.max_rank)

        self._save_checkpoint(server_round, aggregated_parameters, aggregated_metrics)
        self.logger.info(
            "round=%s clients=%s total_examples=%s mean_loss=%.6f mean_train_time_sec=%.3f bytes=%.0f",
            server_round,
            aggregated_metrics.get("num_clients", len(results)),
            aggregated_metrics.get("total_examples", 0),
            _to_float(aggregated_metrics.get("loss")),
            _to_float(aggregated_metrics.get("train_time_sec")),
            _to_float(aggregated_metrics.get("transmitted_bytes")),
        )
        return aggregated_parameters, aggregated_metrics
