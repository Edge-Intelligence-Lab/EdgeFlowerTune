from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from flwr.common import (
    FitIns,
    GetParametersIns,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from safetensors.numpy import save_file

from lshaped.config import AppConfig
from lshaped.server.classic_fedavg_strategy import (
    ClassicFedAvgLoraStrategy,
    _to_float,
    _to_str,
)


def _clone_parameters(parameters: Parameters) -> Parameters:
    return ndarrays_to_parameters(
        [np.asarray(array, dtype=np.float32).copy() for array in parameters_to_ndarrays(parameters)]
    )


class LocalOnlyLoraStrategy(ClassicFedAvgLoraStrategy):
    def __init__(self, cfg: AppConfig, logger) -> None:
        super().__init__(cfg=cfg, logger=logger)
        self.initial_parameters: Parameters | None = None
        self.client_parameters_by_id: dict[str, Parameters] = {}
        self.client_id_by_cid: dict[str, str] = {}

    def initialize_parameters(self, client_manager: ClientManager) -> Parameters | None:
        if self.initial_parameters is not None:
            return _clone_parameters(self.initial_parameters)

        available = self._wait_for_required_clients(client_manager)
        clients = client_manager.sample(num_clients=1, min_num_clients=1)
        if not clients:
            raise RuntimeError(f"No clients available after wait (available={available})")

        get_res = clients[0].get_parameters(
            GetParametersIns(config={}),
            timeout=self.cfg.flower.round_timeout,
            group_id=0,
        )
        self.initial_parameters = _clone_parameters(get_res.parameters)
        self.logger.info("Initialized local-only baseline from client cid=%s", clients[0].cid)
        return _clone_parameters(self.initial_parameters)

    def _parameters_for_client(self, client: ClientProxy) -> Parameters:
        client_id = self.client_id_by_cid.get(client.cid)
        if client_id and client_id in self.client_parameters_by_id:
            return _clone_parameters(self.client_parameters_by_id[client_id])
        if self.initial_parameters is None:
            raise RuntimeError("localonly_lora requires initialized parameters before configure_fit")
        return _clone_parameters(self.initial_parameters)

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
            scheduled.append((client, FitIns(parameters=self._parameters_for_client(client), config=fit_config)))
        return scheduled

    def _save_client_checkpoint(self, server_round: int, client_id: str, parameters: Parameters) -> None:
        if self.cfg.logging.save_every_rounds <= 0:
            return
        if server_round % self.cfg.logging.save_every_rounds != 0:
            return

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
            "client_id": client_id,
            "lora_r": str(self.cfg.model.lora_r),
            "lora_alpha": str(self.cfg.model.lora_alpha),
            "target_modules": ",".join(self.cfg.model.lora_target_modules),
        }
        checkpoint_dir = Path(self.cfg.runtime.output_dir) / "checkpoints" / client_id
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        save_file(
            tensor_map,
            str(checkpoint_dir / f"round_{server_round:06d}.safetensors"),
            metadata=metadata,
        )

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Scalar]]:
        if failures:
            self.logger.warning("Round %s failures: %s", server_round, len(failures))
        if not results:
            if self.initial_parameters is None:
                return None, {}
            return _clone_parameters(self.initial_parameters), {}

        self._record_client_metrics(server_round, results)
        fit_metrics: list[tuple[int, dict[str, Scalar]]] = []

        for client_proxy, fit_res in results:
            metrics = fit_res.metrics
            client_id = _to_str(metrics.get("client_id"), client_proxy.cid)
            self.client_id_by_cid[client_proxy.cid] = client_id
            self.client_parameters_by_id[client_id] = _clone_parameters(fit_res.parameters)
            fit_metrics.append((int(fit_res.num_examples), metrics))

            parameter_names = _to_str(metrics.get("parameter_names"))
            if parameter_names and self.parameter_names is None:
                self.parameter_names = [name for name in parameter_names.split(",") if name]

            self._save_client_checkpoint(server_round, client_id, fit_res.parameters)

        aggregated_metrics = self._aggregate_fit_metrics(fit_metrics) or {}
        self.logger.info(
            "round=%s clients=%s total_examples=%s mean_loss=%.6f mean_objective_loss=%.6f mean_prox_term=%.6f mean_train_time_sec=%.3f bytes=%.0f local_only=true",
            server_round,
            aggregated_metrics.get("num_clients", len(results)),
            aggregated_metrics.get("total_examples", 0),
            _to_float(aggregated_metrics.get("loss")),
            _to_float(aggregated_metrics.get("objective_loss")),
            _to_float(aggregated_metrics.get("prox_term")),
            _to_float(aggregated_metrics.get("train_time_sec")),
            _to_float(aggregated_metrics.get("transmitted_bytes")),
        )
        if self.initial_parameters is None:
            return None, aggregated_metrics
        return _clone_parameters(self.initial_parameters), aggregated_metrics
