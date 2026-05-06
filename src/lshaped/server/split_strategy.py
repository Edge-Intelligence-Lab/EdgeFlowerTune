from __future__ import annotations

import concurrent.futures
from pathlib import Path
import time
from typing import Any

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters
from flwr.server.client_proxy import ClientProxy
import flwr.server.server as flwr_server_mod
from flwr.server.strategy import FedAvg

from lshaped.common.logging_utils import append_csv, append_jsonl
from lshaped.common.protocol import payload_from_fit_result
from lshaped.config import AppConfig
from lshaped.server.gemma_suffix_trainer import GemmaSuffixTrainer, StepOutput


_FIT_RECEIVE_TIMESTAMPS: dict[tuple[int, str], float] = {}
_ORIGINAL_FIT_CLIENT = flwr_server_mod.fit_client


def _timed_fit_clients(
    client_instructions,
    max_workers,
    timeout,
    group_id,
):
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        submitted_fs = {
            executor.submit(_ORIGINAL_FIT_CLIENT, client_proxy, ins, timeout, group_id): client_proxy
            for client_proxy, ins in client_instructions
        }
        results = []
        failures = []
        for future in concurrent.futures.as_completed(submitted_fs):
            client_proxy = submitted_fs[future]
            _FIT_RECEIVE_TIMESTAMPS[(int(group_id), client_proxy.cid)] = time.time()
            flwr_server_mod._handle_finished_future_after_fit(
                future=future,
                results=results,
                failures=failures,
            )
    return results, failures


flwr_server_mod.fit_clients = _timed_fit_clients


class SplitLoraStrategy(FedAvg):
    def __init__(self, cfg: AppConfig, logger) -> None:
        if cfg.model.training_mode.strip().lower() != "lora":
            raise ValueError("SplitLoRA requires model.training_mode=lora")
        self.cfg = cfg
        self.logger = logger
        self.output_dir = Path(cfg.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.trainer = GemmaSuffixTrainer(cfg)
        self._dummy_parameters = ndarrays_to_parameters([np.zeros((1,), dtype=np.float32)])
        self.round_dispatch_ts: dict[int, float] = {}

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.cfg.flower.min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=self.cfg.flower.min_available_clients,
            evaluate_fn=None,
            on_fit_config_fn=self._fit_config,
            on_evaluate_config_fn=None,
            accept_failures=self.cfg.federated.accept_failures,
            initial_parameters=self._dummy_parameters,
            fit_metrics_aggregation_fn=None,
            evaluate_metrics_aggregation_fn=None,
            inplace=True,
        )

    def _fit_config(self, server_round: int) -> dict[str, Any]:
        task_type = "next_token_lm" if self.cfg.dataset.source == "wikitext_raw" else "multiple_choice"
        dispatch_ts = time.time()
        self.round_dispatch_ts[int(server_round)] = dispatch_ts
        return {
            "server_round": server_round,
            "mode": "train",
            "task_type": task_type,
            "split_layer": self.cfg.client.split_layer,
            "local_steps": self.cfg.federated.local_steps,
            "batch_size": self.cfg.dataset.batch_size,
            "max_seq_len": self.cfg.dataset.max_seq_len,
            "server_dispatch_ts": dispatch_ts,
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Any]]:
        aggregation_start = time.perf_counter()
        if failures and not self.cfg.federated.accept_failures:
            raise RuntimeError(f"SplitLoRA round {server_round} had failures: {failures}")
        if not results:
            raise RuntimeError(f"SplitLoRA round {server_round} received no client results")
        if failures:
            self.logger.warning(
                "SplitLoRA round %s proceeding with %s results and %s failures",
                server_round,
                len(results),
                len(failures),
            )

        client_metrics: list[tuple[str, Any, StepOutput, int]] = []
        for client_proxy, fit_res in results:
            payload = payload_from_fit_result(fit_res.parameters, fit_res.metrics)
            receive_ts = _FIT_RECEIVE_TIMESTAMPS.get((int(server_round), client_proxy.cid), 0.0)
            if payload.server_dispatch_ts <= 0.0:
                payload.server_dispatch_ts = self.round_dispatch_ts.get(int(server_round), 0.0)
            payload.server_receive_ts = receive_ts
            client_upload_time_sec = float(fit_res.metrics.get("client_upload_write_time_sec", 0.0) or 0.0)
            client_download_time_sec = float(fit_res.metrics.get("client_download_read_time_sec", 0.0) or 0.0)
            if client_download_time_sec > 0.0:
                payload.download_time_sec = client_download_time_sec
            if client_upload_time_sec > 0.0:
                payload.upload_time_sec = client_upload_time_sec
            elif receive_ts > 0.0 and payload.response_ready_ts > 0.0:
                payload.upload_time_sec = abs(receive_ts - payload.response_ready_ts)
            metrics = self.trainer.train_batch(payload)
            client_id = str(fit_res.metrics.get("client_id", client_proxy.cid))
            self.trainer.log_step(server_round, client_id, "train", payload, metrics)
            client_metrics.append((client_id, payload, metrics, int(getattr(fit_res, "num_examples", 0) or 0)))

        total_examples = sum(num_examples for _, _, _, num_examples in client_metrics)
        num_clients = len(client_metrics)
        mean_loss = sum(metrics.loss for _, _, metrics, _ in client_metrics) / num_clients
        mean_accuracy = sum(metrics.accuracy for _, _, metrics, _ in client_metrics) / num_clients
        mean_train_time_sec = sum(metrics.round_time_sec for _, _, metrics, _ in client_metrics) / num_clients
        mean_step_time_sec = sum(metrics.mean_step_time_sec for _, _, metrics, _ in client_metrics) / num_clients
        max_step_time_sec = max(metrics.max_step_time_sec for _, _, metrics, _ in client_metrics)
        mean_bytes = sum(metrics.transmitted_bytes for _, _, metrics, _ in client_metrics) / num_clients
        mean_queue_size = sum(metrics.queue_size for _, _, metrics, _ in client_metrics) / num_clients
        mean_download_time_sec = sum(payload.download_time_sec for _, payload, _, _ in client_metrics) / num_clients
        mean_upload_time_sec = sum(payload.upload_time_sec for _, payload, _, _ in client_metrics) / num_clients
        mean_download_bytes = sum(payload.download_bytes for _, payload, _, _ in client_metrics) / num_clients
        mean_upload_bytes = sum(payload.upload_bytes for _, payload, _, _ in client_metrics) / num_clients
        mean_avg_rss_mb = sum(payload.avg_rss_mb for _, payload, _, _ in client_metrics) / num_clients
        mean_peak_rss_mb = sum(payload.peak_rss_mb for _, payload, _, _ in client_metrics) / num_clients
        mean_client_power_w = sum(payload.client_power_w for _, payload, _, _ in client_metrics) / num_clients
        aggregation_time_sec = time.perf_counter() - aggregation_start

        summary = {
            "round": server_round,
            "num_clients": num_clients,
            "num_failures": len(failures),
            "total_examples": total_examples,
            "mean_loss": mean_loss,
            "mean_accuracy": mean_accuracy,
            "mean_train_time_sec": mean_train_time_sec,
            "mean_step_time_sec": mean_step_time_sec,
            "max_step_time_sec": max_step_time_sec,
            "aggregation_time_sec": aggregation_time_sec,
            "mean_download_time_sec": mean_download_time_sec,
            "mean_upload_time_sec": mean_upload_time_sec,
            "mean_download_bytes": mean_download_bytes,
            "mean_upload_bytes": mean_upload_bytes,
            "mean_transmitted_bytes": mean_bytes,
            "mean_avg_rss_mb": mean_avg_rss_mb,
            "mean_peak_rss_mb": mean_peak_rss_mb,
            "mean_client_power_w": mean_client_power_w,
            "mean_queue_size": mean_queue_size,
        }
        append_csv(self.output_dir / "summary_rounds.csv", summary)
        append_jsonl(self.output_dir / "summary_rounds.jsonl", summary)

        if self.cfg.logging.save_every_rounds > 0 and server_round % self.cfg.logging.save_every_rounds == 0:
            self.trainer.save_checkpoint(server_round, include_optimizer=self.cfg.logging.save_optimizer)

        self.logger.info(
            "round=%s clients=%s total_examples=%s mean_loss=%.6f mean_train_time_sec=%.3f mean_step_time_sec=%.3f bytes=%s aggregation_time_sec=%.3f",
            server_round,
            num_clients,
            total_examples,
            mean_loss,
            mean_train_time_sec,
            mean_step_time_sec,
            int(mean_bytes),
            aggregation_time_sec,
        )
        return self._dummy_parameters, summary
