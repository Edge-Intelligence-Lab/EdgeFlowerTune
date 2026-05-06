from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from lshaped.common.logging_utils import append_csv, append_jsonl
from lshaped.config import AppConfig

try:
    from safetensors.torch import save_file as save_safetensors_file
except Exception:  # pragma: no cover - optional dependency fallback
    save_safetensors_file = None


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


@dataclass
class AdapterParamSpec:
    name: str
    shape: tuple[int, ...]


class GemmaLoraAdapterSchema:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cpu")
        self.model_dtype = torch.float32
        self.output_dir = Path(cfg.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.adapter_path = self.output_dir / "adapter_schema.json"

        self.model = self._build_model()
        self.param_specs = self._collect_trainable_params()
        if not self.param_specs:
            raise RuntimeError("No trainable LoRA parameters found for classic FedAvg")
        self.initial_parameters = ndarrays_to_parameters(
            [np.zeros(spec.shape, dtype=np.float32) for spec in self.param_specs]
        )
        self._write_schema()

    def _build_model(self):
        if self.cfg.model.training_mode.strip().lower() != "lora":
            raise ValueError(
                "Classic FedAvg+LoRA requires model.training_mode=lora, "
                f"got {self.cfg.model.training_mode!r}"
            )

        if self.cfg.model.model_name_or_path == "__random_gemma__":
            layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
            config = Gemma3TextConfig(
                vocab_size=128,
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=32,
                max_position_embeddings=self.cfg.dataset.max_seq_len,
                sliding_window=min(64, self.cfg.dataset.max_seq_len),
                pad_token_id=0,
                bos_token_id=1,
                eos_token_id=2,
                layer_types=layer_types,
            )
            model = Gemma3ForCausalLM(config).to(self.device, dtype=self.model_dtype)
        else:
            model = AutoModelForCausalLM.from_pretrained(
                self.cfg.model.model_name_or_path,
                torch_dtype=self.model_dtype,
            ).to(self.device)

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.model.lora_r,
            lora_alpha=self.cfg.model.lora_alpha,
            lora_dropout=self.cfg.model.lora_dropout,
            target_modules=self.cfg.model.lora_target_modules,
            bias="none",
        )
        model = get_peft_model(model, lora_cfg).to(self.device)
        if self.cfg.model.freeze_input_embeddings:
            for param in model.get_input_embeddings().parameters():
                param.requires_grad = False
        model.eval()
        return model

    def _collect_trainable_params(self) -> list[AdapterParamSpec]:
        specs: list[AdapterParamSpec] = []
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            specs.append(AdapterParamSpec(name=name, shape=tuple(int(dim) for dim in param.shape)))
        return specs

    def _write_schema(self) -> None:
        payload = {
            "algorithm": "fedavg_lora",
            "client_mode": self.cfg.client.client_mode,
            "model_name_or_path": self.cfg.model.model_name_or_path,
            "training_mode": self.cfg.model.training_mode,
            "lora": {
                "r": self.cfg.model.lora_r,
                "alpha": self.cfg.model.lora_alpha,
                "dropout": self.cfg.model.lora_dropout,
                "target_modules": list(self.cfg.model.lora_target_modules),
            },
            "param_specs": [
                {"name": spec.name, "shape": list(spec.shape), "dtype": "float32"}
                for spec in self.param_specs
            ],
        }
        self.adapter_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    def checkpoint_dir(self, round_num: int) -> Path:
        return self.output_dir / "checkpoints" / f"round_{round_num:06d}"

    def save_round_checkpoint(
        self,
        round_num: int,
        parameters: Parameters,
        aggregated_metrics: dict[str, Any],
        client_ids: list[str],
        num_examples: int,
        total_weight: float,
    ) -> Path:
        ndarrays = parameters_to_ndarrays(parameters)
        if len(ndarrays) != len(self.param_specs):
            raise ValueError(
                f"Adapter tensor count mismatch: expected {len(self.param_specs)}, got {len(ndarrays)}"
            )

        state: dict[str, torch.Tensor] = {}
        for spec, array in zip(self.param_specs, ndarrays, strict=True):
            tensor = torch.from_numpy(np.asarray(array)).detach().cpu().to(torch.float32)
            if tuple(int(dim) for dim in tensor.shape) != spec.shape:
                raise ValueError(
                    f"Shape mismatch for {spec.name}: expected {spec.shape}, got {tuple(tensor.shape)}"
                )
            state[spec.name] = tensor

        round_dir = self.checkpoint_dir(round_num)
        round_dir.mkdir(parents=True, exist_ok=True)
        adapter_path = round_dir / "adapter.safetensors"
        metadata = {
            "algorithm": "fedavg_lora",
            "round": str(round_num),
            "num_examples": str(num_examples),
            "total_weight": str(total_weight),
            "client_ids": ",".join(client_ids),
            "param_count": str(len(self.param_specs)),
        }
        if save_safetensors_file is not None:
            save_safetensors_file(state, str(adapter_path), metadata=metadata)
        else:  # pragma: no cover - fallback path for environments without safetensors
            torch.save({"state_dict": state, "metadata": metadata}, round_dir / "adapter.pt")

        manifest = {
            "round": round_num,
            "checkpoint_path": str(adapter_path if save_safetensors_file is not None else round_dir / "adapter.pt"),
            "client_ids": client_ids,
            "num_examples": num_examples,
            "total_weight": total_weight,
            "aggregated_metrics": _json_safe(aggregated_metrics),
            "param_specs": [
                {"name": spec.name, "shape": list(spec.shape), "dtype": "float32"}
                for spec in self.param_specs
            ],
        }
        (round_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        row = {
            "round": round_num,
            "num_clients": len(client_ids),
            "num_examples": num_examples,
            "total_weight": total_weight,
            "checkpoint_path": manifest["checkpoint_path"],
        }
        row.update({k: _json_safe(v) for k, v in aggregated_metrics.items()})
        append_csv(self.output_dir / "metrics.csv", row)
        append_jsonl(self.output_dir / "metrics.jsonl", row)
        return round_dir


class ClassicFedAvgLoRAStrategy(FedAvg):
    def __init__(self, cfg: AppConfig, logger) -> None:
        self.cfg = cfg
        self.logger = logger
        self.schema = GemmaLoraAdapterSchema(cfg)
        self.output_dir = Path(cfg.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Classic FedAvg+LoRA initialized: params=%s checkpoint_dir=%s",
            len(self.schema.param_specs),
            self.output_dir / "checkpoints",
        )

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=0.0,
            min_fit_clients=self.cfg.flower.min_fit_clients,
            min_evaluate_clients=0,
            min_available_clients=self.cfg.flower.min_available_clients,
            evaluate_fn=None,
            on_fit_config_fn=self._fit_config,
            on_evaluate_config_fn=None,
            accept_failures=False,
            initial_parameters=self.schema.initial_parameters,
            fit_metrics_aggregation_fn=self._aggregate_fit_metrics,
            evaluate_metrics_aggregation_fn=None,
            inplace=True,
        )

    def _fit_config(self, server_round: int) -> dict[str, Any]:
        return {
            "server_round": server_round,
            "algorithm": self.cfg.federated.algorithm,
            "client_mode": self.cfg.client.client_mode,
            "batch_size": self.cfg.dataset.batch_size,
            "max_seq_len": self.cfg.dataset.max_seq_len,
            "local_steps": self.cfg.federated.local_steps,
            "learning_rate": self.cfg.model.learning_rate,
            "weight_decay": self.cfg.model.weight_decay,
            "grad_clip_norm": self.cfg.model.grad_clip_norm,
            "training_mode": self.cfg.model.training_mode,
            "lora_r": self.cfg.model.lora_r,
            "lora_alpha": self.cfg.model.lora_alpha,
            "lora_dropout": self.cfg.model.lora_dropout,
            "lora_target_modules": ",".join(self.cfg.model.lora_target_modules),
            "aggregate_by_num_examples": self.cfg.federated.aggregate_by_num_examples,
        }

    def _aggregate_fit_metrics(self, metrics: list[tuple[int, dict[str, Any]]]) -> dict[str, Any]:
        if not metrics:
            return {}
        total_examples = sum(max(int(num_examples), 0) for num_examples, _ in metrics)
        total_weight = float(total_examples if self.cfg.federated.aggregate_by_num_examples else len(metrics))
        if total_weight <= 0.0:
            total_weight = float(len(metrics))

        numeric_keys: set[str] = set()
        for _, entry in metrics:
            for key, value in entry.items():
                if isinstance(value, (bool, int, float, np.integer, np.floating)):
                    numeric_keys.add(str(key))

        aggregated: dict[str, Any] = {
            "num_clients": len(metrics),
            "total_examples": total_examples,
            "total_weight": total_weight,
        }
        for key in sorted(numeric_keys):
            numerator = 0.0
            for num_examples, entry in metrics:
                value = entry.get(key)
                if not isinstance(value, (bool, int, float, np.integer, np.floating)):
                    continue
                weight = float(num_examples if self.cfg.federated.aggregate_by_num_examples else 1.0)
                numerator += float(value) * weight
            aggregated[key] = numerator / total_weight if total_weight > 0.0 else 0.0
        return aggregated

    def aggregate_fit(
        self,
        server_round: int,
        results: list[tuple[ClientProxy, Any]],
        failures: list[Any],
    ) -> tuple[Parameters | None, dict[str, Any]]:
        if failures:
            raise RuntimeError(f"Classic FedAvg round {server_round} had failures: {failures}")

        aggregated_parameters, aggregated_metrics = super().aggregate_fit(server_round, results, failures)
        if aggregated_parameters is None:
            return None, aggregated_metrics

        client_ids = []
        total_examples = 0
        total_weight = 0.0
        for client_proxy, fit_res in results:
            client_id = str(fit_res.metrics.get("client_id", client_proxy.cid))
            client_ids.append(client_id)
            total_examples += int(getattr(fit_res, "num_examples", 0) or 0)
            total_weight += float(
                getattr(fit_res, "num_examples", 0) or 0
                if self.cfg.federated.aggregate_by_num_examples
                else 1.0
            )

        round_dir = self.schema.save_round_checkpoint(
            round_num=server_round,
            parameters=aggregated_parameters,
            aggregated_metrics=aggregated_metrics,
            client_ids=client_ids,
            num_examples=total_examples,
            total_weight=total_weight,
        )
        self.logger.info(
            "round=%s federated_algorithm=%s clients=%s total_examples=%s total_weight=%.1f checkpoint=%s",
            server_round,
            self.cfg.federated.algorithm,
            len(results),
            total_examples,
            total_weight,
            round_dir,
        )
        return aggregated_parameters, aggregated_metrics


# Compatibility alias for older imports. The active implementation is classic FedAvg+LoRA.
LShapedStrategy = ClassicFedAvgLoRAStrategy
