from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    run_name: str
    output_dir: str
    seed: int = 7


@dataclass
class FlowerConfig:
    server_address: str
    grpc_max_message_length: int
    num_rounds: int
    min_available_clients: int
    min_fit_clients: int
    sample_clients: int
    round_timeout: int
    eval_every_rounds: int = 0
    client_wait_timeout: int = 60


@dataclass
class FederatedConfig:
    algorithm: str = "fedavg_lora"
    local_steps: int = 1
    local_epochs: int = 0
    grad_accum_steps: int = 1
    prox_mu: float = 0.0
    client_lora_ranks: dict[str, int] = field(default_factory=dict)
    aggregate_by_num_examples: bool = True
    accept_failures: bool = False


@dataclass
class DatasetConfig:
    source: str
    source_path: str
    split: str
    eval_split: str
    num_clients: int
    client_ids: list[str]
    batch_size: int
    max_seq_len: int
    partition_mode: str
    dirichlet_alpha: float
    smoke_test_examples: int = 0


@dataclass
class ModelConfig:
    model_name_or_path: str
    device: str
    dtype: str
    target_embedding_mode: str
    freeze_input_embeddings: bool
    grad_clip_norm: float
    learning_rate: float
    weight_decay: float
    training_mode: str = "full"
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )


@dataclass
class LossConfig:
    temperature: float
    queue_size: int
    use_in_batch_negatives: bool = True


@dataclass
class ClientConfig:
    backend: str = "mobilefinetuner_cpp"
    client_mode: str = "classic_lora"
    split_layer: int = 0
    upload_dtype: str = "float32"
    pad_to_max_seq_len: bool = False


@dataclass
class LoggingConfig:
    save_every_rounds: int
    log_every_rounds: int
    save_optimizer: bool = True


@dataclass
class AppConfig:
    runtime: RuntimeConfig
    flower: FlowerConfig
    federated: FederatedConfig
    dataset: DatasetConfig
    model: ModelConfig
    loss: LossConfig
    client: ClientConfig
    logging: LoggingConfig


def _load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise TypeError(f"Config root must be a mapping, got: {type(data)!r}")
    return data


def load_config(path: str | Path) -> AppConfig:
    raw = _load_yaml(path)
    cfg = AppConfig(
        runtime=RuntimeConfig(**raw["runtime"]),
        flower=FlowerConfig(**raw["flower"]),
        federated=FederatedConfig(**raw.get("federated", {})),
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        loss=LossConfig(**raw["loss"]),
        client=ClientConfig(**raw["client"]),
        logging=LoggingConfig(**raw["logging"]),
    )
    if cfg.federated.algorithm not in {"fedavg_lora", "fedprox_lora", "flexlora", "splitlora", "localonly_lora"}:
        raise ValueError(f"Unsupported federated.algorithm: {cfg.federated.algorithm}")
    if cfg.federated.algorithm == "fedprox_lora" and cfg.federated.prox_mu <= 0.0:
        raise ValueError("fedprox_lora requires federated.prox_mu > 0")
    if cfg.federated.algorithm in {"fedavg_lora", "localonly_lora"} and cfg.federated.prox_mu < 0.0:
        raise ValueError("federated.prox_mu must be >= 0")
    if cfg.federated.algorithm == "splitlora":
        if cfg.model.training_mode.strip().lower() != "lora":
            raise ValueError("splitlora requires model.training_mode=lora")
        if cfg.client.split_layer != 0:
            raise ValueError("splitlora currently requires client.split_layer == 0")
    if cfg.federated.algorithm == "flexlora":
        if cfg.federated.prox_mu != 0.0:
            raise ValueError("flexlora currently requires federated.prox_mu == 0")
        if not cfg.federated.client_lora_ranks:
            raise ValueError("flexlora requires federated.client_lora_ranks")
        missing = [client_id for client_id in cfg.dataset.client_ids if client_id not in cfg.federated.client_lora_ranks]
        if missing:
            raise ValueError(
                "flexlora requires ranks for all dataset.client_ids, missing: " + ",".join(missing)
            )
        for client_id, rank in cfg.federated.client_lora_ranks.items():
            if rank <= 0:
                raise ValueError(f"flexlora rank must be > 0 for client {client_id}, got {rank}")
    return cfg
