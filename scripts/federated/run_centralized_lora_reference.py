from __future__ import annotations

import argparse
import csv
import json
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, TaskType, get_peft_model
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class RuntimeConfig:
    run_name: str
    output_dir: str
    seed: int = 7


@dataclass
class DatasetConfig:
    source_path: str
    num_clients: int
    client_ids: list[str]
    batch_size: int
    max_seq_len: int
    partition_mode: str
    shuffle: bool = True
    answer_prefix: str = " "


@dataclass
class ModelConfig:
    model_name_or_path: str
    device: str
    dtype: str
    grad_clip_norm: float
    learning_rate: float
    weight_decay: float
    training_mode: str = "lora"
    lora_r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    lora_target_modules: list[str] = field(default_factory=list)


@dataclass
class TrainConfig:
    max_steps: int
    logging_steps: int
    save_every_steps: int = 0


@dataclass
class CentralizedReferenceConfig:
    runtime: RuntimeConfig
    dataset: DatasetConfig
    model: ModelConfig
    train: TrainConfig


class MaskedMMLUDataset(Dataset):
    def __init__(self, csv_path: Path, tokenizer, seq_len: int, answer_prefix: str) -> None:
        self.items: list[dict[str, torch.Tensor]] = []
        with csv_path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                prompt = (
                    f"Question: {row['question']}\n"
                    f"A. {row['A']}\n"
                    f"B. {row['B']}\n"
                    f"C. {row['C']}\n"
                    f"D. {row['D']}\n"
                    f"Answer: "
                )
                answer = answer_prefix + str(row["answer"]).strip().upper()
                prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
                answer_ids = tokenizer.encode(answer, add_special_tokens=False)
                eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.pad_token_id
                if eos_id is None:
                    raise RuntimeError("Tokenizer must expose eos_token_id or pad_token_id")

                input_ids = prompt_ids + answer_ids + [eos_id]
                if len(input_ids) > seq_len:
                    continue
                labels = [-100] * len(prompt_ids) + answer_ids + [-100]
                attention_mask = [1] * len(input_ids)
                while len(input_ids) < seq_len:
                    input_ids.append(eos_id)
                    labels.append(-100)
                    attention_mask.append(0)

                self.items.append(
                    {
                        "input_ids": torch.tensor(input_ids, dtype=torch.long),
                        "labels": torch.tensor(labels, dtype=torch.long),
                        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
                    }
                )
        if not self.items:
            raise RuntimeError(f"No usable samples found in {csv_path}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.items[index]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run centralized Gemma-270M LoRA reference training")
    parser.add_argument("--config", required=True, help="Path to centralized YAML config")
    return parser.parse_args()


def load_config(path: str | Path) -> CentralizedReferenceConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    return CentralizedReferenceConfig(
        runtime=RuntimeConfig(**raw["runtime"]),
        dataset=DatasetConfig(**raw["dataset"]),
        model=ModelConfig(**raw["model"]),
        train=TrainConfig(**raw["train"]),
    )


def resolve_dtype(name: str) -> torch.dtype:
    normalized = name.strip().lower()
    if normalized in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if normalized in {"fp16", "float16"}:
        return torch.float16
    return torch.float32


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed)

    output_dir = Path(cfg.runtime.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = MaskedMMLUDataset(
        csv_path=Path(cfg.dataset.source_path),
        tokenizer=tokenizer,
        seq_len=cfg.dataset.max_seq_len,
        answer_prefix=cfg.dataset.answer_prefix,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.dataset.batch_size,
        shuffle=cfg.dataset.shuffle,
        drop_last=False,
    )

    device = torch.device(cfg.model.device)
    dtype = resolve_dtype(cfg.model.dtype)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.model.model_name_or_path,
        torch_dtype=dtype,
    )
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
        target_modules=list(cfg.model.lora_target_modules),
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=cfg.model.learning_rate,
        weight_decay=cfg.model.weight_decay,
    )

    metrics_path = output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "step",
                "loss",
                "train_time_sec",
                "num_examples_seen",
            ],
        )
        writer.writeheader()

        step = 0
        num_examples_seen = 0
        while step < cfg.train.max_steps:
            for batch in loader:
                if step >= cfg.train.max_steps:
                    break
                start_time = time.perf_counter()
                batch = {key: value.to(device) for key, value in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.model.grad_clip_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_time_sec = time.perf_counter() - start_time

                step += 1
                num_examples_seen += int(batch["input_ids"].size(0))
                writer.writerow(
                    {
                        "step": step,
                        "loss": float(loss.item()),
                        "train_time_sec": train_time_sec,
                        "num_examples_seen": num_examples_seen,
                    }
                )
                handle.flush()

                if cfg.train.logging_steps > 0 and step % cfg.train.logging_steps == 0:
                    print(
                        f"[centralized] step={step} loss={loss.item():.6f} "
                        f"examples={num_examples_seen} train_time_sec={train_time_sec:.3f}"
                    )
                if cfg.train.save_every_steps > 0 and step % cfg.train.save_every_steps == 0:
                    model.save_pretrained(checkpoints_dir / f"step_{step:06d}_adapter")

    summary = {
        "run_name": cfg.runtime.run_name,
        "num_clients_reference": cfg.dataset.num_clients,
        "client_ids_reference": cfg.dataset.client_ids,
        "dataset_source_path": cfg.dataset.source_path,
        "partition_mode_reference": cfg.dataset.partition_mode,
        "max_steps": cfg.train.max_steps,
        "batch_size": cfg.dataset.batch_size,
        "max_seq_len": cfg.dataset.max_seq_len,
        "lora_r": cfg.model.lora_r,
        "lora_alpha": cfg.model.lora_alpha,
        "lora_target_modules": cfg.model.lora_target_modules,
        "num_examples_total": len(dataset),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    model.save_pretrained(checkpoints_dir / "final_adapter")
    tokenizer.save_pretrained(checkpoints_dir / "final_adapter")
    print(f"[centralized] saved final adapter to {checkpoints_dir / 'final_adapter'}")


if __name__ == "__main__":
    main()
