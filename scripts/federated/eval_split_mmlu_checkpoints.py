from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from peft import load_peft_weights, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from lshaped.common.protocol import ClientBatchPayload, transmitted_bytes
from lshaped.common.simple_tokenizer import SimpleCharTokenizer
from lshaped.config import load_config
from lshaped.server.gemma_suffix_trainer import GemmaSuffixTrainer


@dataclass
class MMLUSample:
    question: str
    option_a: str
    option_b: str
    option_c: str
    option_d: str
    answer: str

    def prompt(self) -> str:
        return (
            f"Question: {self.question}\n"
            f"A. {self.option_a}\n"
            f"B. {self.option_b}\n"
            f"C. {self.option_c}\n"
            f"D. {self.option_d}\n"
            "Answer: "
        )


class EmbeddingBackend:
    def __init__(self, model_name_or_path: str, target_embedding_mode: str, max_seq_len: int) -> None:
        self.target_embedding_mode = target_embedding_mode
        if model_name_or_path == "__random_gemma__":
            self.tokenizer = SimpleCharTokenizer()
            layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
            config = Gemma3TextConfig(
                vocab_size=self.tokenizer.vocab_size,
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=32,
                max_position_embeddings=max_seq_len,
                sliding_window=min(64, max_seq_len),
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                layer_types=layer_types,
            )
            model = Gemma3ForCausalLM(config)
            embedding_layer = model.get_input_embeddings()
            self.embed_weight = embedding_layer.weight.detach().cpu()
            raw_scale = getattr(embedding_layer, "embed_scale", 1.0)
            self.embed_scale = float(raw_scale.item() if hasattr(raw_scale, "item") else raw_scale)
            del model
            return

        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype=torch.float32)
        embedding_layer = model.get_input_embeddings()
        self.tokenizer = tokenizer
        self.embed_weight = embedding_layer.weight.detach().cpu()
        raw_scale = getattr(embedding_layer, "embed_scale", 1.0)
        self.embed_scale = float(raw_scale.item() if hasattr(raw_scale, "item") else raw_scale)
        del model

    def answer_token_id(self, letter: str) -> int:
        letter = letter.strip().upper()
        if isinstance(self.tokenizer, SimpleCharTokenizer):
            ids = self.tokenizer.encode(letter, add_special_tokens=False)
            if not ids:
                raise RuntimeError(f"Simple tokenizer could not encode answer token {letter!r}")
            return int(ids[-1])
        spaced = self.tokenizer.encode(f" {letter}", add_special_tokens=False)
        if spaced:
            return int(spaced[-1])
        plain = self.tokenizer.encode(letter, add_special_tokens=False)
        if not plain:
            raise RuntimeError(f"Tokenizer could not encode answer token {letter!r}")
        return int(plain[-1])

    def encode_batch(self, prompts: list[str], max_seq_len: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if isinstance(self.tokenizer, SimpleCharTokenizer):
            input_ids, attention_mask = self.tokenizer.encode_batch(prompts, max_seq_len=max_seq_len)
            acts = self.embed_weight[input_ids] * self.embed_scale
            valid_lengths = attention_mask.sum(dim=1).to(torch.int32)
            return (
                acts.numpy().astype(np.float32),
                attention_mask.numpy().astype(np.int32),
                valid_lengths.numpy().astype(np.int32),
            )

        toks = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=max_seq_len,
            return_tensors="pt",
            add_special_tokens=True,
        )
        input_ids = toks["input_ids"].cpu()
        attention_mask = toks["attention_mask"].cpu().to(torch.int32)
        acts = self.embed_weight[input_ids] * self.embed_scale
        valid_lengths = attention_mask.sum(dim=1).to(torch.int32)
        return (
            acts.numpy().astype(np.float32),
            attention_mask.numpy().astype(np.int32),
            valid_lengths.numpy().astype(np.int32),
        )

    def target_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        token_tensor = torch.from_numpy(token_ids.astype(np.int64))
        emb = self.embed_weight[token_tensor]
        if self.target_embedding_mode == "scaled_input_embedding":
            emb = emb * self.embed_scale
        elif self.target_embedding_mode != "raw_embedding_weight":
            raise ValueError(f"Unsupported target_embedding_mode: {self.target_embedding_mode}")
        return emb.numpy().astype(np.float32)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate SplitLoRA MMLU checkpoints")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--eval-csv", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--only-rounds", default="")
    parser.add_argument("--gpu", default="")
    return parser.parse_args()


def load_mmlu_csv(path: str | Path) -> list[MMLUSample]:
    samples: list[MMLUSample] = []
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"question", "A", "B", "C", "D", "answer"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing MMLU columns: {sorted(missing)}")
        for row in reader:
            answer = (row.get("answer") or "").strip().upper()
            if answer not in {"A", "B", "C", "D"}:
                raise ValueError(f"Invalid answer {answer!r}")
            samples.append(
                MMLUSample(
                    question=row["question"],
                    option_a=row["A"],
                    option_b=row["B"],
                    option_c=row["C"],
                    option_d=row["D"],
                    answer=answer,
                )
            )
    if not samples:
        raise RuntimeError(f"No samples found in {path}")
    return samples


def shard_samples(samples: list[MMLUSample], client_ids: list[str]) -> dict[str, list[MMLUSample]]:
    shards = {client_id: [] for client_id in client_ids}
    for index, sample in enumerate(samples):
        client_id = client_ids[index % len(client_ids)]
        shards[client_id].append(sample)
    return shards


def batched(items: list[MMLUSample], batch_size: int) -> Iterable[list[MMLUSample]]:
    for offset in range(0, len(items), batch_size):
        yield items[offset : offset + batch_size]


def parse_round_filter(raw: str) -> set[int]:
    if not raw.strip():
        return set()
    result: set[int] = set()
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start_text, end_text = part.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if end < start:
                raise ValueError(f"Invalid round range: {part}")
            result.update(range(start, end + 1))
        else:
            result.add(int(part))
    return result


def make_payload(
    backend: EmbeddingBackend,
    client_id: str,
    round_id: int,
    batch_id: int,
    samples: list[MMLUSample],
    max_seq_len: int,
) -> ClientBatchPayload:
    prompts = [sample.prompt() for sample in samples]
    answers = [sample.answer for sample in samples]
    activation, attention_mask, valid_lengths = backend.encode_batch(prompts, max_seq_len=max_seq_len)
    token_ids = np.asarray([backend.answer_token_id(answer) for answer in answers], dtype=np.int32)
    target_embedding = backend.target_embeddings(token_ids)
    return ClientBatchPayload(
        client_id=client_id,
        batch_id=batch_id,
        mode="eval",
        task_type="multiple_choice",
        split_layer=0,
        activation=activation,
        target_embedding=target_embedding,
        attention_mask=attention_mask,
        target_token_ids=token_ids,
        valid_lengths=valid_lengths,
        answer_labels=answers,
        transmitted_bytes=transmitted_bytes(activation, target_embedding, attention_mask, token_ids, valid_lengths),
        server_round=round_id,
        client_backend="python_eval",
    )


def main() -> None:
    args = parse_args()
    if args.gpu:
        import os

        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    cfg = load_config(args.config)
    checkpoint_dir = Path(args.checkpoint_dir)
    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)

    trainer = GemmaSuffixTrainer(cfg)
    trainer.model.eval()
    if hasattr(trainer.queue, "items"):
        trainer.queue.items.clear()
    backend = EmbeddingBackend(
        model_name_or_path=cfg.model.model_name_or_path,
        target_embedding_mode=cfg.model.target_embedding_mode,
        max_seq_len=cfg.dataset.max_seq_len,
    )

    all_samples = load_mmlu_csv(args.eval_csv)
    shards = shard_samples(all_samples, cfg.dataset.client_ids)
    round_filter = parse_round_filter(args.only_rounds)

    checkpoints = sorted(checkpoint_dir.glob("round_*_adapter"))
    if round_filter:
        checkpoints = [path for path in checkpoints if int(path.name.split("_")[1]) in round_filter]
    if not checkpoints:
        raise RuntimeError(f"No adapter checkpoints found under {checkpoint_dir}")

    rows: list[dict[str, object]] = []
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "round",
                "checkpoint",
                "eval_loss",
                "eval_accuracy",
                "eval_examples",
                "eval_batches",
                "num_clients",
            ],
        )
        writer.writeheader()

        for checkpoint in checkpoints:
            round_id = int(checkpoint.name.split("_")[1])
            adapter_state = load_peft_weights(str(checkpoint), device="cpu")
            set_peft_model_state_dict(trainer.model, adapter_state, adapter_name="default")
            trainer.model.eval()

            total_examples = 0
            total_batches = 0
            weighted_loss = 0.0
            weighted_accuracy = 0.0

            for client_id in cfg.dataset.client_ids:
                samples = shards[client_id]
                for batch_id, batch_samples in enumerate(batched(samples, cfg.dataset.batch_size)):
                    payload = make_payload(
                        backend=backend,
                        client_id=client_id,
                        round_id=round_id,
                        batch_id=batch_id,
                        samples=batch_samples,
                        max_seq_len=cfg.dataset.max_seq_len,
                    )
                    metrics = trainer.eval_batch(payload)
                    batch_size = len(batch_samples)
                    total_examples += batch_size
                    total_batches += 1
                    weighted_loss += metrics.loss * batch_size
                    weighted_accuracy += metrics.accuracy * batch_size

            if total_examples <= 0:
                raise RuntimeError(f"No eval examples processed for checkpoint {checkpoint}")

            row = {
                "round": round_id,
                "checkpoint": checkpoint.name,
                "eval_loss": weighted_loss / total_examples,
                "eval_accuracy": weighted_accuracy / total_examples,
                "eval_examples": total_examples,
                "eval_batches": total_batches,
                "num_clients": len(cfg.dataset.client_ids),
            }
            writer.writerow(row)
            handle.flush()
            rows.append(row)
            print(
                f"[eval] round={row['round']} "
                f"loss={row['eval_loss']:.6f} "
                f"acc={row['eval_accuracy']:.6f} "
                f"examples={row['eval_examples']} "
                f"batches={row['eval_batches']}",
                flush=True,
            )

    summary = {
        "config": args.config,
        "checkpoint_dir": str(checkpoint_dir),
        "eval_csv": args.eval_csv,
        "num_checkpoints": len(rows),
        "rows": rows,
    }
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
