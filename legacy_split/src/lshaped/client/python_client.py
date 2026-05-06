from __future__ import annotations

import argparse
import random
import time
from dataclasses import dataclass

import flwr as fl
import numpy as np
import psutil
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from lshaped.common.protocol import ClientBatchPayload, transmitted_bytes
from lshaped.common.simple_tokenizer import SimpleCharTokenizer
from lshaped.config import AppConfig, load_config
from lshaped.data.mmlu import ClientShard, load_samples, shard_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run loopback Python client")
    parser.add_argument("--config", required=True)
    parser.add_argument("--client-id", required=True)
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def current_rss_mb() -> float:
    return float(psutil.Process().memory_info().rss / (1024 * 1024))


@dataclass
class EmbeddingBackend:
    tokenizer: object
    embed_weight: torch.Tensor
    embed_scale: float
    target_embedding_mode: str

    @classmethod
    def from_model(cls, cfg: AppConfig) -> "EmbeddingBackend":
        if cfg.model.model_name_or_path == "__random_gemma__":
            tokenizer = SimpleCharTokenizer()
            layer_types = ["sliding_attention", "full_attention", "sliding_attention", "full_attention"]
            config = Gemma3TextConfig(
                vocab_size=tokenizer.vocab_size,
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=4,
                num_attention_heads=4,
                num_key_value_heads=1,
                head_dim=32,
                max_position_embeddings=cfg.dataset.max_seq_len,
                sliding_window=min(64, cfg.dataset.max_seq_len),
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                layer_types=layer_types,
            )
            model = Gemma3ForCausalLM(config)
            embedding_layer = model.get_input_embeddings()
            embed_weight = embedding_layer.weight.detach().cpu()
            embed_scale = float(getattr(embedding_layer, "embed_scale", torch.tensor(1.0)).item())
            del model
            return cls(
                tokenizer=tokenizer,
                embed_weight=embed_weight,
                embed_scale=embed_scale,
                target_embedding_mode=cfg.model.target_embedding_mode,
            )

        tokenizer = AutoTokenizer.from_pretrained(cfg.model.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(cfg.model.model_name_or_path, torch_dtype=torch.float32)
        embedding_layer = model.get_input_embeddings()
        embed_weight = embedding_layer.weight.detach().cpu()
        embed_scale = float(getattr(embedding_layer, "embed_scale", torch.tensor(1.0)).item())
        del model
        return cls(
            tokenizer=tokenizer,
            embed_weight=embed_weight,
            embed_scale=embed_scale,
            target_embedding_mode=cfg.model.target_embedding_mode,
        )

    def answer_token_id(self, letter: str) -> int:
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
            return acts.numpy().astype(np.float32), attention_mask.numpy().astype(np.int32), valid_lengths.numpy().astype(np.int32)

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
        return acts.numpy().astype(np.float32), attention_mask.numpy().astype(np.int32), valid_lengths.numpy().astype(np.int32)

    def target_embeddings(self, token_ids: np.ndarray) -> np.ndarray:
        token_tensor = torch.from_numpy(token_ids.astype(np.int64))
        emb = self.embed_weight[token_tensor]
        if self.target_embedding_mode == "scaled_input_embedding":
            emb = emb * self.embed_scale
        elif self.target_embedding_mode != "raw_embedding_weight":
            raise ValueError(f"Unsupported target_embedding_mode: {self.target_embedding_mode}")
        return emb.numpy().astype(np.float32)


class LShapedNumPyClient(fl.client.NumPyClient):
    def __init__(self, cfg: AppConfig, client_id: str) -> None:
        self.cfg = cfg
        self.client_id = client_id
        self.backend = EmbeddingBackend.from_model(cfg)

        train_samples = load_samples(cfg.dataset, cfg.dataset.split)
        shards = shard_samples(train_samples, cfg.dataset, cfg.runtime.seed)
        if client_id not in shards:
            raise KeyError(f"Client id {client_id!r} not found in shard map")
        self.train_shard = ClientShard(shards[client_id], cfg.dataset.batch_size)

        eval_samples = load_samples(cfg.dataset, cfg.dataset.eval_split)
        eval_shards = shard_samples(eval_samples, cfg.dataset, cfg.runtime.seed + 1)
        self.eval_shard = ClientShard(eval_shards.get(client_id, shards[client_id]), cfg.dataset.batch_size)
        self.batch_id = 0

    def get_parameters(self, config):  # noqa: ANN001
        return [np.zeros((1,), dtype=np.float32)]

    def fit(self, parameters, config):  # noqa: ANN001
        round_start = time.perf_counter()
        mode = str(config.get("mode", "train"))
        split_layer = int(config.get("split_layer", 0))
        max_seq_len = int(config.get("max_seq_len", self.cfg.dataset.max_seq_len))
        server_round = int(config.get("server_round", -1))
        assert split_layer == 0, "Current prototype only supports split_layer=0"

        shard = self.eval_shard if mode == "eval" else self.train_shard
        batch = shard.next_batch()
        prompts = [sample.prompt() for sample in batch]
        answers = [sample.answer for sample in batch]
        token_ids = np.asarray([self.backend.answer_token_id(ans) for ans in answers], dtype=np.int32)
        encode_start = time.perf_counter()
        activation, attention_mask, valid_lengths = self.backend.encode_batch(prompts, max_seq_len=max_seq_len)
        target_embedding = self.backend.target_embeddings(token_ids)
        encode_time = time.perf_counter() - encode_start
        serialize_start = time.perf_counter()
        payload = ClientBatchPayload(
            client_id=self.client_id,
            batch_id=self.batch_id,
            mode=mode,
            split_layer=split_layer,
            activation=activation,
            target_embedding=target_embedding,
            attention_mask=attention_mask,
            target_token_ids=token_ids,
            valid_lengths=valid_lengths,
            answer_labels=answers,
            transmitted_bytes=transmitted_bytes(activation, target_embedding, attention_mask, token_ids, valid_lengths),
            server_round=server_round,
            client_backend="python_loopback",
            client_encode_time_sec=encode_time,
            client_serialize_time_sec=0.0,
            client_round_time_sec=0.0,
            client_rss_mb=current_rss_mb(),
            client_power_w=-1.0,
        )
        metrics = payload.to_metrics()
        metrics["client_serialize_time_sec"] = float(time.perf_counter() - serialize_start)
        metrics["client_round_time_sec"] = float(time.perf_counter() - round_start)
        self.batch_id += 1
        return fl.common.parameters_to_ndarrays(payload.to_parameters()), len(batch), metrics

    def evaluate(self, parameters, config):  # noqa: ANN001
        return 0.0, 0, {}


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg.runtime.seed)
    client = LShapedNumPyClient(cfg=cfg, client_id=args.client_id)
    server_address = cfg.flower.server_address
    if server_address.startswith("0.0.0.0:"):
        server_address = f"127.0.0.1:{server_address.split(':', 1)[1]}"
    fl.client.start_numpy_client(
        server_address=server_address,
        client=client,
        grpc_max_message_length=cfg.flower.grpc_max_message_length,
    )


if __name__ == "__main__":
    main()
