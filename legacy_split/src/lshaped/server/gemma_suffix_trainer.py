from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForCausalLM
from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig

from lshaped.common.logging_utils import append_csv, append_jsonl
from lshaped.common.protocol import ClientBatchPayload
from lshaped.common.resource_monitor import ResourceMonitor
from lshaped.common.simple_tokenizer import SimpleCharTokenizer
from lshaped.config import AppConfig
from lshaped.server.activation_loss import activation_contrastive_loss, multiple_choice_accuracy
from lshaped.server.negative_queue import NegativeEmbeddingQueue


def _torch_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unsupported dtype: {name}")
    return mapping[name]


@dataclass
class StepOutput:
    loss: float
    loss_exp: float
    contrastive_ppl_proxy: float
    accuracy: float
    round_time_sec: float
    queue_size: int
    transmitted_bytes: int
    rss_mb: float
    gpu_mem_mb: float
    gpu_power_w: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "loss_exp": self.loss_exp,
            "contrastive_ppl_proxy": self.contrastive_ppl_proxy,
            "accuracy": self.accuracy,
            "round_time_sec": self.round_time_sec,
            "queue_size": self.queue_size,
            "transmitted_bytes": self.transmitted_bytes,
            "rss_mb": self.rss_mb,
            "gpu_mem_mb": self.gpu_mem_mb,
            "gpu_power_w": self.gpu_power_w,
        }


class GemmaSuffixTrainer:
    def __init__(self, cfg: AppConfig) -> None:
        self.cfg = cfg
        self.device = torch.device(cfg.model.device if torch.cuda.is_available() else "cpu")
        self.model_dtype = _torch_dtype(cfg.model.dtype)
        self.federated_algorithm = cfg.federated.algorithm.strip().lower()
        self.output_dir = Path(cfg.runtime.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.monitor = ResourceMonitor(str(self.device))

        self.tokenizer, self.model = self._build_model_and_tokenizer()
        self._configure_training_mode()
        self.model.train()

        if self.uses_fedavg_lora() and self.cfg.model.training_mode.strip().lower() != "lora":
            raise ValueError("federated.algorithm=fedavg_lora requires model.training_mode=lora")

        if cfg.model.freeze_input_embeddings:
            for param in self.model.get_input_embeddings().parameters():
                param.requires_grad = False

        self.optimizer = None if self.uses_fedavg_lora() else self._make_optimizer()
        self.queue = NegativeEmbeddingQueue(cfg.loss.queue_size, self.device)
        self.step = 0
        self.answer_token_ids = torch.tensor(
            [self._answer_token_id(letter) for letter in ("A", "B", "C", "D")],
            dtype=torch.long,
            device=self.device,
        )

    def _build_model_and_tokenizer(self):
        if self.cfg.model.model_name_or_path == "__random_gemma__":
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
                max_position_embeddings=self.cfg.dataset.max_seq_len,
                sliding_window=min(64, self.cfg.dataset.max_seq_len),
                pad_token_id=tokenizer.pad_token_id,
                bos_token_id=tokenizer.bos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                layer_types=layer_types,
            )
            model = Gemma3ForCausalLM(config).to(self.device, dtype=self.model_dtype)
            return tokenizer, model

        tokenizer = AutoTokenizer.from_pretrained(self.cfg.model.model_name_or_path)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            self.cfg.model.model_name_or_path,
            torch_dtype=self.model_dtype,
            attn_implementation="eager",
        ).to(self.device)
        return tokenizer, model

    def _configure_training_mode(self) -> None:
        mode = self.cfg.model.training_mode.strip().lower()
        if mode == "full":
            return
        if mode != "lora":
            raise ValueError(f"Unsupported training_mode: {self.cfg.model.training_mode}")

        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.cfg.model.lora_r,
            lora_alpha=self.cfg.model.lora_alpha,
            lora_dropout=self.cfg.model.lora_dropout,
            target_modules=self.cfg.model.lora_target_modules,
            bias="none",
        )
        self.model = get_peft_model(self.model, lora_cfg).to(self.device)

    def uses_fedavg_lora(self) -> bool:
        return self.federated_algorithm == "fedavg_lora"

    def _trainable_named_parameters(self) -> list[tuple[str, torch.nn.Parameter]]:
        return [(name, param) for name, param in self.model.named_parameters() if param.requires_grad]

    def _make_optimizer(self) -> AdamW:
        params = [param for _, param in self._trainable_named_parameters()]
        if not params:
            raise RuntimeError("No trainable parameters found when constructing optimizer")
        return AdamW(
            params,
            lr=self.cfg.model.learning_rate,
            weight_decay=self.cfg.model.weight_decay,
        )

    def export_trainable_state(self) -> dict[str, torch.Tensor]:
        return {
            name: param.detach().cpu().clone()
            for name, param in self._trainable_named_parameters()
        }

    def load_trainable_state(self, state: dict[str, torch.Tensor]) -> None:
        trainable = dict(self._trainable_named_parameters())
        if trainable.keys() != state.keys():
            missing = sorted(set(trainable) - set(state))
            extra = sorted(set(state) - set(trainable))
            raise KeyError(f"Trainable state mismatch. missing={missing} extra={extra}")
        with torch.no_grad():
            for name, param in trainable.items():
                tensor = state[name].to(device=param.device, dtype=param.dtype)
                param.copy_(tensor)

    def aggregate_trainable_states(
        self,
        states: list[dict[str, torch.Tensor]],
        weights: list[float],
    ) -> dict[str, torch.Tensor]:
        if not states:
            raise ValueError("aggregate_trainable_states requires at least one client state")
        if len(states) != len(weights):
            raise ValueError("states and weights must have the same length")

        total_weight = float(sum(weights))
        if total_weight <= 0.0:
            weights = [1.0 for _ in states]
            total_weight = float(len(weights))

        names = list(states[0].keys())
        aggregated: dict[str, torch.Tensor] = {}
        for name in names:
            acc: torch.Tensor | None = None
            ref_dtype = states[0][name].dtype
            for state, weight in zip(states, weights, strict=True):
                tensor = state[name].to(dtype=torch.float32)
                acc = tensor.mul(float(weight)) if acc is None else acc.add(tensor, alpha=float(weight))
            assert acc is not None
            aggregated[name] = acc.div(total_weight).to(dtype=ref_dtype)

        self.load_trainable_state(aggregated)
        return aggregated

    def queue_snapshot(self) -> NegativeEmbeddingQueue:
        snapshot = NegativeEmbeddingQueue(self.cfg.loss.queue_size, self.device)
        negatives = self.queue.as_tensor()
        if negatives is not None:
            snapshot.push(negatives)
        return snapshot

    def push_round_targets(self, payloads: list[ClientBatchPayload]) -> None:
        for payload in payloads:
            target_embedding = torch.from_numpy(payload.target_embedding).to(
                device=self.device,
                dtype=self.model_dtype,
            )
            self.queue.push(target_embedding)

    def _answer_token_id(self, letter: str) -> int:
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

    def _candidate_embeddings(self) -> torch.Tensor:
        embedding_layer = self.model.get_input_embeddings()
        return embedding_layer(self.answer_token_ids)

    def _select_prediction_state(self, hidden_states: torch.Tensor, valid_lengths: torch.Tensor) -> torch.Tensor:
        assert hidden_states.ndim == 3
        assert valid_lengths.ndim == 1
        positions = (valid_lengths - 1).clamp_min(0)
        batch_idx = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        return hidden_states[batch_idx, positions, :]

    def _run_backbone(self, activation: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.model(
            inputs_embeds=activation,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        if outputs.hidden_states is None:
            raise RuntimeError("Model forward did not return hidden_states")
        return outputs.hidden_states[-1]

    def _label_indices(self, payload: ClientBatchPayload) -> torch.Tensor:
        assert payload.answer_labels
        return torch.tensor(
            [ord(x.strip().upper()) - ord("A") for x in payload.answer_labels],
            dtype=torch.long,
            device=self.device,
        )

    def _step_common(
        self,
        payload: ClientBatchPayload,
        train: bool,
        *,
        optimizer: AdamW | None,
        queue: NegativeEmbeddingQueue | None,
        update_queue: bool,
    ) -> StepOutput:
        start = time.perf_counter()
        payload.validate()

        activation = torch.from_numpy(payload.activation).to(device=self.device, dtype=self.model_dtype)
        target_embedding = torch.from_numpy(payload.target_embedding).to(device=self.device, dtype=self.model_dtype)
        attention_mask = torch.from_numpy(payload.attention_mask).to(device=self.device, dtype=torch.long)
        valid_lengths = torch.from_numpy(payload.valid_lengths).to(device=self.device, dtype=torch.long)

        assert activation.shape[:2] == attention_mask.shape
        assert activation.shape[0] == target_embedding.shape[0]
        assert activation.shape[2] == target_embedding.shape[1]

        if train:
            if optimizer is None:
                raise RuntimeError("Training step requires an optimizer")
            optimizer.zero_grad(set_to_none=True)

        last_hidden_state = self._run_backbone(activation=activation, attention_mask=attention_mask)
        query = self._select_prediction_state(last_hidden_state, valid_lengths)
        negatives = queue.as_tensor() if queue is not None else None
        loss, _ = activation_contrastive_loss(
            query=query,
            positive=target_embedding,
            negatives=negatives,
            temperature=self.cfg.loss.temperature,
            use_in_batch_negatives=self.cfg.loss.use_in_batch_negatives,
        )
        labels = self._label_indices(payload)
        accuracy = multiple_choice_accuracy(query, self._candidate_embeddings(), labels)

        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [param for _, param in self._trainable_named_parameters()],
                self.cfg.model.grad_clip_norm,
            )
            optimizer.step()
            if update_queue and queue is not None:
                queue.push(target_embedding)
            self.step += 1

        snap = self.monitor.snapshot()
        elapsed = time.perf_counter() - start
        loss_value = float(loss.detach().cpu().item())
        return StepOutput(
            loss=loss_value,
            loss_exp=float(math.exp(min(20.0, loss_value))),
            contrastive_ppl_proxy=float(math.exp(min(20.0, loss_value))),
            accuracy=accuracy,
            round_time_sec=elapsed,
            queue_size=0 if queue is None else len(queue),
            transmitted_bytes=payload.transmitted_bytes,
            rss_mb=snap.rss_mb,
            gpu_mem_mb=snap.gpu_mem_mb,
            gpu_power_w=-1.0 if snap.gpu_power_w is None else float(snap.gpu_power_w),
        )

    def train_batch(self, payload: ClientBatchPayload) -> StepOutput:
        return self._step_common(
            payload,
            train=True,
            optimizer=self.optimizer,
            queue=self.queue,
            update_queue=True,
        )

    def train_batch_fedavg_lora(
        self,
        payload: ClientBatchPayload,
        global_state: dict[str, torch.Tensor],
        queue_snapshot: NegativeEmbeddingQueue | None,
    ) -> tuple[dict[str, torch.Tensor], StepOutput]:
        if not self.uses_fedavg_lora():
            raise RuntimeError("train_batch_fedavg_lora called while federated.algorithm is not fedavg_lora")
        self.load_trainable_state(global_state)
        local_optimizer = self._make_optimizer()
        metrics: StepOutput | None = None
        for _ in range(max(1, int(self.cfg.federated.local_steps))):
            metrics = self._step_common(
                payload,
                train=True,
                optimizer=local_optimizer,
                queue=queue_snapshot,
                update_queue=False,
            )
        assert metrics is not None
        return self.export_trainable_state(), metrics

    def eval_batch(self, payload: ClientBatchPayload) -> StepOutput:
        with torch.no_grad():
            return self._step_common(
                payload,
                train=False,
                optimizer=None,
                queue=self.queue,
                update_queue=False,
            )

    def save_checkpoint(self, round_id: int, include_optimizer: bool = True) -> None:
        ckpt_dir = self.output_dir / "checkpoints"
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        state = {
            "round": round_id,
            "step": self.step,
            "federated_algorithm": self.cfg.federated.algorithm,
            "model": self.model.state_dict(),
        }
        if include_optimizer and self.optimizer is not None:
            state["optimizer"] = self.optimizer.state_dict()
        torch.save(state, ckpt_dir / f"round_{round_id:06d}.pt")
        if self.cfg.model.training_mode.strip().lower() == "lora" and hasattr(self.model, "save_pretrained"):
            adapter_dir = ckpt_dir / f"round_{round_id:06d}_adapter"
            self.model.save_pretrained(adapter_dir)

    def log_step(self, round_id: int, client_id: str, mode: str, payload: ClientBatchPayload, metrics: StepOutput) -> None:
        row = {
            "round": round_id,
            "client_id": client_id,
            "mode": mode,
            "federated_algorithm": self.cfg.federated.algorithm,
            "fedavg_local_steps": max(1, int(getattr(self.cfg.federated, "local_steps", 1))),
            "server_round_from_client": payload.server_round,
            "client_backend": payload.client_backend,
            "client_encode_time_sec": payload.client_encode_time_sec,
            "client_serialize_time_sec": payload.client_serialize_time_sec,
            "client_round_time_sec": payload.client_round_time_sec,
            "client_rss_mb": payload.client_rss_mb,
            "client_power_w": payload.client_power_w,
            **metrics.as_dict(),
        }
        append_jsonl(self.output_dir / "metrics.jsonl", row)
        append_csv(self.output_dir / "metrics.csv", row)
