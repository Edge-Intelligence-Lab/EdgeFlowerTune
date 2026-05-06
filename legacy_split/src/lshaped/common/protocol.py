from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np
from flwr.common import ndarrays_to_parameters, parameters_to_ndarrays
from flwr.common.typing import Parameters


class TensorSlot(IntEnum):
    ACTIVATION = 0
    TARGET_EMBEDDING = 1
    ATTENTION_MASK = 2
    TARGET_TOKEN_IDS = 3
    VALID_LENGTHS = 4


PAYLOAD_VERSION = 1


@dataclass
class ClientBatchPayload:
    client_id: str
    batch_id: int
    mode: str
    split_layer: int
    activation: np.ndarray
    target_embedding: np.ndarray
    attention_mask: np.ndarray
    target_token_ids: np.ndarray
    valid_lengths: np.ndarray
    answer_labels: list[str]
    transmitted_bytes: int
    retry_count: int = 0
    server_round: int = -1
    client_backend: str = ""
    client_encode_time_sec: float = 0.0
    client_serialize_time_sec: float = 0.0
    client_round_time_sec: float = 0.0
    client_rss_mb: float = -1.0
    client_power_w: float = -1.0

    def validate(self) -> None:
        assert self.split_layer == 0, "Current prototype only supports split_layer=0"
        assert self.activation.ndim == 3, f"activation shape must be [B,S,H], got {self.activation.shape}"
        assert self.target_embedding.ndim == 2, f"target_embedding shape must be [B,H], got {self.target_embedding.shape}"
        assert self.attention_mask.ndim == 2, f"attention_mask shape must be [B,S], got {self.attention_mask.shape}"
        assert self.target_token_ids.ndim == 1, f"target_token_ids shape must be [B], got {self.target_token_ids.shape}"
        assert self.valid_lengths.ndim == 1, f"valid_lengths shape must be [B], got {self.valid_lengths.shape}"

        bsz, seq_len, hidden = self.activation.shape
        assert self.target_embedding.shape == (bsz, hidden)
        assert self.attention_mask.shape == (bsz, seq_len)
        assert self.target_token_ids.shape == (bsz,)
        assert self.valid_lengths.shape == (bsz,)
        assert len(self.answer_labels) == bsz

    def to_parameters(self) -> Parameters:
        self.validate()
        arrays = [
            np.ascontiguousarray(self.activation),
            np.ascontiguousarray(self.target_embedding),
            np.ascontiguousarray(self.attention_mask),
            np.ascontiguousarray(self.target_token_ids),
            np.ascontiguousarray(self.valid_lengths),
        ]
        return ndarrays_to_parameters(arrays)

    def to_metrics(self) -> dict[str, Any]:
        self.validate()
        return {
            "protocol_version": PAYLOAD_VERSION,
            "client_id": self.client_id,
            "batch_id": int(self.batch_id),
            "mode": self.mode,
            "split_layer": int(self.split_layer),
            "activation_shape": "x".join(str(x) for x in self.activation.shape),
            "activation_dtype": str(self.activation.dtype),
            "target_shape": "x".join(str(x) for x in self.target_embedding.shape),
            "target_dtype": str(self.target_embedding.dtype),
            "attention_shape": "x".join(str(x) for x in self.attention_mask.shape),
            "attention_dtype": str(self.attention_mask.dtype),
            "answer_labels": ",".join(self.answer_labels),
            "transmitted_bytes": int(self.transmitted_bytes),
            "retry_count": int(self.retry_count),
            "server_round": int(self.server_round),
            "client_backend": self.client_backend,
            "client_encode_time_sec": float(self.client_encode_time_sec),
            "client_serialize_time_sec": float(self.client_serialize_time_sec),
            "client_round_time_sec": float(self.client_round_time_sec),
            "client_rss_mb": float(self.client_rss_mb),
            "client_power_w": float(self.client_power_w),
        }


def transmitted_bytes(*arrays: np.ndarray) -> int:
    return int(sum(int(arr.nbytes) for arr in arrays))


def payload_from_fit_result(parameters: Parameters, metrics: dict[str, Any]) -> ClientBatchPayload:
    arrays = parameters_to_ndarrays(parameters)
    assert len(arrays) == 5, f"Expected 5 tensors, got {len(arrays)}"
    payload = ClientBatchPayload(
        client_id=str(metrics["client_id"]),
        batch_id=int(metrics["batch_id"]),
        mode=str(metrics["mode"]),
        split_layer=int(metrics["split_layer"]),
        activation=np.asarray(arrays[TensorSlot.ACTIVATION]),
        target_embedding=np.asarray(arrays[TensorSlot.TARGET_EMBEDDING]),
        attention_mask=np.asarray(arrays[TensorSlot.ATTENTION_MASK]),
        target_token_ids=np.asarray(arrays[TensorSlot.TARGET_TOKEN_IDS]).reshape(-1).astype(np.int32),
        valid_lengths=np.asarray(arrays[TensorSlot.VALID_LENGTHS]).reshape(-1).astype(np.int32),
        answer_labels=str(metrics.get("answer_labels", "")).split(",") if metrics.get("answer_labels") else [],
        transmitted_bytes=int(metrics.get("transmitted_bytes", 0)),
        retry_count=int(metrics.get("retry_count", 0)),
        server_round=int(metrics.get("server_round", -1)),
        client_backend=str(metrics.get("client_backend", "")),
        client_encode_time_sec=float(metrics.get("client_encode_time_sec", 0.0)),
        client_serialize_time_sec=float(metrics.get("client_serialize_time_sec", 0.0)),
        client_round_time_sec=float(metrics.get("client_round_time_sec", 0.0)),
        client_rss_mb=float(metrics.get("client_rss_mb", -1.0)),
        client_power_w=float(metrics.get("client_power_w", -1.0)),
    )
    payload.validate()
    return payload
