from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from flwr.common import Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from safetensors.numpy import load as load_safetensors_bytes
from safetensors.numpy import save as save_safetensors_bytes


def read_file_bytes(path: str | Path) -> bytes:
    return Path(path).read_bytes()


def write_file_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def adapter_bytes_to_state(data: bytes) -> dict[str, np.ndarray]:
    state = load_safetensors_bytes(data)
    return {name: np.asarray(value).copy() for name, value in state.items()}


def adapter_state_to_bytes(
    state: dict[str, np.ndarray],
    metadata: dict[str, str] | None = None,
) -> bytes:
    contiguous = {name: np.ascontiguousarray(value) for name, value in state.items()}
    return save_safetensors_bytes(contiguous, metadata=metadata)


def adapter_bytes_to_parameters(data: bytes) -> Parameters:
    blob = np.frombuffer(data, dtype=np.uint8).copy()
    return ndarrays_to_parameters([blob])


def parameters_to_adapter_bytes(parameters: Parameters) -> bytes:
    arrays = parameters_to_ndarrays(parameters)
    if len(arrays) != 1:
        raise ValueError(f"Expected exactly one adapter blob array, got {len(arrays)}")
    blob = np.asarray(arrays[0], dtype=np.uint8)
    return blob.tobytes()


def aggregate_adapter_states(
    states: Iterable[dict[str, np.ndarray]],
    weights: Iterable[float],
) -> dict[str, np.ndarray]:
    states = list(states)
    weights = [float(x) for x in weights]
    if not states:
        raise ValueError("aggregate_adapter_states requires at least one state")
    if len(states) != len(weights):
        raise ValueError("states and weights must have the same length")

    total_weight = float(sum(weights))
    if total_weight <= 0.0:
        weights = [1.0 for _ in states]
        total_weight = float(len(states))

    ref_names = list(states[0].keys())
    for state in states[1:]:
        if list(state.keys()) != ref_names:
            raise KeyError("Adapter tensor keys do not match across clients")

    aggregated: dict[str, np.ndarray] = {}
    for name in ref_names:
        ref = np.asarray(states[0][name])
        acc = np.zeros_like(ref, dtype=np.float32)
        for state, weight in zip(states, weights, strict=True):
            acc += np.asarray(state[name], dtype=np.float32) * float(weight)
        aggregated[name] = (acc / total_weight).astype(ref.dtype, copy=False)
    return aggregated
