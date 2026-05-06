from __future__ import annotations

from collections import deque

import torch


class NegativeEmbeddingQueue:
    def __init__(self, max_size: int, device: torch.device) -> None:
        assert max_size > 0
        self.max_size = max_size
        self.device = device
        self._items: deque[torch.Tensor] = deque(maxlen=max_size)

    def __len__(self) -> int:
        return len(self._items)

    def push(self, embeddings: torch.Tensor) -> None:
        assert embeddings.ndim == 2
        with torch.no_grad():
            for row in embeddings.detach():
                self._items.append(row.to(self.device))

    def as_tensor(self) -> torch.Tensor | None:
        if not self._items:
            return None
        return torch.stack(list(self._items), dim=0)

