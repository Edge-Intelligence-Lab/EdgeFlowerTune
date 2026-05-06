from __future__ import annotations

import string

import torch


class SimpleCharTokenizer:
    def __init__(self) -> None:
        symbols = sorted(set(string.printable + "\n"))
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.bos_token_id = 2
        self.unk_token_id = 3
        offset = 4
        self.char_to_id = {ch: idx + offset for idx, ch in enumerate(symbols)}
        self.id_to_char = {idx + offset: ch for idx, ch in enumerate(symbols)}
        self.vocab_size = offset + len(symbols)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        ids = [self.char_to_id.get(ch, self.unk_token_id) for ch in text]
        if add_special_tokens:
            return [self.bos_token_id] + ids
        return ids

    def encode_batch(self, texts: list[str], max_seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        assert max_seq_len > 0
        encoded = [self.encode(text, add_special_tokens=True)[:max_seq_len] for text in texts]
        max_len = max(len(ids) for ids in encoded)
        input_ids = torch.full((len(texts), max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((len(texts), max_len), dtype=torch.int32)
        for row, ids in enumerate(encoded):
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1
        return input_ids, attention_mask
