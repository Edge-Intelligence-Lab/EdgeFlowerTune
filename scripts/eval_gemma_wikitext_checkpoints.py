#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_RE = re.compile(r"gemma_lora_step(\d+)\.safetensors$")


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested cuda but CUDA is unavailable")
        return torch.device("cuda")
    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise SystemExit("Requested mps but MPS is unavailable")
        return torch.device("mps")
    return torch.device("cpu")


def iter_checkpoints(run_dir: Path, include_final: bool) -> List[Tuple[str, Optional[int], Path]]:
    items: List[Tuple[str, Optional[int], Path]] = []
    for path in run_dir.glob("gemma_lora_step*.safetensors"):
        m = STEP_RE.match(path.name)
        if m:
            items.append((path.name, int(m.group(1)), path))
    items.sort(key=lambda x: x[1] if x[1] is not None else -1)
    if include_final:
        final_path = run_dir / "gemma_lora.safetensors"
        if final_path.exists():
            items.append((final_path.name, None, final_path))
    return items


class WikiTextDataset(Dataset):
    def __init__(self, path: Path, tokenizer: AutoTokenizer, seq_len: int, eos_token_id: int):
        self.seq_len = seq_len
        self.samples = self._build_samples(path, tokenizer, seq_len, eos_token_id)

    @staticmethod
    def _load_tokens(path: Path, tokenizer: AutoTokenizer, eos_token_id: int) -> List[int]:
        tokens: List[int] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    tokens.append(eos_token_id)
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                tokens.extend(ids)
                tokens.append(eos_token_id)
        return tokens

    @classmethod
    def _build_samples(
        cls, path: Path, tokenizer: AutoTokenizer, seq_len: int, eos_token_id: int
    ) -> List[torch.Tensor]:
        tokens = cls._load_tokens(path, tokenizer, eos_token_id)
        samples: List[torch.Tensor] = []
        need = seq_len + 1
        for start in range(0, len(tokens) - need + 1, seq_len):
            window = tokens[start : start + seq_len]
            samples.append(torch.tensor(window, dtype=torch.long))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        ids = self.samples[idx]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids, dtype=torch.long),
            "labels": ids.clone(),
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def build_model(base_model_dir: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float32)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.eval()
    return model


def load_cpp_lora_into_hf(model: torch.nn.Module, lora_path: Path) -> None:
    mapped: Dict[str, torch.Tensor] = {}
    with safe_open(str(lora_path), framework="pt", device="cpu") as f:
        for key in f.keys():
            parts = key.split(".")
            if len(parts) != 5 or parts[0] != "layer" or parts[2] != "attn":
                raise ValueError(f"Unexpected key format: {key}")
            layer_idx = int(parts[1])
            target = parts[3]
            which = parts[4]
            if target == "q":
                hf_target = "q_proj"
            elif target == "k":
                hf_target = "k_proj"
            elif target == "v":
                hf_target = "v_proj"
            elif target == "proj":
                hf_target = "o_proj"
            else:
                raise ValueError(f"Unexpected target in key: {key}")
            if which not in {"lora_A", "lora_B"}:
                raise ValueError(f"Unexpected tensor name in key: {key}")
            hf_key = (
                f"base_model.model.model.layers.{layer_idx}.self_attn.{hf_target}."
                f"{which}.default.weight"
            )
            mapped[hf_key] = f.get_tensor(key)

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    filtered_missing = [k for k in missing if "lora_" in k]
    if filtered_missing or unexpected:
        raise ValueError(
            f"Failed to load {lora_path.name}: missing={filtered_missing[:8]} unexpected={unexpected[:8]}"
        )


@torch.no_grad()
def evaluate_checkpoint(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    total_loss = 0.0
    total_batches = 0
    total_correct = 0
    total_count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        logits = outputs.logits[:, :-1, :]
        labels = batch["labels"][:, 1:]
        preds = logits.argmax(dim=-1)

        total_loss += float(outputs.loss.item())
        total_batches += 1
        total_correct += int((preds == labels).sum().item())
        total_count += int(labels.numel())

    mean_loss = total_loss / max(total_batches, 1)
    return {
        "loss": mean_loss,
        "ppl": math.exp(mean_loss),
        "accuracy": total_correct / max(total_count, 1),
        "batches": total_batches,
        "tokens": total_count,
    }


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_existing_rows(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        return []
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Gemma WikiText checkpoints from C++ LoRA safetensors.")
    parser.add_argument("--run_dir", required=True, help="Run directory containing gemma_lora_step*.safetensors")
    parser.add_argument("--base_model_dir", required=True, help="HF Gemma base model directory")
    parser.add_argument("--data_dir", default="${EDGEFLOWERTUNE_ROOT}/data/wikitext2/wikitext-2-raw")
    parser.add_argument("--split", choices=["valid", "test"], default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--max_batches", type=int, default=0, help="0 means full split")
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all checkpoints")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    seq_match = re.search(r"s(\d+)$", run_dir.name)
    if not seq_match:
        raise SystemExit(f"Cannot infer seq_len from run dir name: {run_dir.name}")
    seq_len = int(seq_match.group(1))

    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, padding_side="right", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    test_file = Path(args.data_dir) / f"wiki.{args.split}.raw"
    dataset = WikiTextDataset(test_file, tokenizer, seq_len=seq_len, eos_token_id=tokenizer.eos_token_id)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=False, collate_fn=collate_batch)
    model = build_model(args.base_model_dir, device)

    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    rows: List[Dict[str, object]] = read_existing_rows(output_jsonl)
    completed = {(str(row["checkpoint"]), str(row["step"])) for row in rows}

    checkpoints = iter_checkpoints(run_dir, include_final=args.include_final)
    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]

    for ckpt_name, step, ckpt_path in checkpoints:
        step_value = step if step is not None else "final"
        if (ckpt_name, str(step_value)) in completed:
            continue
        load_cpp_lora_into_hf(model, ckpt_path)
        metrics = evaluate_checkpoint(
            model,
            loader,
            device=device,
            max_batches=args.max_batches if args.max_batches > 0 else None,
        )
        row = {
            "run_dir": run_dir.name,
            "seq_len": seq_len,
            "checkpoint": ckpt_name,
            "step": step_value,
            "split": args.split,
            "loss": round(metrics["loss"], 6),
            "ppl": round(metrics["ppl"], 6),
            "accuracy": round(metrics["accuracy"], 6),
            "batches": metrics["batches"],
            "tokens": metrics["tokens"],
        }
        rows.append(row)
        completed.add((ckpt_name, str(step_value)))
        print(json.dumps(row, ensure_ascii=False))
        write_jsonl(output_jsonl, rows)
        write_csv(output_csv, rows)


if __name__ == "__main__":
    main()
