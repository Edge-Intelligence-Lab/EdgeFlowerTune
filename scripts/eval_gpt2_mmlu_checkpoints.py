#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from safetensors import safe_open
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model


STEP_RE = re.compile(r"lora_step(\d+)\.safetensors$")


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
    for path in run_dir.glob("lora_step*.safetensors"):
        m = STEP_RE.match(path.name)
        if m:
            items.append((path.name, int(m.group(1)), path))
    items.sort(key=lambda x: x[1] if x[1] is not None else -1)
    if include_final:
        final_path = run_dir / "lora.safetensors"
        if final_path.exists():
            items.append((final_path.name, None, final_path))
    return items


class JsonlMaskedDataset(Dataset):
    def __init__(self, path: Path, seq_len: int, pad_id: int):
        self.samples: List[Dict[str, torch.Tensor]] = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                ids = rec.get("ids", [])
                mask = rec.get("mask", [])
                if not isinstance(ids, list) or not isinstance(mask, list):
                    continue
                if len(ids) != len(mask) or not ids:
                    continue
                ids = ids[:seq_len]
                mask = mask[:seq_len]
                if len(ids) < seq_len:
                    pad_n = seq_len - len(ids)
                    ids = ids + [pad_id] * pad_n
                    mask = mask + [0] * pad_n
                ids_t = torch.tensor(ids, dtype=torch.long)
                mask_t = torch.tensor(mask, dtype=torch.long)
                attn = torch.ones_like(ids_t, dtype=torch.long)
                labels = torch.full_like(ids_t, -100)
                labels = torch.where(mask_t > 0, ids_t, labels)
                self.samples.append(
                    {
                        "input_ids": ids_t,
                        "attention_mask": attn,
                        "labels": labels,
                        "answer_mask": mask_t,
                    }
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return self.samples[idx]


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in batch[0].keys()}


def build_model(base_model_dir: str, device: torch.device) -> torch.nn.Module:
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=torch.float32)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["attn.c_attn", "attn.c_proj"],
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
            tensor = f.get_tensor(key)
            parts = key.split(".")
            if len(parts) != 5 or parts[0] != "layer" or parts[2] != "attn":
                raise ValueError(f"Unexpected key format: {key}")
            layer_idx = int(parts[1])
            target = parts[3]
            which = parts[4]
            if target == "qkv":
                hf_target = "c_attn"
            elif target == "proj":
                hf_target = "c_proj"
            else:
                raise ValueError(f"Unexpected target in key: {key}")
            if which not in {"lora_A", "lora_B"}:
                raise ValueError(f"Unexpected tensor name in key: {key}")
            hf_key = (
                f"base_model.model.transformer.h.{layer_idx}.attn.{hf_target}."
                f"{which}.default.weight"
            )
            mapped[hf_key] = tensor.transpose(0, 1).contiguous()

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
    total_token_correct = 0
    total_token_count = 0
    total_example_correct = 0
    total_example_count = 0

    for batch_idx, batch in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break
        answer_mask = batch.pop("answer_mask")
        batch = {k: v.to(device) for k, v in batch.items()}
        answer_mask = answer_mask.to(device)

        outputs = model(**batch)
        loss = outputs.loss
        preds = outputs.logits[:, :-1, :].argmax(dim=-1)
        target_ids = batch["input_ids"][:, 1:]
        masked_positions = answer_mask[:, 1:] > 0

        token_correct = ((preds == target_ids) & masked_positions).sum().item()
        token_count = masked_positions.sum().item()

        per_example_correct = ((preds == target_ids) | (~masked_positions)).all(dim=1)
        example_correct = per_example_correct.sum().item()
        example_count = target_ids.shape[0]

        total_loss += float(loss.item())
        total_batches += 1
        total_token_correct += int(token_correct)
        total_token_count += int(token_count)
        total_example_correct += int(example_correct)
        total_example_count += int(example_count)

    mean_loss = total_loss / max(total_batches, 1)
    ppl = math.exp(mean_loss)
    token_acc = total_token_correct / max(total_token_count, 1)
    exact_acc = total_example_correct / max(total_example_count, 1)
    return {
        "loss": mean_loss,
        "ppl": ppl,
        "token_accuracy": token_acc,
        "exact_accuracy": exact_acc,
        "batches": total_batches,
        "answer_tokens": total_token_count,
        "examples": total_example_count,
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


def infer_seq_len(run_dir_name: str) -> int:
    m = re.search(r"s(\d+)$", run_dir_name)
    if not m:
        raise SystemExit(f"Cannot infer seq_len from run dir name: {run_dir_name}")
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GPT-2 MMLU checkpoints from C++ LoRA safetensors.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--jsonl_test", required=True)
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Evaluate only the first N checkpoints; 0 means all.")
    parser.add_argument("--max_batches", type=int, default=0, help="Evaluate only the first N batches; 0 means full test set.")
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    seq_len = infer_seq_len(run_dir.name)
    device = resolve_device(args.device)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, padding_side="right", use_fast=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dataset = JsonlMaskedDataset(Path(args.jsonl_test), seq_len=seq_len, pad_id=tokenizer.pad_token_id)
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
        if args.max_batches > 0:
            metrics = evaluate_checkpoint(model, loader, device=device, max_batches=args.max_batches)
        else:
            metrics = evaluate_checkpoint(model, loader, device=device, max_batches=None)
        row = {
            "run_dir": run_dir.name,
            "seq_len": seq_len,
            "checkpoint": ckpt_name,
            "step": step_value,
            "split": "test",
            "loss": round(metrics["loss"], 6),
            "ppl": round(metrics["ppl"], 6),
            "exact_accuracy": round(metrics["exact_accuracy"], 6),
            "token_accuracy": round(metrics["token_accuracy"], 6),
            "batches": metrics["batches"],
            "examples": metrics["examples"],
            "answer_tokens": metrics["answer_tokens"],
        }
        rows.append(row)
        completed.add((ckpt_name, str(step_value)))
        print(json.dumps(row, ensure_ascii=False))
        write_jsonl(output_jsonl, rows)
        write_csv(output_csv, rows)


if __name__ == "__main__":
    main()
