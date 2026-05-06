#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from safetensors import safe_open
from transformers import AutoModelForCausalLM, AutoTokenizer


STEP_RE = re.compile(r"lora_step(\d+)\.safetensors$")
PT_STEP_DIR_RE = re.compile(r"step(\d+)$")


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


def iter_checkpoints(
    run_dir: Path, include_final: bool, max_step: int = 0
) -> List[Tuple[str, Optional[int], Path]]:
    items: List[Tuple[str, Optional[int], Path]] = []
    for path in run_dir.glob("lora_step*.safetensors"):
        m = STEP_RE.match(path.name)
        if m:
            step = int(m.group(1))
            if max_step > 0 and step > max_step:
                continue
            items.append((path.name, step, path))
    for path in run_dir.iterdir():
        if not path.is_dir():
            continue
        m = PT_STEP_DIR_RE.fullmatch(path.name)
        if not m:
            continue
        step = int(m.group(1))
        if max_step > 0 and step > max_step:
            continue
        adapter_file = path / "adapter_model.safetensors"
        if adapter_file.exists():
            items.append((path.name, step, path))
    items.sort(key=lambda x: x[1] if x[1] is not None else -1)
    if include_final:
        final_path = run_dir / "lora.safetensors"
        if final_path.exists():
            items.append((final_path.name, None, final_path))
        pt_final_file = run_dir / "adapter_model.safetensors"
        if pt_final_file.exists():
            items.append((pt_final_file.name, None, run_dir))
    return items


def build_model(base_model_dir: str, device: torch.device) -> torch.nn.Module:
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(base_model_dir, torch_dtype=dtype)
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.to(device)
    model.eval()
    return model


def load_cpp_lora_into_hf(model: torch.nn.Module, lora_path: Path) -> None:
    if lora_path.is_dir():
        adapter_path = lora_path / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise ValueError(f"Missing adapter_model.safetensors in {lora_path}")
        mapped: Dict[str, torch.Tensor] = {}
        with safe_open(str(adapter_path), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                mapped[key] = tensor
                if ".lora_A.weight" in key:
                    mapped[key.replace(".lora_A.weight", ".lora_A.default.weight")] = tensor
                if ".lora_B.weight" in key:
                    mapped[key.replace(".lora_B.weight", ".lora_B.default.weight")] = tensor
        missing, unexpected = model.load_state_dict(mapped, strict=False)
        filtered_missing = [k for k in missing if "lora_" in k]
        filtered_unexpected: List[str] = []
        if filtered_missing or filtered_unexpected:
            raise ValueError(
                f"Failed to load {lora_path.name}: missing={filtered_missing[:8]} unexpected={filtered_unexpected[:8]}"
            )
        return

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
            elif target == "v":
                hf_target = "v_proj"
            else:
                raise ValueError(f"Unexpected target in key: {key}")
            if which not in {"lora_A", "lora_B"}:
                raise ValueError(f"Unexpected tensor name in key: {key}")
            hf_key = (
                f"base_model.model.model.layers.{layer_idx}.self_attn.{hf_target}."
                f"{which}.default.weight"
            )
            mapped[hf_key] = f.get_tensor(key).transpose(0, 1).contiguous()

    missing, unexpected = model.load_state_dict(mapped, strict=False)
    filtered_missing = [k for k in missing if "lora_" in k]
    if filtered_missing or unexpected:
        raise ValueError(
            f"Failed to load {lora_path.name}: missing={filtered_missing[:8]} unexpected={unexpected[:8]}"
        )


def read_examples(path: Path) -> List[Dict[str, object]]:
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            ex = json.loads(line)
            if ex.get("label", "") in (None, "", -1):
                continue
            examples.append(ex)
    return examples


def encode_pair(tokenizer, prompt: str, ending: str) -> Tuple[List[int], List[int]]:
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    ending_ids = tokenizer.encode(" " + ending.strip(), add_special_tokens=False)
    return prompt_ids, ending_ids


def prepare_examples(
    tokenizer,
    examples: List[Dict[str, object]],
    max_examples: int = 0,
) -> List[Dict[str, object]]:
    prepared: List[Dict[str, object]] = []
    for ex in examples:
        prompt = f"Context: {' '.join(str(ex['ctx']).strip().split())}\nEnding:"
        candidates: List[Tuple[List[int], List[int]]] = []
        for ending in ex["endings"]:
            candidates.append(encode_pair(tokenizer, prompt, str(ending)))
        prepared.append({"gold": int(ex["label"]), "candidates": candidates})
        if max_examples > 0 and len(prepared) >= max_examples:
            break
    return prepared


@torch.inference_mode()
def score_candidates_batch(
    model: torch.nn.Module,
    device: torch.device,
    prompt_ending_pairs: List[Tuple[List[int], List[int]]],
) -> List[Tuple[float, int]]:
    if not prompt_ending_pairs:
        return []

    sequences: List[List[int]] = []
    prompt_token_counts: List[int] = []
    ending_counts: List[int] = []
    max_len = 0
    for prompt_ids, ending_ids in prompt_ending_pairs:
        ending_count = len(ending_ids)
        if ending_count <= 0:
            raise ValueError("Empty ending ids")
        ids = prompt_ids + ending_ids
        sequences.append(ids)
        prompt_token_counts.append(max(len(prompt_ids) - 1, 0))
        ending_counts.append(ending_count)
        max_len = max(max_len, len(ids))

    pad_id = 0
    batch_size = len(sequences)
    input_ids = torch.full((batch_size, max_len), pad_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long, device=device)
    for i, ids in enumerate(sequences):
        seq = torch.tensor(ids, dtype=torch.long, device=device)
        input_ids[i, : len(ids)] = seq
        attention_mask[i, : len(ids)] = 1

    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
    targets = input_ids[:, 1:]
    token_log_probs = -F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        targets.reshape(-1),
        reduction="none",
    ).view_as(targets)

    scores: List[Tuple[float, int]] = []
    for i in range(batch_size):
        start = prompt_token_counts[i]
        end = start + ending_counts[i]
        selected = token_log_probs[i, start:end]
        scores.append((float(selected.sum().item()), int(ending_counts[i])))
    return scores


@torch.inference_mode()
def evaluate_hellaswag(
    model: torch.nn.Module,
    prepared_examples: List[Dict[str, object]],
    device: torch.device,
    batch_size: int = 128,
) -> Dict[str, float]:
    total = len(prepared_examples)
    correct_raw = 0
    correct_norm = 0
    scores_raw_by_example: List[List[float]] = [[0.0] * 4 for _ in range(total)]
    scores_norm_by_example: List[List[float]] = [[0.0] * 4 for _ in range(total)]

    pending_pairs: List[Tuple[List[int], List[int]]] = []
    pending_meta: List[Tuple[int, int]] = []

    def flush_pending() -> None:
        nonlocal pending_pairs, pending_meta
        if not pending_pairs:
            return
        scores = score_candidates_batch(model, device, pending_pairs)
        for (ex_idx, cand_idx), (raw_score, tok_count) in zip(pending_meta, scores):
            scores_raw_by_example[ex_idx][cand_idx] = raw_score
            scores_norm_by_example[ex_idx][cand_idx] = raw_score / max(tok_count, 1)
        pending_pairs = []
        pending_meta = []

    for ex_idx, ex in enumerate(prepared_examples):
        for cand_idx, pair in enumerate(ex["candidates"]):
            pending_pairs.append(pair)
            pending_meta.append((ex_idx, cand_idx))
            if len(pending_pairs) >= max(batch_size, 1):
                flush_pending()
    flush_pending()

    for ex_idx, ex in enumerate(prepared_examples):
        gold = int(ex["gold"])
        pred_raw = max(range(4), key=lambda i: scores_raw_by_example[ex_idx][i])
        pred_norm = max(range(4), key=lambda i: scores_norm_by_example[ex_idx][i])
        correct_raw += int(pred_raw == gold)
        correct_norm += int(pred_norm == gold)

    return {
        "examples": total,
        "acc_raw": correct_raw / max(total, 1),
        "acc_norm": correct_norm / max(total, 1),
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
            if line:
                rows.append(json.loads(line))
    return rows


def infer_seq_len(run_dir_name: str) -> int:
    m = re.search(r"(?:^|_)s(\d+)(?:_|$)", run_dir_name)
    if not m:
        raise SystemExit(f"Cannot infer seq_len from run dir name: {run_dir_name}")
    return int(m.group(1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen HellaSwag checkpoints from C++ LoRA safetensors.")
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--hellaswag_jsonl", required=True)
    parser.add_argument("--base_model_dir", required=True)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="auto")
    parser.add_argument("--include_final", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max_step", type=int, default=0)
    parser.add_argument("--max_examples", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--output_jsonl", required=True)
    parser.add_argument("--output_csv", required=True)
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    seq_len = infer_seq_len(run_dir.name)
    device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_dir, use_fast=True)
    model = build_model(args.base_model_dir, device)
    examples = read_examples(Path(args.hellaswag_jsonl))
    prepared_examples = prepare_examples(tokenizer, examples, max_examples=args.max_examples)

    output_jsonl = Path(args.output_jsonl)
    output_csv = Path(args.output_csv)
    rows: List[Dict[str, object]] = read_existing_rows(output_jsonl)
    completed = {(str(r["checkpoint"]), str(r["step"])) for r in rows}
    checkpoints = iter_checkpoints(run_dir, include_final=args.include_final, max_step=args.max_step)
    if args.limit > 0:
        checkpoints = checkpoints[: args.limit]

    for ckpt_name, step, ckpt_path in checkpoints:
        step_value = step if step is not None else "final"
        if (ckpt_name, str(step_value)) in completed:
            continue
        load_cpp_lora_into_hf(model, ckpt_path)
        metrics = evaluate_hellaswag(
            model=model,
            prepared_examples=prepared_examples,
            device=device,
            batch_size=args.batch_size,
        )
        row = {
            "run_dir": run_dir.name,
            "seq_len": seq_len,
            "checkpoint": ckpt_name,
            "step": step_value,
            "split": "validation",
            "examples": metrics["examples"],
            "acc_raw": round(metrics["acc_raw"], 6),
            "acc_norm": round(metrics["acc_norm"], 6),
        }
        rows.append(row)
        write_jsonl(output_jsonl, rows)
        write_csv(output_csv, rows)
        print(json.dumps(row, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
