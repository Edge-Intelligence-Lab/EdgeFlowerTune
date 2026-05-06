import argparse
import json
import math
import os
import random
from typing import Iterable, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, PeftModel, get_peft_model
except ImportError as e:  # pragma: no cover - import-time guard
    raise SystemExit("Please install peft to run this script: pip install peft") from e


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    if device_arg == "cuda":
        if not torch.cuda.is_available():
            raise SystemExit("Requested --device cuda but CUDA is not available")
        return torch.device("cuda")
    if device_arg == "mps":
        if not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available():
            raise SystemExit("Requested --device mps but MPS is not available")
        return torch.device("mps")
    return torch.device("cpu")


class WikiTextDataset(Dataset):
    """
    Mirrors the C++ WikiText2Dataset: concat lines with EOS, chunk into fixed windows.
    Labels equal input_ids; loss does the shift internally.
    """

    def __init__(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        seq_len: int,
        eos_token_id: int,
        data_fraction: float = 1.0,
        insert_eos_between_lines: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        tokens = self._load(path, tokenizer, eos_token_id, insert_eos_between_lines)
        if data_fraction < 1.0:
            keep = max(seq_len + 1, int(len(tokens) * data_fraction))
            tokens = tokens[:keep]
        self.chunks = self._chunk(tokens, drop_last, eos_token_id)

    def _load(
        self,
        path: str,
        tokenizer: AutoTokenizer,
        eos_token_id: int,
        insert_eos_between_lines: bool,
    ) -> List[int]:
        tokens: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if insert_eos_between_lines:
                        tokens.append(eos_token_id)
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                tokens.extend(ids)
                if insert_eos_between_lines:
                    tokens.append(eos_token_id)
        return tokens

    def _chunk(self, tokens: List[int], drop_last: bool, pad_id: int) -> List[torch.Tensor]:
        # Align with C++ version: need seq_len+1 tokens available to form a valid chunk
        # C++ uses: for s in range(0, N - (S+1) + 1, stride) where need = S+1
        # HuggingFace does the shift internally (logits[:-1] vs labels[1:])
        chunks: List[torch.Tensor] = []
        n = len(tokens)
        need = self.seq_len + 1  # Align with C++: need seq_len+1 tokens available
        for start in range(0, n - need + 1, self.seq_len):
            window = tokens[start : start + self.seq_len]
            chunks.append(torch.tensor(window, dtype=torch.long))
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        ids = self.chunks[idx]
        attn = torch.ones_like(ids, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}


def collate_batch(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


class JsonlMaskedDataset(Dataset):
    """
    JSONL dataset {"ids": [...], "mask": [...]} with labels masked in-place.
    Aligns with C++ JSONL mode used for MMLU (no extra shift done here).
    """

    def __init__(self, path: str, seq_len: int, pad_id: int):
        self.samples = []
        self.seq_len = seq_len
        self.pad_id = pad_id
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                try:
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
                    attn = torch.ones_like(ids_t, dtype=torch.long)
                    labels = torch.full_like(ids_t, -100)
                    mask_t = torch.tensor(mask, dtype=torch.long)
                    labels = torch.where(mask_t > 0, ids_t, labels)
                    self.samples.append({"input_ids": ids_t, "attention_mask": attn, "labels": labels})
                except Exception:
                    continue

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        return self.samples[idx]


def load_fixed_batch(batch_dir: str) -> dict:
    def load(name: str) -> np.ndarray:
        path = os.path.join(batch_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing {name} in {batch_dir}")
        return np.load(path)

    input_ids = load("input_ids.npy").astype(np.int64)
    attention = load("attention_mask.npy")
    if attention.dtype != np.int64:
        attention = attention.astype(np.int64)
    labels = load("labels.npy").astype(np.int64)
    return {
        "input_ids": torch.from_numpy(input_ids),
        "attention_mask": torch.from_numpy(attention),
        "labels": torch.from_numpy(labels),
    }


def resolve_meta_path(token_path: str, meta_arg: str) -> str:
    if meta_arg and os.path.exists(meta_arg):
        return meta_arg
    guess = os.path.join(os.path.dirname(token_path), "meta.json")
    if os.path.exists(guess):
        return guess
    raise FileNotFoundError("meta.json not found; provide --pretokenized_meta")


class PretokenizedStreamDataset(Dataset):
    """
    Reads an int32 token stream + meta.json (C++ pretokenized format).
    Produces fixed windows aligned with C++ WikiText2Dataset.
    """

    def __init__(
        self,
        token_path: str,
        meta_path: str,
        split: str,
        seq_len: int,
        data_fraction: float = 1.0,
        drop_last: bool = True,
    ):
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        splits = meta.get("splits", {})
        if split not in splits and split == "valid" and "validation" in splits:
            split = "validation"
        if split not in splits:
            raise ValueError(f"split '{split}' not found in meta.json")
        info = splits[split]
        offset = int(info.get("offset", 0))
        length = int(info.get("length", 0))
        if length <= 0:
            raise ValueError(f"split '{split}' has invalid length")

        use_len = length
        if data_fraction < 1.0:
            min_tokens = seq_len + 1
            limited = int(length * data_fraction)
            use_len = max(min_tokens, min(length, limited))

        byte_offset = offset * 4  # int32
        tokens = np.memmap(token_path, dtype=np.int32, mode="r", offset=byte_offset, shape=(use_len,))

        self.seq_len = seq_len
        self.tokens = tokens
        self.starts = self._build_starts(len(tokens), drop_last)

    def _build_starts(self, n: int, drop_last: bool) -> List[int]:
        starts: List[int] = []
        need = self.seq_len + 1
        for s in range(0, n - need + 1, self.seq_len):
            starts.append(s)
        if not drop_last:
            if not starts or starts[-1] + need < n:
                s = max(0, n - need)
                if not starts or starts[-1] != s:
                    starts.append(s)
        return starts

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, idx: int):
        start = self.starts[idx]
        window = self.tokens[start : start + self.seq_len]
        ids = torch.from_numpy(np.asarray(window, dtype=np.int64))
        attn = torch.ones_like(ids, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}


def cycle(loader: Iterable):
    while True:
        for item in loader:
            yield item


def build_target_modules(target_mode: str, override: str) -> List[str]:
    if override:
        return [m.strip() for m in override.split(",") if m.strip()]
    if target_mode == "attn":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if target_mode == "light":
        return ["q_proj", "v_proj"]
    return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def make_scheduler(step: int, total_steps: int, warmup_steps: int, base_lr: float, mode: str) -> float:
    # Align with C++ Gemma: step is 1-indexed in C++, so we use step+1 here
    # C++ uses: if (step <= warmup_steps) return lr * step / warmup_steps
    # PyTorch step is 0-indexed, so step+1 corresponds to C++ step
    step_1indexed = step + 1
    if warmup_steps > 0 and step_1indexed <= warmup_steps:
        return base_lr * float(step_1indexed) / float(max(1, warmup_steps))
    remain = max(1, total_steps - warmup_steps)
    progress = float(step_1indexed - warmup_steps) / float(remain)
    progress = min(max(progress, 0.0), 1.0)
    if mode == "cosine":
        return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    return base_lr * (1.0 - progress)


def evaluate(model, dataloader: DataLoader, device: torch.device, max_batches: int) -> float:
    model.eval()
    losses: List[float] = []
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= max_batches:
                break
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            losses.append(out.loss.item())
    model.train()
    if not losses:
        return float("inf")
    return math.exp(sum(losses) / len(losses))


def main():
    parser = argparse.ArgumentParser(description="PyTorch Gemma LoRA finetune (alignment build)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--jsonl_train", type=str, default="")
    parser.add_argument("--jsonl_valid", type=str, default="")
    parser.add_argument("--pretokenized_path", type=str, default="")
    parser.add_argument("--pretokenized_meta", type=str, default="")
    parser.add_argument("--fixed_batch_dir", type=str, default="")
    parser.add_argument("--fixed_batch_steps", type=int, default=0, help="Train N steps on fixed batch")
    parser.add_argument("--output_dir", type=str, default="./gemma_lora_pt")
    parser.add_argument(
        "--torch_dtype",
        type=str,
        default="float32",
        choices=["float32", "bfloat16", "auto"],
        help="torch dtype for model weights (use float32 to mirror C++ alignment)",
    )
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler", type=str, default="linear", choices=["linear", "cosine"])
    parser.add_argument("--target_mode", type=str, default="full", choices=["full", "attn", "light"])
    parser.add_argument("--lora_targets", type=str, default="")
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=32.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Execution device. Default is CPU for alignment; auto prefers CUDA, then CPU.",
    )
    parser.add_argument(
        "--use_fast_tokenizer",
        action="store_true",
        help="Use Hugging Face fast tokenizer backend (useful when model dir only has tokenizer.json).",
    )
    parser.set_defaults(shuffle=False)
    parser.add_argument("--shuffle", dest="shuffle", action="store_true", help="Enable DataLoader shuffling")
    parser.add_argument(
        "--no_shuffle",
        dest="shuffle",
        action="store_false",
        help="Disable DataLoader shuffling (default for alignment)",
    )
    args = parser.parse_args()

    set_seed(args.seed)
    device = resolve_device(args.device)
    if args.torch_dtype == "float32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None  # let HF decide

    os.makedirs(args.output_dir, exist_ok=True)
    best_dir = os.path.join(args.output_dir, "best")
    best_train_loss = float("inf")
    best_eval_ppl = float("inf")

    tok = AutoTokenizer.from_pretrained(
        args.model_dir,
        padding_side="right",
        use_fast=args.use_fast_tokenizer,
    )
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    use_fixed = bool(args.fixed_batch_dir)
    use_jsonl = bool(args.jsonl_train)
    use_pretok = bool(args.pretokenized_path)
    if use_fixed:
        train_dataset = None
        eval_dataset = None
        train_loader = None
        eval_loader = None
    elif use_jsonl:
        train_dataset: Dataset = JsonlMaskedDataset(args.jsonl_train, args.seq_len, tok.pad_token_id)
        eval_dataset: Dataset = (
            JsonlMaskedDataset(args.jsonl_valid, args.seq_len, tok.pad_token_id)
            if args.jsonl_valid
            else train_dataset
        )
    elif use_pretok:
        meta_path = resolve_meta_path(args.pretokenized_path, args.pretokenized_meta)
        train_dataset = PretokenizedStreamDataset(
            args.pretokenized_path,
            meta_path,
            split="train",
            seq_len=args.seq_len,
            data_fraction=args.data_fraction,
            drop_last=True,
        )
        eval_dataset = PretokenizedStreamDataset(
            args.pretokenized_path,
            meta_path,
            split="valid",
            seq_len=args.seq_len,
            data_fraction=1.0,
            drop_last=False,
        )
    else:
        train_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.train.raw"),
            tok,
            seq_len=args.seq_len,
            eos_token_id=tok.eos_token_id,
            data_fraction=args.data_fraction,
            insert_eos_between_lines=True,
            drop_last=True,
        )
        eval_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.valid.raw"),
            tok,
            seq_len=args.seq_len,
            eos_token_id=tok.eos_token_id,
            data_fraction=1.0,
            insert_eos_between_lines=True,
            drop_last=False,
        )

    if not use_fixed:
        train_loader = DataLoader(
            train_dataset,
            batch_size=max(1, args.batch),
            shuffle=args.shuffle,
            drop_last=True,
            collate_fn=collate_batch,
        )
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=max(1, args.batch),
            shuffle=False,
            drop_last=False,
            collate_fn=collate_batch,
        )

    model = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch_dtype)
    target_modules = build_target_modules(args.target_mode, args.lora_targets)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    if use_fixed:
        total_steps = 1
        warmup_steps = 0
    else:
        micro_per_epoch = math.ceil(len(train_loader))
        total_updates = (
            math.ceil(micro_per_epoch / max(1, args.grad_accum)) * args.epochs
        )
        if args.max_steps and args.max_steps > 0:
            total_steps = args.max_steps
        else:
            total_steps = total_updates
        warmup_steps = int(total_steps * args.warmup_ratio)

    print("\n========== PyTorch Gemma LoRA Finetune (alignment) ==========")
    print(f"Device: {device}")
    if not use_fixed:
        print(f"Train sequences: {len(train_dataset)}, Eval sequences: {len(eval_dataset)}")
    else:
        print(f"Fixed batch dir: {args.fixed_batch_dir}")
    print(f"Total steps: {total_steps}, grad_accum: {args.grad_accum}, warmup_steps: {warmup_steps}")
    print(f"LoRA r/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Targets: {','.join(target_modules)}")
    if not args.shuffle:
        print("DataLoader shuffle: DISABLED")

    model.to(device)
    model.train()

    if use_fixed:
        batch = load_fixed_batch(args.fixed_batch_dir)
        batch = {k: v.to(device) for k, v in batch.items()}
        if args.fixed_batch_steps and args.fixed_batch_steps > 0:
            total_steps = int(args.fixed_batch_steps)
            warmup_steps = int(total_steps * args.warmup_ratio)
            model.train()
            for step in range(total_steps):
                optimizer.zero_grad()
                out = model(**batch)
                loss = out.loss
                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                cur_lr = make_scheduler(step, total_steps, warmup_steps, args.learning_rate, args.lr_scheduler)
                for group in optimizer.param_groups:
                    group["lr"] = cur_lr
                optimizer.step()
                if (step + 1) % max(1, args.logging_steps) == 0:
                    print(
                        f"[FixedBatch] step {step+1}/{total_steps} "
                        f"lr {cur_lr:.6f} loss {loss.item():.4f}"
                    )
            return
        model.eval()
        with torch.no_grad():
            out = model(**batch)
            print(f"[FixedBatch] loss {out.loss.item():.6f}")
        return

    global_step = 0
    ema_loss = None
    token_counter = 0
    train_iter = cycle(train_loader)

    while global_step < total_steps:
        accum_loss = 0.0
        accum_tokens = 0
        optimizer.zero_grad()
        for _ in range(max(1, args.grad_accum)):
            batch = next(train_iter)
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss
            (loss / max(1, args.grad_accum)).backward()
            accum_loss += loss.item()
            accum_tokens += int(batch["attention_mask"].sum().item())

        if args.max_grad_norm and args.max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)

        lr_cur = make_scheduler(global_step, total_steps, warmup_steps, args.learning_rate, args.lr_scheduler)
        for g in optimizer.param_groups:
            g["lr"] = lr_cur
        optimizer.step()

        global_step += 1
        token_counter += accum_tokens
        avg_loss = accum_loss / float(max(1, args.grad_accum))
        if ema_loss is None:
            ema_loss = avg_loss
        else:
            beta = 0.9
            ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss

        if global_step % max(1, args.logging_steps) == 0:
            ppl = math.exp(avg_loss)
            print(
                f"[Train] step {global_step}/{total_steps} "
                f"lr {lr_cur:.6f} loss {avg_loss:.4f} ppl {ppl:.2f} tokens {accum_tokens}"
            )
            if args.eval_steps == 0 and avg_loss < best_train_loss:
                best_train_loss = avg_loss
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tok.save_pretrained(best_dir)
                print(f"[Best] step {global_step} train_loss {best_train_loss:.4f} saved {best_dir}")

        if args.eval_steps > 0 and global_step % args.eval_steps == 0:
            valid_ppl = evaluate(model, eval_loader, device, args.eval_batches)
            print(
                f"[Eval] step {global_step}/{total_steps} valid_ppl {valid_ppl:.2f} "
                f"ema_loss {ema_loss:.4f} total_tokens {token_counter}"
            )
            if valid_ppl < best_eval_ppl:
                best_eval_ppl = valid_ppl
                os.makedirs(best_dir, exist_ok=True)
                model.save_pretrained(best_dir)
                tok.save_pretrained(best_dir)
                print(f"[Best] step {global_step} valid_ppl {best_eval_ppl:.2f} saved {best_dir}")

        if args.save_every > 0 and global_step % args.save_every == 0:
            ckpt_dir = f"{args.output_dir}_step{global_step}"
            os.makedirs(ckpt_dir, exist_ok=True)
            model.save_pretrained(ckpt_dir)
            tok.save_pretrained(ckpt_dir)
            print(f"[Checkpoint] saved {ckpt_dir}")

    model.save_pretrained(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print(f"\n🎉 Gemma LoRA training done. Saved adapter to {args.output_dir}")
    print(f"Total steps {global_step}, total tokens {token_counter}, final EMA loss {ema_loss:.4f}")


if __name__ == "__main__":
    main()
