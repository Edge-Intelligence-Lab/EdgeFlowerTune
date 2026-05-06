import argparse
import json
import math
import os
import random
import struct
from typing import Iterable, List

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import LoraConfig, get_peft_model, PeftModel
except ImportError as e:  # pragma: no cover - import-time guard
    raise SystemExit("Please install peft to run this script: pip install peft") from e


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fnv1a_hash_int32(tensor: torch.Tensor) -> int:
    # Expect int32/long tensor on CPU
    data = tensor.to(torch.int32).cpu().numpy().ravel()
    FNV_OFFSET = 1469598103934665603
    FNV_PRIME = 1099511628211
    h = FNV_OFFSET
    for v in data:
        v = int(v) & 0xFFFFFFFF
        for shift in (0, 8, 16, 24):
            h ^= (v >> shift) & 0xFF
            h = (h * FNV_PRIME) & 0xFFFFFFFFFFFFFFFF
    return h


def load_fixed_batch_bin(path: str) -> dict:
    with open(path, "rb") as f:
        header = f.read(8)
        if len(header) != 8:
            raise ValueError("Invalid fixed_batch_bin header")
        b, s = struct.unpack("ii", header)
        n = b * s
        ids = f.read(n * 4)
        attn = f.read(n * 4)
        if len(ids) != n * 4 or len(attn) != n * 4:
            raise ValueError("Invalid fixed_batch_bin payload")
    input_ids = torch.frombuffer(ids, dtype=torch.int32).clone().view(b, s).long()
    attention_mask = torch.frombuffer(attn, dtype=torch.float32).clone().view(b, s).long()
    labels = input_ids.clone()
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


class WikiTextDataset(Dataset):
    """
    Mirrors operators/finetune_ops/data/wikitext2_dataset for alignment:
    - concat lines with EOS between samples
    - fixed-length chunks; drop_last for train, keep tail for eval
    - labels identical to input_ids; HF handles shift internally
    """

    def __init__(
        self,
        path: str,
        tokenizer,
        seq_len: int,
        stride: int = -1,
        eos_token_id: int = 50256,
        data_fraction: float = 1.0,
        insert_eos_between_lines: bool = True,
        drop_last: bool = True,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.stride = seq_len if stride <= 0 else stride
        tokens = self._load_tokens(
            path, tokenizer, eos_token_id, insert_eos_between_lines
        )
        if data_fraction < 1.0:
            keep = max(seq_len + 1, int(len(tokens) * data_fraction))
            tokens = tokens[:keep]
        self.chunks = self._chunk(tokens, drop_last)

    def _load_tokens(
        self,
        path: str,
        tokenizer,
        eos_token_id: int,
        insert_eos_between_lines: bool,
    ) -> List[int]:
        toks: List[int] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.rstrip("\n")
                if line == "":
                    if insert_eos_between_lines:
                        toks.append(eos_token_id)
                    continue
                ids = tokenizer.encode(line, add_special_tokens=False)
                toks.extend(ids)
                if insert_eos_between_lines:
                    toks.append(eos_token_id)
        return toks

    def _chunk(self, tokens: List[int], drop_last: bool) -> List[torch.Tensor]:
        chunks: List[torch.Tensor] = []
        n = len(tokens)
        need = self.seq_len + 1
        for start in range(0, n - need + 1, self.stride):
            window = tokens[start : start + self.seq_len]
            chunks.append(torch.tensor(window, dtype=torch.long))
        if not drop_last and n >= need:
            last_start = (n - need) // self.stride * self.stride
            if last_start + self.seq_len < n and last_start + need > n:
                window = tokens[-self.seq_len :]
                chunks.append(torch.tensor(window, dtype=torch.long))
        return chunks

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int):
        ids = self.chunks[idx]
        attn = torch.ones_like(ids, dtype=torch.long)
        return {"input_ids": ids, "attention_mask": attn, "labels": ids.clone()}


class JsonlMaskedDataset(Dataset):
    """
    JSONL dataset {"ids": [...], "mask": [...]} (masked causal LM, no shift here).
    Matches the C++ JSONL mode used for MMLU finetuning.
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


def resolve_meta_path(token_path: str, meta_arg: str) -> str:
    if meta_arg and os.path.exists(meta_arg):
        return meta_arg
    guess = os.path.join(os.path.dirname(token_path), "meta.json")
    if os.path.exists(guess):
        return guess
    raise FileNotFoundError("meta.json not found; provide --pretokenized_meta")


class PretokenizedStreamDataset(Dataset):
    """
    Reads an int32 token stream + meta.json written by the offline HF pretokenizer.
    Produces fixed windows aligned with the C++ WikiText2Dataset pretokenized mode.
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

        byte_offset = offset * 4
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


def collate_batch(batch: List[dict]) -> dict:
    keys = batch[0].keys()
    return {k: torch.stack([b[k] for b in batch], dim=0) for k in keys}


def cycle(loader: Iterable):
    while True:
        for item in loader:
            yield item


def append_jsonl(path: str, line: str) -> None:
    if not path:
        return
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_scheduler(step: int, total_steps: int, warmup_steps: int, base_lr: float, mode: str) -> float:
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
    parser = argparse.ArgumentParser(description="PyTorch Qwen2.5-0.5B LoRA finetune (alignment)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, default="data/wikitext2/wikitext-2-raw")
    parser.add_argument("--jsonl_train", type=str, default="")
    parser.add_argument("--jsonl_valid", type=str, default="")
    parser.add_argument("--pretokenized_path", type=str, default="")
    parser.add_argument("--pretokenized_meta", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="./qwen_lora_pt")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--steps", type=int, default=0)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--lr_scheduler", type=str, default="cosine", choices=["cosine", "linear"])
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--data_fraction", type=float, default=1.0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--target_mode", type=str, default="qv", choices=["qv", "full"])
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=float, default=16.0)
    parser.add_argument("--lora_dropout", type=float, default=0.0)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--eval_steps", type=int, default=0)
    parser.add_argument("--eval_batches", type=int, default=50)
    parser.add_argument("--save_every", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume_from", type=str, default="")
    parser.add_argument("--eval_out", type=str, default="")
    parser.add_argument("--shuffle", action="store_true", help="Enable DataLoader shuffling")
    parser.add_argument("--no_shuffle", action="store_true", help="Disable DataLoader shuffling")
    parser.add_argument("--dump_first_batch", action="store_true", help="Print first batch tokens and hash")
    parser.add_argument("--dump_first_tokens", type=int, default=16, help="How many tokens to print")
    parser.add_argument("--exit_after_dump", action="store_true", help="Exit after dumping first batch")
    parser.add_argument("--fixed_batch_bin", type=str, default="", help="Use fixed batch from C++ dump and exit")
    parser.add_argument("--fixed_batch_steps", type=int, default=0, help="Train N steps on fixed batch")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.steps <= 0 and args.epochs <= 0:
        args.epochs = 1
    if not args.eval_out:
        args.eval_out = os.path.join(args.output_dir, "eval.jsonl")

    tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right")
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    use_jsonl = bool(args.jsonl_train)
    use_pretok = bool(args.pretokenized_path)
    if use_pretok:
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
    elif use_jsonl:
        train_dataset: Dataset = JsonlMaskedDataset(args.jsonl_train, args.seq_len, tok.pad_token_id)
        eval_dataset: Dataset = (
            JsonlMaskedDataset(args.jsonl_valid, args.seq_len, tok.pad_token_id)
            if args.jsonl_valid
            else train_dataset
        )
    else:
        train_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.train.raw"),
            tok,
            seq_len=args.seq_len,
            stride=-1,
            eos_token_id=tok.eos_token_id,
            data_fraction=args.data_fraction,
            insert_eos_between_lines=True,
            drop_last=True,
        )
        eval_dataset = WikiTextDataset(
            os.path.join(args.data_dir, "wiki.valid.raw"),
            tok,
            seq_len=args.seq_len,
            stride=-1,
            eos_token_id=tok.eos_token_id,
            data_fraction=1.0,
            insert_eos_between_lines=True,
            drop_last=False,
        )

    collate_fn = collate_batch
    shuffle_train = bool(args.shuffle) and not args.no_shuffle
    train_loader = DataLoader(
        train_dataset,
        batch_size=max(1, args.batch),
        shuffle=shuffle_train,
        drop_last=True,
        collate_fn=collate_fn,
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=max(1, args.batch),
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
    )

    base_model = AutoModelForCausalLM.from_pretrained(args.model_dir)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"] if args.target_mode == "full" else ["q_proj", "v_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=target_modules,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    if args.resume_from:
        model = PeftModel.from_pretrained(base_model, args.resume_from, is_trainable=True)
    else:
        model = get_peft_model(base_model, lora_cfg)
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=args.learning_rate, weight_decay=args.weight_decay)

    steps_per_epoch = math.ceil(len(train_loader) / max(1, args.grad_accum))
    total_steps = args.steps if args.steps > 0 else steps_per_epoch * max(1, args.epochs)
    warmup_steps = args.warmup_steps

    print("\n========== PyTorch Qwen2.5-0.5B LoRA Finetune (alignment) ==========")
    if use_pretok:
        print(f"Pretokenized stream: {args.pretokenized_path}")
    elif use_jsonl:
        print(f"JSONL train: {args.jsonl_train}")
    else:
        print(f"Raw data dir: {args.data_dir}")
    if not shuffle_train:
        print("DataLoader shuffle: DISABLED")
    print(f"Train sequences: {len(train_dataset)}, Eval sequences: {len(eval_dataset)}")
    print(f"Total steps: {total_steps}, steps/epoch: {steps_per_epoch}, grad_accum: {args.grad_accum}")
    print(f"LoRA rank/alpha/dropout: {args.lora_r}/{args.lora_alpha}/{args.lora_dropout}")
    print(f"Targets: {','.join(target_modules)}")

    def save_adapter_dir(path: str):
        os.makedirs(path, exist_ok=True)
        model.save_pretrained(path)
        tok.save_pretrained(path)

    model.to(device)

    if args.dump_first_batch:
        batch = next(iter(train_loader))
        input_ids = batch["input_ids"]
        tok_n = max(0, int(args.dump_first_tokens))
        sample0 = input_ids[0].tolist()
        sample1 = input_ids[1].tolist() if input_ids.shape[0] > 1 else []
        print(f"[Dump] first_batch shape={tuple(input_ids.shape)}")
        print(f"[Dump] sample0 first{tok_n}={sample0[:tok_n]}")
        if sample1:
            print(f"[Dump] sample1 first{tok_n}={sample1[:tok_n]}")
        print(f"[Dump] batch_hash_fnv1a64={fnv1a_hash_int32(input_ids)}")
        if args.exit_after_dump:
            return

    if args.fixed_batch_bin:
        fixed = load_fixed_batch_bin(args.fixed_batch_bin)
        fixed = {k: v.to(device) for k, v in fixed.items()}
        model.train()
        if args.fixed_batch_steps > 0:
            total_steps = int(args.fixed_batch_steps)
            for step in range(total_steps):
                optimizer.zero_grad()
                out = model(**fixed)
                loss = out.loss
                loss.backward()
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(trainable, args.max_grad_norm)
                cur_lr = make_scheduler(step, total_steps, args.warmup_steps, args.learning_rate, args.lr_scheduler)
                for group in optimizer.param_groups:
                    group["lr"] = cur_lr
                optimizer.step()
                if (step + 1) % max(1, args.logging_steps) == 0:
                    print(
                        f"[FixedBatch] step {step+1}/{total_steps} lr {cur_lr:.6f} "
                        f"loss {loss.item():.4f}"
                    )
            save_adapter_dir(args.output_dir)
            return
        else:
            out = model(**fixed)
            loss = out.loss.item()
            print(f"[FixedBatch] loss {loss:.6f}")
            return
    model.train()

    ema_loss = None
    token_counter = 0
    train_iter = cycle(train_loader)
    best_valid_ppl = float("inf")

    for step in range(total_steps):
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

        lr_cur = make_scheduler(step, total_steps, warmup_steps, args.learning_rate, args.lr_scheduler)
        for g in optimizer.param_groups:
            g["lr"] = lr_cur
        optimizer.step()

        avg_loss = accum_loss / float(max(1, args.grad_accum))
        token_counter += accum_tokens
        if ema_loss is None:
            ema_loss = avg_loss
        else:
            beta = 0.9
            ema_loss = beta * ema_loss + (1.0 - beta) * avg_loss

        if (step + 1) % max(1, args.logging_steps) == 0:
            ppl = math.exp(avg_loss)
            print(
                f"[Train] step {step + 1}/{total_steps} "
                f"lr {lr_cur:.6f} loss {avg_loss:.4f} ppl {ppl:.2f} "
                f"ema_loss {ema_loss:.4f} tokens {accum_tokens}"
            )

        if args.eval_steps > 0 and (step + 1) % args.eval_steps == 0:
            valid_ppl = evaluate(model, eval_loader, device, args.eval_batches)
            print(
                f"[Eval] step {step + 1}/{total_steps} valid_ppl {valid_ppl:.2f} "
                f"ema_loss {ema_loss:.4f} total_tokens {token_counter}"
            )
            append_jsonl(
                args.eval_out,
                json.dumps(
                    {
                        "step": step + 1,
                        "valid_ppl": valid_ppl,
                        "ema_loss": ema_loss,
                        "total_tokens": token_counter,
                    },
                    ensure_ascii=False,
                ),
            )
            if valid_ppl < best_valid_ppl:
                best_valid_ppl = valid_ppl
                best_dir = os.path.join(args.output_dir, "best")
                save_adapter_dir(best_dir)
                print(f"[Best] updated {best_dir}")

        if args.save_every > 0 and (step + 1) % args.save_every == 0:
            ckpt_dir = os.path.join(args.output_dir, f"step{step + 1}")
            save_adapter_dir(ckpt_dir)
            print(f"[Checkpoint] saved {ckpt_dir}")

    save_adapter_dir(args.output_dir)
    print(f"\n🎉 Qwen LoRA training done. Saved adapter to {args.output_dir}")
    print(f"Total steps {total_steps}, total tokens {token_counter}, final EMA loss {ema_loss:.4f}")


if __name__ == "__main__":
    main()
