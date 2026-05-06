import argparse
import csv
import math
import os
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from peft import PeftModel
except ImportError as e:  # pragma: no cover
    raise SystemExit("Please install peft to run this script: pip install peft") from e


def read_mmlu_csv(path: Path) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        try:
            first = next(reader)
        except StopIteration:
            return items

        cols = [c.strip().lower() for c in first]

        def find_col(name: str) -> int:
            try:
                return cols.index(name)
            except ValueError:
                return -1

        idx_question = find_col("question")
        idx_a = find_col("a")
        idx_b = find_col("b")
        idx_c = find_col("c")
        idx_d = find_col("d")
        idx_answer = find_col("answer")
        idx_subject = find_col("subject")
        has_header = (
            idx_question >= 0 and idx_a >= 0 and idx_b >= 0 and idx_c >= 0 and idx_d >= 0 and idx_answer >= 0
        )

        def handle_row(row: List[str]):
            if len(row) < 6:
                return
            subject = row[idx_subject].strip() if (has_header and idx_subject >= 0) else path.stem
            if has_header:
                q_idx, a_idx, b_idx, c_idx, d_idx, ans_idx = (
                    idx_question,
                    idx_a,
                    idx_b,
                    idx_c,
                    idx_d,
                    idx_answer,
                )
            else:
                q_idx, a_idx, b_idx, c_idx, d_idx, ans_idx = (0, 1, 2, 3, 4, 5)
            if max(q_idx, a_idx, b_idx, c_idx, d_idx, ans_idx) >= len(row):
                return
            ans = row[ans_idx].strip().upper()
            items.append(
                {
                    "subject": subject,
                    "question": row[q_idx].strip(),
                    "A": row[a_idx].strip(),
                    "B": row[b_idx].strip(),
                    "C": row[c_idx].strip(),
                    "D": row[d_idx].strip(),
                    "answer": ans[0] if ans else "A",
                }
            )

        if not has_header:
            handle_row(first)
        for row in reader:
            if not row:
                continue
            handle_row(row)
    return items


def build_prompt(q: Dict[str, str], shots: List[Dict[str, str]] | None = None) -> str:
    def one(x: Dict[str, str]) -> str:
        return (
            "Question: "
            + x["question"]
            + "\n"
            + "A. "
            + x["A"]
            + "\n"
            + "B. "
            + x["B"]
            + "\n"
            + "C. "
            + x["C"]
            + "\n"
            + "D. "
            + x["D"]
            + "\n"
            + "Answer: "
        )

    prompt = ""
    if shots:
        for s in shots:
            prompt += one(s)
            prompt += s["answer"]
            prompt += "\n\n"
    prompt += one(q)
    return prompt


def get_choice_ids(tokenizer) -> Dict[str, int]:
    def last_id(text: str) -> int:
        ids = tokenizer.encode(text, add_special_tokens=False)
        if not ids:
            ids = tokenizer.encode(text.strip(), add_special_tokens=False)
        return ids[-1] if ids else -1

    return {
        "A": last_id(" A"),
        "B": last_id(" B"),
        "C": last_id(" C"),
        "D": last_id(" D"),
    }


def evaluate(
    model,
    tokenizer,
    mmlu_root: Path,
    split: str,
    fewshot: int,
    device: torch.device,
    max_subjects: int | None = None,
    max_seq_len: int = 0,
) -> Tuple[float, float, Dict[str, Tuple[int, int]]]:
    split_dir = mmlu_root / split
    subj2items: Dict[str, List[Dict[str, str]]] = {}
    for csv_path in sorted(split_dir.glob("*.csv")):
        items = read_mmlu_csv(csv_path)
        for it in items:
            subj2items.setdefault(it["subject"], []).append(it)
        if max_subjects is not None and len(subj2items) >= max_subjects:
            break

    choice_ids = get_choice_ids(tokenizer)
    if any(v < 0 for v in choice_ids.values()):
        raise RuntimeError("Failed to get choice token ids for A/B/C/D")

    per_subject: Dict[str, Tuple[int, int]] = {}
    total_correct = 0
    total_count = 0

    model.eval()
    for subject, items in subj2items.items():
        if not items:
            continue
        correct = 0
        count = 0
        shots = items[:fewshot] if fewshot > 0 else []
        for q in items:
            shots_ex = [s for s in shots if s is not q] if shots else []
            prompt = build_prompt(q, shots_ex if fewshot > 0 else None)
            inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
            if max_seq_len > 0 and inputs["input_ids"].shape[1] > max_seq_len:
                inputs = {
                    "input_ids": inputs["input_ids"][:, -max_seq_len:],
                    "attention_mask": inputs["attention_mask"][:, -max_seq_len:],
                }
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs)
                logits = out.logits[0, -1]
                logp = torch.log_softmax(logits, dim=-1)
                scores = {k: logp[v].item() for k, v in choice_ids.items()}
                pred = max(scores.items(), key=lambda kv: kv[1])[0]
            if pred == q["answer"]:
                correct += 1
            count += 1
        per_subject[subject] = (correct, count)
        total_correct += correct
        total_count += count

    macro = 0.0
    for correct, count in per_subject.values():
        macro += (float(correct) / float(count)) if count > 0 else 0.0
    if per_subject:
        macro /= float(len(per_subject))
    micro = float(total_correct) / float(total_count) if total_count > 0 else 0.0
    return macro, micro, per_subject


def main() -> None:
    parser = argparse.ArgumentParser(description="Gemma MMLU eval (PT + LoRA)")
    parser.add_argument("--model_dir", type=str, required=True)
    parser.add_argument("--lora_dir", type=str, required=True)
    parser.add_argument("--mmlu_root", type=str, default="data/mmlu/data")
    parser.add_argument("--split", type=str, default="dev", choices=["dev", "test"])
    parser.add_argument("--fewshot", type=int, default=0)
    parser.add_argument("--max_seq_len", type=int, default=0)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--torch_dtype", type=str, default="float32", choices=["float32", "bfloat16", "auto"])
    parser.add_argument("--out", type=str, default="")
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.torch_dtype == "float32":
        torch_dtype = torch.float32
    elif args.torch_dtype == "bfloat16":
        torch_dtype = torch.bfloat16
    else:
        torch_dtype = None

    tok = AutoTokenizer.from_pretrained(args.model_dir, padding_side="right", use_fast=False)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id

    base = AutoModelForCausalLM.from_pretrained(args.model_dir, torch_dtype=torch_dtype)
    base.to(device)
    model = PeftModel.from_pretrained(base, args.lora_dir)
    model.to(device)

    macro, micro, per_subject = evaluate(
        model, tok, Path(args.mmlu_root), args.split, args.fewshot, device, max_seq_len=args.max_seq_len
    )

    print(f"[MMLU] split={args.split} fewshot={args.fewshot}")
    print(f"Macro={macro*100:.2f}% | Micro={micro*100:.2f}%")
    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            for subject, (correct, count) in sorted(per_subject.items()):
                acc = float(correct) / float(count) if count > 0 else 0.0
                f.write(
                    f'{{"task":"mmlu","subject":"{subject}","n":{count},"acc":{acc}}}\n'
                )
            f.write(
                f'{{"task":"mmlu","macro":{macro},"micro":{micro},"split":"{args.split}","fewshot":{args.fewshot}}}\n'
            )


if __name__ == "__main__":
    main()
