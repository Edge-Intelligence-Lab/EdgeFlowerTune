#!/usr/bin/env python3
"""
Generate PyTorch/HuggingFace GPT-2 gold standard logits (for C++ forward alignment)
"""

import torch
import json
import os
from pathlib import Path
from transformers import GPT2TokenizerFast, GPT2LMHeadModel

def main():
    repo_root = Path(__file__).resolve().parents[3]
    model_dir = os.environ.get("GPT2_PRETRAINED_DIR", str(repo_root / "gpt2_lora_finetune" / "pretrained" / "gpt2"))
    output_path = os.environ.get("PT_GOLD_LOGITS", str(repo_root / "operators" / "finetune_ops" / "graph" / "pt_last_logits.json"))

    # Load pretrained model (disable dropout)
    tok = GPT2TokenizerFast.from_pretrained(model_dir)
    model = GPT2LMHeadModel.from_pretrained(model_dir)
    model.eval()
    
    # Fixed input
    text = "Hello, world!\n"
    x = tok(text, return_tensors="pt")
    
    print(f"Input text: {repr(text)}")
    print(f"Input IDs: {x['input_ids'].tolist()}")
    print(f"Attention mask: {x['attention_mask'].tolist()}")
    
    # Forward (dropout disabled)
    with torch.no_grad():
        logits = model(**x).logits  # [1, S, V]
    
    # Save last token's logits
    last_logits = logits[0, -1].float().cpu()  # [V]
    
    # Top-5
    topv, topi = torch.topk(last_logits, 5)
    print(f"\nPyTorch top-5 IDs:  {topi.tolist()}")
    print(f"PyTorch top-5 vals: {[f'{v:.6f}' for v in topv.tolist()]}")
    
    # Save as JSON (for C++ reading)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(last_logits.tolist(), f)
    
    print(f"\nSaved PyTorch logits to: {output_path}")
    print(f"Logits shape: {last_logits.shape}, dtype: {last_logits.dtype}")
    print(f"Argmax: {last_logits.argmax().item()}")

if __name__ == "__main__":
    main()
