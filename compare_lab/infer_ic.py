"""In-container batched HF inference over an eval jsonl -> predictions jsonl.

Standalone (no compare_lab import -> no qf-lib dependency) so it runs inside the
NVIDIA pytorch container. Loads Qwen3-4B (+ optional LoRA adapter), greedy-decodes
each prompt, parses the final [[[CLASS]]] tag, writes {ticker, as_of, signal, pred, idx}.

    python /work/infer_ic.py --eval /work/eval_mm.jsonl --out /work/preds.jsonl \
        [--adapter /work/adapter_dir] [--batch 16]
"""
from __future__ import annotations

import argparse
import json
import re

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = "Qwen/Qwen3-4B-Instruct-2507"
CLASSES = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")  # idx 0..4
IDX = {c: i for i, c in enumerate(CLASSES)}
_TAG = re.compile(r"\[\[\[\s*(STRONG_SELL|SELL|HOLD|BUY|STRONG_BUY)\s*\]\]\]")


def parse_class(text: str):
    m = _TAG.findall(text)
    return m[-1] if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--adapter", default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--max-new", type=int, default=300)
    ap.add_argument("--max-length", type=int, default=4096,
                    help="prompt truncation cap; raise to 8192 for +summary prompts")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.eval)]
    print(f"[infer] {len(rows)} prompts | adapter={args.adapter or 'BASE'}", flush=True)

    tok = AutoTokenizer.from_pretrained(BASE)
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE, torch_dtype=torch.bfloat16, device_map="cuda")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()
    model.eval()

    # render chat prompts (add_generation_prompt so the model continues as assistant)
    texts = [tok.apply_chat_template(r["prompt"], tokenize=False,
                                     add_generation_prompt=True) for r in rows]

    preds = []
    for i in range(0, len(texts), args.batch):
        chunk = texts[i:i + args.batch]
        enc = tok(chunk, return_tensors="pt", padding=True, truncation=True,
                  max_length=args.max_length).to("cuda")
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=args.max_new,
                                 do_sample=False, pad_token_id=tok.pad_token_id)
        gen = out[:, enc["input_ids"].shape[1]:]
        for j, g in enumerate(gen):
            txt = tok.decode(g, skip_special_tokens=True)
            cls = parse_class(txt)
            r = rows[i + j]
            preds.append({"ticker": r["ticker"], "as_of": r["as_of"],
                          "signal": r["signal"], "pred": cls,
                          "idx": IDX[cls] if cls else None})
        print(f"[infer] {min(i+args.batch,len(texts))}/{len(texts)}", flush=True)

    with open(args.out, "w") as f:
        for p in preds:
            f.write(json.dumps(p) + "\n")
    n_valid = sum(p["idx"] is not None for p in preds)
    print(f"[infer] wrote {len(preds)} preds ({n_valid} valid) -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
