"""LoRA SFT v0 for the 5-class trading signal (TRL + PEFT).

Teaches Qwen3-4B-Instruct-2507 to answer the snapshot prompt with a short,
format-correct thesis whose final [[[CLASS]]] aligns with the deterministic
volatility label. Runs on a single DGX Spark GB10 (BF16, gradient checkpointing).

Designed to run inside the NVIDIA PyTorch container with trl/peft pip-installed:
    pip install -q "trl>=0.12" "peft>=0.13" "datasets>=3.0" "accelerate>=1.0"
    python -m compare_lab.sft.train --data compare_lab/sft/data \
        --out compare_lab/sft/out --epochs 1 [--smoke]
"""
from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Instruct-2507")
    ap.add_argument("--data", default="compare_lab/sft/data")
    ap.add_argument("--out", default="compare_lab/sft/out")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-seq", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--completion-only", action="store_true",
                    help="mask the prompt; train loss only on the assistant turn "
                         "(fixes the v0 HOLD-collapse from full-sequence loss)")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 3 steps to verify GB10 compatibility")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    data = Path(args.data)
    ds = load_dataset("json", data_files={
        "train": str(data / "train.jsonl"),
        "val": str(data / "val.jsonl"),
    })
    if args.smoke:
        ds["train"] = ds["train"].select(range(min(64, len(ds["train"]))))
        ds["val"] = ds["val"].select(range(min(16, len(ds["val"]))))

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="cuda")

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = SFTConfig(
        output_dir=args.out,
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=args.epochs,
        max_steps=3 if args.smoke else -1,
        learning_rate=args.lr,
        bf16=True,
        gradient_checkpointing=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="no" if args.smoke else "epoch",
        max_length=args.max_seq,
        packing=False,
        assistant_only_loss=args.completion_only,
        report_to=[],
    )

    trainer = SFTTrainer(
        model=model, args=cfg, peft_config=peft_cfg,
        train_dataset=ds["train"], eval_dataset=ds["val"],
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[sft] saved adapter to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
