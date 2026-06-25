"""GRPO RL on the SFT-v1 base (paper §5; staged structure/evidence/decision rewards).

Starts from the v1 LoRA (merged into the Qwen3-4B backbone) and trains a fresh
GRPO LoRA that pushes the decision reward while keeping v1's parse/drawdown wins.
The three rewards are kept SEPARATE (`reward_funcs=[...]`, not summed) per the R0
lesson in `rewards.py`; decision is the verifiable core (asymmetric matrix vs the
volatility label, passed per-example via the dataset `label` column).

Runs on DGX Spark GB10 node 2, NVIDIA PyTorch container. HF generation
(`use_vllm=False`) — vLLM+Ray is broken on GB10 (SM 12.1). Run as a SCRIPT,
not `-m` (node 2 lacks qf-lib-harness; `-m` would import `compare_lab/__init__`).

    pip install -q "trl>=0.12" "peft>=0.13" "datasets>=3.0" "accelerate>=1.0"
    pip uninstall -y torchao            # container's pin version-checks vs trl's transformers
    python /work/train.py --data /work/data --base-adapter /work/adapter_v1 \
        --out /work/out [--wandb] [--smoke]
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Run standalone on node 2: rewards.py sits beside this file, not in a package.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rewards import decision_reward, evidence_reward, structure_reward  # noqa: E402

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
LR = 5e-6


def _texts(completions) -> list[str]:
    # conversational dataset → completions[i] = [{"role": "assistant", "content": ...}]
    return [c[0]["content"] for c in completions]


def reward_structure(completions, **kw):
    return [structure_reward(t) for t in _texts(completions)]


def reward_evidence(completions, **kw):
    return [evidence_reward(t) for t in _texts(completions)]


def reward_decision(completions, label, **kw):
    return [decision_reward(t, lbl) for t, lbl in zip(_texts(completions), label)]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="compare_lab/grpo/data")
    ap.add_argument("--base-adapter", default="data/sft_adapter_v1")
    ap.add_argument("--out", default="compare_lab/grpo/out")
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--num-generations", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=1.0,
                    help="generation sampling temp (raise for more GRPO exploration)")
    ap.add_argument("--max-steps", type=int, default=-1)
    ap.add_argument("--wandb", action="store_true")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny subset + 3 steps to verify GB10 compatibility")
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from peft import LoraConfig, PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import GRPOConfig, GRPOTrainer

    data = Path(args.data)
    ds = load_dataset("json", data_files={
        "train": str(data / "train.jsonl"),
        "val": str(data / "val.jsonl"),
    })
    if args.smoke:
        ds["train"] = ds["train"].select(range(min(16, len(ds["train"]))))

    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(
        MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    # GRPO on the v1 base: fold the v1 LoRA into the backbone, then train a fresh
    # GRPO LoRA on top (keeps v1's behaviour as the starting policy).
    model = PeftModel.from_pretrained(base, args.base_adapter).merge_and_unload()

    peft_cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )

    cfg = GRPOConfig(
        output_dir=args.out,
        learning_rate=LR,
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=2 if args.smoke else 4,
        num_generations=args.num_generations,
        temperature=args.temperature,
        num_train_epochs=args.epochs,
        max_steps=3 if args.smoke else args.max_steps,
        max_completion_length=1024,   # prompts (~700 tok) need no cap; TRL 1.6 dropped max_prompt_length
        bf16=True,
        gradient_checkpointing=True,
        use_vllm=False,                  # GB10: vLLM+Ray broken (SM 12.1)
        logging_steps=1,
        save_strategy="epoch",
        report_to=["wandb"] if args.wandb else [],
        run_name="tr1-grpo-v1base",
    )

    trainer = GRPOTrainer(
        model=model, args=cfg, peft_config=peft_cfg,
        reward_funcs=[reward_structure, reward_evidence, reward_decision],
        train_dataset=ds["train"],
        processing_class=tok,
    )
    trainer.train()
    trainer.save_model(args.out)
    print(f"[grpo] saved adapter to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
