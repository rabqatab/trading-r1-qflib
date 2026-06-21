# SFT v0 — volatility-label alignment (sub-project 2, Phase 0→1 kickoff)

Teaches `Qwen3-4B-Instruct-2507` to answer the snapshot prompt with a
format-correct thesis whose final `[[[CLASS]]]` aligns with the deterministic
volatility label (`compare_lab/labeling.py`, paper Algorithm S1).

## Pipeline

1. **Labels** — `compare_lab/labeling.py`, forward returns (`shift(-τ)`),
   horizons {3,7,15}, blended, cut at self-quantiles {0.03,0.15,0.53,0.85}.
   Distribution matches paper Table 2 (3/12/38/32/15%). Unit-tested.
2. **Dataset** — `build_dataset.py`: snapshot prompt → templated, value-grounded
   thesis. **Trained only on pre-2024 data** (`2017-01..2023-12`) so the
   2024-2026 qf-lib OOS comparison stays uncontaminated. 4,224 examples
   (3,772 train / 452 val).
3. **Train** — `train.py`: TRL `SFTTrainer` + PEFT LoRA (r=16), BF16, gradient
   checkpointing.

## Running on DGX Spark (GB10) — gotchas learned

Training runs in the NVIDIA PyTorch container on **node 2** (node 1 holds the
vLLM eval server; heavy-job cap = 1 per node):

```bash
# 1. sync code+data to node 2 (node 2 cannot read alphabridge's NFS home)
rsync -az -e "ssh -i ~/.ssh/id_ed25519" compare_lab/sft/ \
  nvidia@192.168.200.13:/home/nvidia/tr1_sft/

# 2. submit via sparkq (node 2)
sparkq submit 'docker run --rm --gpus all --ipc=host --ulimit memlock=-1 \
  --ulimit stack=67108864 -e NVIDIA_DISABLE_REQUIRE=1 -e HF_HOME=/hf \
  -v /mnt/nfs/ssd1/huggingface_cache:/hf -v /home/nvidia/tr1_sft:/work -w /work \
  nvcr.io/nvidia/pytorch:25.09-py3 bash -c "pip install -q trl>=0.12 peft>=0.13 \
  datasets>=3.0 accelerate>=1.0 && pip uninstall -y torchao && \
  python /work/train.py --data /work/data --out /work/out"' \
  --node 2 --gpu-mem 24G --cpu-mem 24G --tag tr1-sft --max-runtime 2h
```

Three GB10/container gotchas, each cost one smoke iteration:

| Symptom | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: compare_lab` | node 2 can't read alphabridge's NFS home; `-m pkg.mod` also runs `compare_lab/__init__` (needs qf-lib-harness) | rsync to `/home/nvidia/...`; run `train.py` as a **script**, not `-m` |
| `SFTConfig got unexpected kwarg max_seq_length` | TRL ≥0.12 renamed it | use `max_length` |
| `incompatible torchao 0.13 (need >0.16)` | `pip install trl` upgrades transformers, which version-checks the container's pinned torchao | `pip uninstall -y torchao` (unused on the bf16 LoRA path) |

The backbone (`Qwen3-4B-Instruct-2507`, BF16) **loads and trains fine on GB10** —
the spec's Phase-0 backbone-compatibility gate is cleared for the Qwen3-4B path.

## Status

v0 = templated rationale (teaches format + decision↔label alignment). Phase 1
replaces the template with a real teacher (Qwen3-32B) distillation. Evaluate a
trained adapter by serving it (vLLM + LoRA) and running
`run_comparison --llm` against the same backtest.
