# Lever 3a — deeper GRPO on the v1 base (vLLM deep RL)

> **Date**: 2026-06-26
> **Status**: design approved, execute directly (no new code — config + run + eval).
> **Context**: diagnosis showed the all-SELL/all-StrongBuy collapses are specific to
> the tiny-data multimodal SFT (290 ex / 1 yr); **price-only v1 is already diverse
> and is the keeper** (2025-H1 decision dist 38/31/28/1, vs SFT-mm's 97 % SELL). And
> we verified GRPO **vLLM colocate rollout works on GB10** (~3× faster than HF, higher
> entropy). So lever 3 part **a**: redo GRPO on the non-collapsing v1 base, *deeper*,
> using the new tools — fixing the prior GRPO's defects (10 % template-echo, lost
> defense, weak 1-epoch signal). Part **b** (fix the multimodal SFT collapse) is a
> separate later sub-cycle.

## Goal & success criteria

Improve on the prior price-only GRPO (`output_grpo/`: CR +37 %, Sharpe 0.58, but
MDD 21.6 %, NO_TAG 10 %) without its defects.

1. **NO_TAG ≈ 0** — the −2.5 invalid-decision guardrail kills the 10 % template-echo.
2. **Decision distribution stays diverse** — no collapse (target a v1-like spread, not
   all-SELL/all-StrongBuy).
3. **2024–2026 (14-ticker):** match-or-beat the prior GRPO Sharpe (0.58) without the
   echo and without blowing out drawdown.
4. **2025-H1 (12-ticker):** hold up better than the multimodal collapse models
   (defensive, not degenerate).

A null result (deeper RL doesn't beat the shallow one) is still informative — it
isolates "RL depth wasn't the lever either," pointing back to data/reward design.

## Base & data

- **Base**: merge the v1 LoRA (`data/sft_adapter_v1`) into Qwen3-4B, train a fresh
  GRPO LoRA on top (same as the prior price-only GRPO).
- **Data**: reuse the existing price-only GRPO dataset (`compare_lab/grpo/build_dataset`,
  pre-2024 balanced, 300 prompts, leak-safe). No multimodal.

## GRPO config (the "deeper")

All knobs already exist in `compare_lab/grpo/train.py` (no new code):

| knob | value | rationale |
|---|---|---|
| `--use-vllm` | on (colocate, TP=1, `vllm_max_model_length=4096`) | 3× faster + higher entropy (verified) |
| `--num-generations` | 12 (from 8) | larger groups → better advantage estimates; vLLM makes it affordable |
| `--epochs` | 2 (from 1) | deeper than the weak 1-epoch prior run |
| `--temperature` | 1.0 | exploration |
| invalid-decision penalty | −2.5 (already in `rewards.py`) | kills the template-echo |
| `--max-completion-length` / `--save-steps` | 384 / 15 | speed + kill-resilience |
| learning_rate | 5e-6 (unchanged) | stable GRPO LR |
| reward_funcs | structure/evidence/decision (decision-driven; the first two stay 0 on terse v1) | keep staged separation |

Run in the **vLLM container** (`nvcr.io/nvidia/vllm:25.11-py3`) on node 2, `--gpu-mem 64G`.

## Eval

Serve the new GRPO LoRA on node 1 (vLLM `--enable-lora`), backtest on **two** windows,
price-only input, with per-model `VLLM_CACHE_DIR`:
1. **2024–2026, 14-ticker** (the locked OOS) — directly comparable to v1/v2/GRPO.
2. **2025-H1, 12-ticker** (`--universe-mm --oos-start 2025-01-01 --oos-end 2025-07-01`)
   — regime check vs the multimodal cycle.

For each: record CR/Sharpe/MDD/NO_TAG **and** the decision distribution (collapse check).

## Risks & mitigations

- **vLLM colocate stability over a long run** (only smoke-verified) → monitor the first
  ~50 steps; `--save-steps 15` means a crash/kill loses ≤15 steps.
- **num-generations 12 GPU pressure** (vLLM KV + training) → drop to 8 if OOM.
- **Deeper RL may still collapse** (over-optimizing the decision reward) → the diversity
  check (criterion 2) catches it; if so, that's the finding (reward needs an
  entropy/diversity term), recorded honestly.
- **Runtime ~2–3h** on vLLM (vs ~8h+ HF) → acceptable; `--max-runtime 5h`.

## Out of scope (lever 3b, later)

Fixing the multimodal SFT collapse (regularization / data augmentation) then GRPO —
a separate sub-cycle once 3a lands.
