# Stronger verifiable reward — graded continuous (bet × realized signal)

> **Date**: 2026-06-29
> **Status**: design approved, execute.
> **Context**: the prior cycle *repaired* the GRPO failures (collapse, exploration) but
> only reached v1's ceiling (Sharpe ~0.53). The remaining lever named in the memo is a
> **stronger verifiable reward**: replace the coarse 5×5 decision matrix with a dense,
> magnitude-aware reward built on the continuous vol-adjusted signal `make_signal`
> (already factored out of `make_labels`). This is the first attempt to *beat* v1, not
> just match it.

## Goal & success criteria

1. **Beat v1's ceiling** on 2024–2026 (14-eq): Sharpe > 0.53 and/or MDD < 7.9 % — a
   genuine improvement, not a repair.
2. Decision distribution stays diverse (no collapse); NO_TAG low (prompt fix holds).
3. 2025-H1 (12-eq) regime check: ≥ v1's +0.6 % / −0.30 Sharpe.

A null result (graded reward ≈ matrix) is informative — it would say reward *form* isn't
the lever either, pointing to data/architecture.

## Reward — `graded_decision_reward(text, signal)`

In `grpo/rewards.py`:
- Parse the decision → bet ∈ {STRONG_SELL −2, SELL −1, HOLD 0, BUY +1, STRONG_BUY +2}.
- No valid tag → `INVALID_DECISION_PENALTY` (−2.5, keep the echo guardrail).
- `raw = bet * clip(signal, -3, 3)` where `signal` is the realized `make_signal` value
  at that (ticker, date) — the per-day vol-adjusted forward return the 5-class label was
  cut from.
- **Asymmetric downside**: if `raw < 0`, `raw *= 1.5` (losing trades hurt more →
  capital-preservation, the paper's principle ①, now continuous).
- HOLD → 0 (anti-HOLD preserved: any correct directional call beats HOLD when |signal|>0).
- Pure, unit-testable (no model/IO).

## Data — add the continuous `signal`

`grpo/build_dataset.py`: alongside `label`, store `signal = make_signal(prices, forward=True)`
at each sampled (ticker, date). Rows become `{prompt, label, signal}`. Rebuild the
price-only GRPO set (`compare_lab/grpo/data`) with the fixed prompt + signal field.

## Training

GRPO from the **v1-reg SFT base** (last cycle's less-confident, anti-overfit SFT —
`/home/alphabridge/tr1_sft_v1reg/out`), reusing the working exploration setup:
`--use-vllm --num-generations 12 --epochs 2 --temperature 1.2 --beta 0.0
--diversity-weight 0.3 --max-completion-length 384 --save-steps 15`. `reward_funcs` =
[structure, evidence, **graded** (replaces decision), diversity]. Price-only (v1 lineage).
Run on a free GB10 node (vLLM container, `--vllm-max-len 4096` — price prompts ~700 tok).

## Eval

Serve the new LoRA, backtest 2024–2026 (14-eq) + 2025-H1 (12-eq), price-only, per-model
cache. Record CR/Sharpe/MDD/NO_TAG + decision distribution, compare to v1 and the prior
(matrix-reward) GRPOs.

## Components (create / modify)

- `compare_lab/labeling.py` — `make_signal` (already factored out; commit it here).
- `compare_lab/grpo/rewards.py` — add `graded_decision_reward`.
- `compare_lab/grpo/build_dataset.py` — add the `signal` field per record.
- `compare_lab/grpo/train.py` — add `reward_graded(completions, signal, **kw)`; swap it
  in for `reward_decision` in `reward_funcs` (keep structure/evidence/diversity).
- `compare_lab/tests/test_grpo_rewards.py` — `graded_decision_reward` behaviour test
  (sign, magnitude, asymmetry, invalid penalty).

## Risks & mitigations

- **Over-aggressive betting** (big bet on big signal → drawdown) → the ×1.5 downside
  penalty + the diversity reward temper it; MDD is a success metric, watched.
- **Signal is forward-looking** → used only as the *training reward target* (like the
  existing label), never in the backtest (model sees inputs only) — no leak.
- **Reward scale** (raw ∈ ~[−9, 6]) larger than structure/evidence (0–1) → fine, GRPO
  normalizes advantages within group; graded is the intended dominant signal.
- **Still may not beat v1** → that's the honest finding (reward form isn't the lever).

## Out of scope

More/real data expansion (the other named lever) — separate later cycle if reward form
doesn't break the ceiling.
