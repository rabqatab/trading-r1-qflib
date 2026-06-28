# Lever 3 — two parallel tracks: anti-overfit SFT + exploration GRPO

> **Date**: 2026-06-29
> **Status**: design approved, execute.
> **Context**: the negatives converged on one root cause — small models **over-fit to
> degenerate confidence**. v1 (eval-loss 0.089) is so confident (entropy 0.028) that
> GRPO can't explore (lever 3a got worse); the multimodal SFT collapses to all-SELL.
> Both are the same disease. The echo was a separate prompt bug, already fixed
> (`_PROMPT_HEADER`, NO_TAG 8.3→2.6 %). This cycle attacks the over-confidence on two
> lineages in parallel, then RL on top.

## Shared recipe — anti-overfit SFT

Both tracks retrain SFT with a less-confident, higher-entropy recipe (new flags in
`sft/train.py`): **`label_smoothing` (0.1)** + **higher LoRA `dropout` (0.05→0.15)** +
**fewer epochs (2→1)**. Goal: a base whose logits aren't razor-peaked, so RL can sample
diverse rollouts and the SFT doesn't collapse to one class.

## Track A — exploration GRPO on the price-only lineage (node 1)

1. Retrain the price-only SFT with the anti-overfit recipe → `data/sft_adapter_v1_reg`.
2. GRPO on it with **all three exploration levers**:
   - **diversity reward** — new `diversity_reward(completions, **kw)` in `grpo/rewards.py`:
     per group, reward each completion by how *rare* its decision is within the group
     (rarer → higher), small weight, so the policy is nudged off the single-mode prior.
   - **lower KL `--beta`** (default ~0.04 → 0.0 or 0.01) so the policy can drift from the
     confident ref.
   - vLLM rollout (`--use-vllm`), fixed prompt, temperature 1.2.
3. Eval (serve + backtest): 2024–2026 (14-eq) + 2025-H1 (12-eq).

## Track B — regularized multimodal SFT → GRPO (node 2)

1. Retrain the multimodal SFT with the **same anti-overfit recipe** → `data/sft_adapter_mm_reg`.
2. GRPO on it (vLLM, fixed prompt) — same as the prior multimodal GRPO config.
3. Eval: 2025-H1 (12-eq).

## Parallel execution (two GB10 nodes)

```
Round 1 (SFT, ~30 min each):  node1 v1-reg SFT      ‖  node2 mm-reg SFT
Round 2 (GRPO, ~2–4 h each):  node1 v1-reg GRPO     ‖  node2 mm-reg GRPO
Round 3 (eval):               serve + backtest each (serialise on a free node)
```
Node 1 is alphabridge's home (repo + qf-lib-harness local — no rsync; `train.py` is
standalone so it runs as a script either way). Node 2 via rsync as usual. sparkq cap is
1 heavy job / node, so the two tracks occupy one node each.

## Components (create / modify)

- `sft/train.py` — add `--label-smoothing` (default 0.0) and `--lora-dropout` (default 0.05).
- `grpo/rewards.py` — add `diversity_reward(completions, **kw)` (rare-decision bonus).
- `grpo/train.py` — add `--beta` (KL coeff) and `--diversity-weight`; wire the diversity
  reward into `reward_funcs` when weight > 0.
- Tests: `test_grpo_rewards.py` gets a `diversity_reward` shape/behaviour check.

## Success criteria

1. **Track A**: base entropy meaningfully above 0.028; GRPO decision distribution diverse
   (not 50 % StrongBuy); 2024–2026 Sharpe ≥ the prior GRPO's 0.58 without the echo.
2. **Track B**: the all-SELL collapse is gone (a spread distribution like v1's 38/31/28);
   2025-H1 ≈ or > v1's +0.6 % / MDD 9.3 %.
3. Both: NO_TAG stays low (prompt fix), no template echo.

A null result on either is informative (records that anti-overfit/exploration wasn't the
lever for that lineage); both are evaluated honestly.

## Risks & mitigations

- **Three levers at once (Track A) → can't attribute** which helped. Accepted ("make it
  work" first); if A succeeds we can ablate later.
- **diversity reward gameable** → small weight (≤0.3) + monitor the decision distribution.
- **label smoothing too high → format breaks** → conservative 0.1; smoke-check parse rate.
- **node-1 training contends with ollama** (14 GB) → ~114 GB free, fine; serve rounds
  pick whichever node is free.
- **vLLM colocate stability over long runs** → `--save-steps 15`, monitor first ~50 steps.

## Out of scope

Data augmentation (paper ensemble) for the multimodal set — deferred; this cycle isolates
the regularization lever.
