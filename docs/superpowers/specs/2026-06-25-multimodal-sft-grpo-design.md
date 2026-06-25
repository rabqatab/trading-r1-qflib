# Multimodal SFT→GRPO — gap-closing cycle 1

> **Date**: 2026-06-25
> **Status**: design approved, ready for implementation plan
> **Context**: the price-only lineage (SFT v1 keeper, v2 distill regression, GRPO
> best-Sharpe-but-lost-discipline) sits far below the paper on absolute Sharpe.
> The dominant cause is the **data/modality gap**: we trained on price+technical
> only, the paper on 5 modalities × 100K samples. This cycle closes the biggest
> lever — feed the multimodal store into the snapshot and retrain SFT→GRPO.

## Goal & success criteria

Per the paper-summary §9.6, the bar is **relative improvement + trend match**, not
absolute reproduction (data/weights unreleased).

1. **(primary) Multimodal LLM > price-only LLM** on the same 2025-H1 window —
   evidence that the extra modalities add signal. Measured by an ON/OFF ablation
   of the prompt-only row.
2. **Multimodal SFT→GRPO > multimodal prompt-only** — evidence that training helps.
3. **Parse discipline holds** — NO_TAG low, zero template-echo (the GRPO defect we
   measured last cycle).

A negative result (multimodal does not help) is a valid, publishable outcome — the
ablation reports it honestly.

## Data & leak-safe split

The multimodal store starts 2024 (news is the binding modality, 2024-01→2025-06),
so the clean "pre-2024 train / 2024+ eval" separation is gone. Use a temporal split
inside the multimodal period:

- **Train**: 2024-01-01 → 2024-12-31 (12 months)
- **Eval (OOS)**: 2025-01-01 → 2025-06-30 (6 months, news present)
- No overlap, strictly causal. Eval is short → Sharpe is noisier (accepted; still
  longer than the paper's 3-month holdout).

**Coverage (verified 2026-06-25, `data/qflib_data_store/`)** — rows in 2024 / 2025-H1:
news 7324/4562, fundamentals_pit 705/347, macro_pit 1296/638, sentiment_analyst
1189/481, sentiment_insider_pit 155/312. All five modalities present in both windows.

**Universe**: the **12 equities** (drop SPY/QQQ — ETFs have thin company news /
fundamentals): NVDA MSFT AAPL META AMZN TSLA BRK-B JPM LLY JNJ XOM CVX.

This is a **separate comparison** from the price-only v1/v2/GRPO table (different
window/universe) — not directly comparable; report it as its own lineage.

## SFT stage (multimodal v3)

Reuse the **v1 recipe** (the proven one): templated, value-grounded rationale →
`[[[CLASS]]]`, completion-only loss, class balancing. The multimodal signal enters
through the **input snapshot**, not a distilled rationale.

- **Why not distillation**: v2 (local teacher distill) regressed on every risk axis
  and broke format. Re-distillation (the paper's reverse-reasoning approach) is
  deferred to a later cycle; this cycle isolates the *input* change.
- **Input**: `snapshot.py` with `multimodal=True` (existing `multimodal_context.py`
  PIT-joins news/fundamentals/sentiment/macro). Built on the 2024 train window,
  12-equity universe.
- **Context length**: multimodal snapshots are long. Truncate to **~8k tokens** for
  training (paper-summary §9.7); raise `train.py` `max_length` from 2048 to 8192.
  Watch GB10 memory (BF16 LoRA + grad-ckpt; bump `--gpu-mem`, lower batch if needed).
- **Output**: `data/sft_adapter_mm_v3/`, served `sft-mm-v3`.

## GRPO stage (multimodal, with last cycle's fixes)

Merge the multimodal-SFT LoRA into the backbone → train a fresh GRPO LoRA. Carry the
fixes learned from the price-only GRPO cycle:

- **Parse guardrail (new)**: the decision reward already returns −1.5 for no valid
  tag, but last cycle 10% of outputs echoed the template menu
  `[[[STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL]]]`. Strengthen `decision_reward` (or add
  a small dedicated penalty) so a verbatim-menu / multi-class echo is penalised as
  hard as a wrong call — kill the degenerate mode at the reward level.
- **structure/evidence rewards**: on the price-only terse base they scored a flat 0
  (no gradient). With multimodal input the SFT outputs may be richer; re-check via
  smoke whether they activate. If still flat, stay decision-driven (documented).
- **Stronger RL signal**: raise sampling entropy (generation temperature) and run
  more than 1 epoch — last cycle's near-deterministic policy (entropy 0.026) gave
  GRPO little to exploit.
- **Output**: `data/sft_adapter_mm_grpo/`, served `mm-grpo`.

## Eval & comparison

Run the qf-lib backtest on the **2025-H1** window (`--oos-start`/`--oos-end` flags
added to `run_comparison`, defaulting to the locked config; the multimodal cycle
passes 2025-01-01/2025-07-01). Per-model cache isolation (`VLLM_CACHE_DIR`) and the
parallel LLM client (`LLM_CONCURRENCY`) as documented.

Rows on the 2025-H1 / 12-equity window:
- equal-weight, 12-1 momentum (baselines, recomputed on this window)
- **prompt-only LLM, multimodal OFF** (price-only input) ← ablation control
- **prompt-only LLM, multimodal ON** ← isolates the modality contribution (criterion 1)
- **multimodal SFT v3**
- **multimodal SFT→GRPO** (criterion 2)

Document the table + verdict in the memo; update README #1 row + PROGRESS.

## Components (create / modify)

- `compare_lab/sft/build_dataset.py` — add `--multimodal` + train-window flags (2024).
- `compare_lab/grpo/build_dataset.py` — same: `--multimodal`, 2024 window, 12-equity.
- `compare_lab/sft/train.py` — `max_length` 2048→8192 (flag), adapter out path.
- `compare_lab/grpo/train.py` — generation temperature flag, epochs default >1.
- `compare_lab/grpo/rewards.py` — strengthen the parse/echo guardrail in
  `decision_reward`; keep the staged separation. Add a unit test for the echo case.
- `compare_lab/run_comparison.py` — `--oos-start`/`--oos-end` flags; ensure the
  LLM provider's `multimodal` flag is wired from a CLI/env switch.
- `compare_lab/config.py` — add the multimodal experiment window + 12-equity subset
  as named constants (don't mutate the locked OOS_START/END).
- Launch recipes mirror the GB10 node-2 pattern (`run_node2.sh`, rsync, sparkq).

## Risks & mitigations

- **Short eval (6 mo)** → noisy Sharpe. Mitigation: report HR + MDD alongside, and
  the ON/OFF ablation is within-window (controls for the regime).
- **Context length** blows up GB10 memory / slows training. Mitigation: 8k truncate,
  smoke-gate first, lower batch / raise grad-accum.
- **Multimodal may not help** → that's the honest answer; the ablation surfaces it.
- **Coverage skew** (insider sparse in 2024: 155 rows) → some tickers/dates lack a
  modality; `multimodal_context.render_sections` must degrade gracefully (omit empty
  sections), not crash.

## Out of scope (later cycles)

- Re-distillation with a frontier teacher (paper's reverse-reasoning).
- The full 3-stage easy-to-hard curriculum.
- Extending eval past 2025-06 (needs a fresh news pull).
