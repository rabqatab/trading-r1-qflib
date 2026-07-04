# Distillation v3.1 design — label-first + self-consistency filter (2026-07-04)

> Acts on the two literature-backed fixes from [`2026-07-03-distillation-v3-lit.md`](2026-07-03-distillation-v3-lit.md)
> after v3 landed **IC 0.171 < base 0.205** with a **489/1000 non-termination tax** at max_new=200.
> Goal: close the gap toward the base and kill the format tax. **Not expected to beat the base** —
> the ceiling is input-bound ([`why-the-ceiling.md`](2026-07-03-why-the-ceiling.md)) — so success =
> "matches base / cleanly beats template, 0 invalid at small max_new."

## The two changes

### 1. Label-first ordering (Wadhwa, EMNLP 2024) — FREE transform
v3 target was `{thesis} … [[[LABEL]]]` (thesis-first) → the student learned Opus's ~110-word style
and ran past a 200-token budget half the time. Wadhwa shows rationale **after** the answer distills
*as well or better* and needs no test-time reasoning. So v3.1 target = **`[[[LABEL]]]\n{thesis}`**:
- structurally impossible to non-terminate before the decision (tag is token ~3);
- inference `max_new` can drop to ~24 → fast, 0 invalid by construction;
- the thesis still trains as a soft-label regularizer.
- Implemented by **reordering the 2,742 cached Opus theses** (move the trailing tag to the front) —
  **no new Opus calls.**

### 2. Self-consistency filter (STaR + SCOTT) — reader re-derivation
The core risk of reverse-reasoning on a noisy label is a *fluent justification of a wrong label*.
Filter it: an independent **reader = base Qwen3-4B** reads the thesis **body only (tag stripped)**
and predicts the call; **keep only theses whose body re-derives the true label.** This is STaR's
rationalization filter and SCOTT's consistency objective, done as a one-pass local reader (no new
Opus calls). Expected to drop the theses where Opus rationalized noise — a cleaner, smaller corpus.

## Pipeline (all GPU work via sparkq)

1. **`distill_v31.py --stage prep`** (CPU): from `data_top150_distill/_cache`, keep QC-passers, emit
   (a) `reader_eval.jsonl` = one prompt per thesis asking the reader for the call from the body only;
   (b) `_meta.jsonl` = (key, true_label, label-first record) to join after the reader pass.
2. **reader inference** (sparkq, base Qwen3-4B, `infer_ic.py`, max_new 64) → `reader_preds.jsonl`.
3. **`distill_v31.py --stage filter`** (CPU): keep theses where `reader_pred == true_label`; write the
   **label-first, self-consistent** clean corpus `data_top150_distill_v31/{train,val}.jsonl`.
4. **student SFT** (sparkq, same recipe, 2 epochs) → `sft_adapter_t150_distill_v31`.
5. **eval** (sparkq, chained `--after`, `infer_ic.py`, **max_new 64**) over the same 2025-H1 OOS →
   IC vs base 0.205 / v3-distill 0.171 / template 0.163.

## Pre-registered reads (before results)
- **v3.1 IC ≥ base (0.205):** would contradict the ceiling thesis — investigate for leakage before
  believing it.
- **template < v3.1 < base, 0 invalid at max_new=64:** the expected win — ordering fixed the format
  tax and filtering closed part of the rationale gap, but the input ceiling still caps below base.
- **v3.1 ≈ v3 (0.171):** ordering/filtering don't matter; the gap is purely input-bound (strongest
  ceiling evidence).
- **filter drops a large fraction (say >40 %):** many Opus theses were justifying label noise — a
  finding in itself (validates the SCOTT concern), report the drop rate.
- Report the **invalid rate at max_new=64** as the headline format-tax metric.
