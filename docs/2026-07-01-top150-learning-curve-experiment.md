# Top-150 scale-up: real LLM learning curve + multimodal ceiling (2026-07-01)

> Status: **setup complete, training/inference RUNNING on node 2, results PENDING.**
> This doc captures the method now; the results tables are filled when the run lands.

## Why this experiment

Two open questions from the prior analysis were **gated on a bigger dataset**, which
the co-author delivered on 2026-07-01 (`qflib_top150_r1_trade_ready_20260701.tar.gz`):

1. **#1 — a *real* LLM learning curve.** Earlier we could only draw a *GBM-proxy*
   curve, which [we proved](2026-06-30-gbm-llm-proxy-validity.md) is valid for the
   *input signal ceiling* but **invalid for the LLM's own sample-efficiency**. With
   3,047 multimodal SFT examples (vs the old 267) we can finally train the LLM at
   three data sizes and measure its *own* OOS IC curve — not a proxy.
2. **#2 — does multimodal text break the ~0.24 ceiling at 10× universe?** The prior
   MM_RICH test (14 equities, headlines) found news *did not* help (+news IC 0.187 <
   price-only 0.221). This repeats the test at 150 equities on a clean 2025-H1 OOS.

## What arrived (the bundle)

| Modality | File | Rows | Span | QC |
|---|---|--:|---|---|
| prices (150 tk) | `prices_top150.parquet` | 376,511 | 2015-01 … 2026-05 | 100 |
| news | `news_top150.parquet` | 42,897 | 2024-01 … 2025-06 | 92.5 |
| fundamentals | `fundamentals_top150.parquet` | 35,441 | filings to 2026-06 | 99.2 |
| analyst sentiment | `sentiment_analyst_top150.parquet` | 60,230 | — | 95.8 |
| insider sentiment | `sentiment_insider_top150.parquet` | 31,209 | — | see ⚠ |
| macro (+PIT) | `macro[_pit].parquet` | 5,761 | 2022-01 … 2026-06 | 100 |
| SFT (multimodal) | `sft_top150_mm/{train,val}` | 3,047 / 317 | as-of 2024 | — |
| GRPO (multimodal) | `grpo_top150_mm/{train,val}` | 271 / 29 | as-of 2024 | — |

⚠ **QC stale-gate to flag to the co-author:** `qc_top150_report_rerun.json` reports
`sentiment.hard_pass = false` with `G1_insider_present = false` (all 150 tickers
"insider missing"), yet the shipped `sentiment_insider_top150.parquet` **contains
31,207 real transactions**. The QC pass almost certainly ran *before* the insider file
was populated (or on a wrong path). The data itself is fine; only the QC report is stale.

## Method

### Data placement (repo convention, all git-ignored)
- parquets → `data/qflib_data_store_top150/`
- SFT/GRPO jsonl → `compare_lab/{sft,grpo}/data_top150_mm/`
- canonical-named symlink store for the MM builder → `data/qflib_store_top150_canonical/`
  (three columns needed coercion for the shared renderer: `fundamentals.filing_date`
  → datetime, `fundamentals.value` → numeric, `insider.shares` → numeric; the last one
  reproduces the training prompts' `na sh` insider lines exactly).

### The training targets are TEMPLATED, not distilled
The SFT assistant targets are **deterministic value-grounded rationales** (char length
p50=292, p90=325 — a template with indicator values plugged in), *not* teacher
distillation (no `§8` / `<think>` / reasoning traces). So this curve isolates the
**pure data-scale effect of template-SFT**, with distillation (SFT v2) held as a
separate future lever. Example target:
> "…the medium-term trend is bullish: price sits relative to its 50-day (86.46) and
> 200-day (58.84) moving averages, RSI(14) is 55.30, and MACD reads 0.03. Net of trend
> and momentum, the next-week posture is bullish, warranting a long. [[[BUY]]]"

### Fresh 2025-H1 OOS eval set (`compare_lab/build_oos150_eval.py`)
The shipped `val.jsonl` is **all 2024** (same year as train) → an in-window holdout,
**not time-OOS**, so it can't test contamination or honest generalization. We build a
fresh grid instead: 1,000 (ticker, as-of) points, weekly anchors **2025-01-03 …
2025-06-27** (inside the news window that ends 2025-06-30), 146/150 tickers, using the
**same snapshot code that built the training prompts** → byte-identical structure
(NEWS/FUNDAMENTALS/SENTIMENT/MACRO). Two variants per point:
- `eval_mm.jsonl` — full multimodal (matches training)
- `eval_priceonly.jsonl` — price + technicals only
Truth = `make_signal(adj_close, forward=True)` (the continuous vol-adjusted proxy the
5-class label is cut from). **IC = Spearman(predicted class-index, signal).**

### Learning-curve subsets (stratified by label)
267 / 1,000 / 3,047 (full), preserving the bull-skewed class mix (STRONG_SELL is only
4.7 % — matches the paper's Table 2 skew).

### Train → infer → IC pipeline
- **Train** (`compare_lab/sft/train.py`, +flash-attention 2 added for the ~2.5k-tok
  prompts): LoRA r=16, BF16, completion-only loss, max-length 4096, 2 epochs, eff-batch 16.
- **Infer** (`compare_lab/infer_ic.py`, standalone / no qf-lib import so it runs in the
  NVIDIA container): base ± LoRA, greedy, parse the final `[[[CLASS]]]`.
- **IC** (`compare_lab/compute_ic.py`, uv env, pandas Spearman): IC ± SE, invalid-rate,
  class distribution.

### Execution — node 2 (idle), shared NFS
Node 1 was contended (ollama 14 G + a 40 h embed job); node 2 (gx10-3d56) was idle.
Working set staged to the **shared** `/mnt/nfs/ssd1/tr1_curve` so the node-2 container
mounts it as `/work` (no rsync) and adapters/preds land where node 1 reads them.
Orchestrated by `compare_lab/run_curve.sh` (train 267→1000→3047, then 6 inference
passes). Measured throughput ≈ 0.21 samples/s (267: 46 min, 1000: 2 h21 m, 3047: ~8 h).
Container gotcha (documented in `sft/README.md`): `pip uninstall -y torchao` is
required or PEFT LoRA loading raises an incompatible-torchao ImportError.

## Results (2026-07-02)

### #1 Learning curve (full-MM eval, 2025-H1 OOS, n=1000, SE≈0.032)
| SFT examples | IC | class dist | read |
|--:|--:|--|--|
| **0 (base, prompt-only)** | **+0.205** | BUY 492 / HOLD 269 / SELL 160 (n=951, 49 inval) | **already at the ceiling by its own reasoning** |
| 267 | +0.013 | SELL **899** / STRONG_* 66 / HOLD 35 | **mode-collapse to SELL** → ~zero signal |
| 1,000 | −0.022 | STRONG_* 499 / SELL 377 / BUY 124 | still skewed (barbell) → ~zero signal |
| 3,047 | +0.163 | SELL 297 / HOLD 275 / BUY 269 / STRONG_* 159 | balanced, real signal, but < base |

**Headline finding (the curve flips the story):** the **untrained base** Qwen3-4B already
scores **IC 0.205 — statistically at the GBM ceiling (0.215)** — purely by prompting. Template
-SFT does **not** add signal: it *collapses* at 267/1k (degenerate single-class, IC≈0) and only
*recovers* to 0.163 at 3,047 — still ~1 SE **below** the base it started from. So on this task
template-SFT is at best a lossy re-encoding of ability the base already has, and at small data it
is actively destructive. (Honest stats: base 0.205 vs sft-3047 0.163 is only ~0.9 SE — "SFT
hurts" is *suggestive*, not significant; but "SFT gives no benefit over base" and "small-N SFT
collapses" are both solid.) This still corrects the earlier invalid GBM-proxy "10× data useless"
claim — more data *does* repair collapse — but the deeper lesson is that **the ceiling is reached
before any fine-tuning**, which is exactly what an input-bound (not model-bound) ceiling predicts.

⚠️ **Contamination caveat:** 2025-H1 is inside Qwen3-**2507**'s pretraining window, so the base's
0.205 could carry some recall. But the memory-free GBM also hits 0.215 on the same features, so
~0.21 is genuinely extractable *without* memory → the base's score is plausibly real reasoning,
not just recall. (A strictly post-cutoff slice would settle it; see the ceiling investigation.)

### Same-universe GBM ceiling (how far is 0.163 from achievable?)
`compare_lab/gbm_ceiling.py` — a **valid** GBM use (input-ceiling, not LLM proxy): the same
16 technical indicators the LLM sees, GBM trained on **2024** (36,420 rows), predicting the
**exact same 1,000 2025-H1 eval points**, IC vs the same forward `make_signal`. Leak-safe
(causal features, no train/test overlap, forward signal is target-only).

| predictor | IC on the 1,000 OOS points |
|--|--:|
| GBM continuous (input ceiling) | **+0.215** |
| GBM 5-class-bucketed (LLM-fair coarseness) | +0.206 |
| **base LLM (prompt-only)** | **+0.205** |
| LLM template-SFT 3,047 | +0.163 |

GBM (0.215) and the base LLM (0.205) land together at ~0.21, with template-SFT (0.163) below.
⚠️ **This was NOT the true ceiling** — the dedicated investigation
([`2026-07-03-why-the-ceiling.md`](2026-07-03-why-the-ceiling.md)) shows a one-feature linear
**momentum** model reaches **0.266** on this proxy (GBM/LLM *under-extract*), and that the whole
proxy is ~4× inflated by smoothing — the real **raw-return** ceiling is **≈0.06** (weak-form
EMH). So "GBM 0.215 = input ceiling / LLM captures 76 %" (above) is **corrected there**: read
this GBM number as one under-extracting model on a smoothed proxy, not the information limit.

### #2 Multimodal ceiling (+news/fund/sent/macro vs price-only, same OOS) — CLEAN result
| model | full-MM IC | price-only IC | Δ | verdict |
|--|--:|--:|--:|--|
| **base (prompt-only)** | **+0.205** | **+0.203** | **+0.002** | **news/alt-data add ~nothing over price** |
| SFT-3047 | +0.163 | nan (all-STRONG, OOD) ² | — | not a clean test (modality mismatch) |

**#2 verdict:** on the clean base-model test, full multimodal text (news + fundamentals +
analyst/insider sentiment + macro) beats price-only by **+0.002 IC — a null result.** This
**replicates the prior 14-ticker MM_RICH finding at 10× universe**: at headline-level text
granularity, alt-data does not break the price/technical ceiling. (Article *bodies* / true
alt-data remain the one untested lever — see the investigation.)

¹ The first run used `max_new=200`; the base model writes a ~2,000-char analysis before
its `[[[CLASS]]]`, so all 1000 base rows were truncated-invalid. Re-running base mm+px at
`max_new=1024` (6/6 tag-hit confirmed on a probe). Trained models were unaffected (they
emit a short thesis+tag in <200 tok; 0 invalid).
² Feeding the full-MM-trained model price-only prompts (missing the NEWS/FUND/SENT/MACRO
sections it always saw) drives it out-of-distribution → degenerate all-STRONG. So the clean
#2 must come from the **base** model (no training confound), not this ablation.

## Distillation v3 (Opus 4.8 teacher) — RUNNING

The template-SFT captures only **76 %** of the GBM ceiling (§ above). Is the residual a
*rationale-quality* gap? Distillation is the direct test — but the **prior** attempt (SFT v2,
Qwen3-30B teacher, `sft/distill.py`) was a **regression**: its long §8 theses caused 9.2 %
NO_TAG non-termination and a 2.6× worse drawdown (20.7 % vs v1's 7.9 %). Verdict then: the
*long verbose style* hurt, "revisit with shorter, decisive teacher theses."

v3 does exactly that, with a much stronger teacher:
- **Teacher = Claude Opus 4.8** via the subscription (`claude -p --model opus`, `sft/distill_opus.py`)
  — no API billing, no local GPU (teacher is cloud). Per-example cache → resumable across
  subscription rate-limit windows; low concurrency + backoff.
- **v2 fix baked in:** terse prompt — 3–4 sentences, ≤110 words, each claim grounded in a
  quoted data value, **last line must be exactly `[[[LABEL]]]`.** A 5-example probe: 5/5
  tag-terminated, evidence-rich (quotes RSI/MACD/SMA + headlines + analyst upgrades), and
  even reasons the *class boundary* ("volatility argues against STRONG_BUY size → a measured
  long"). Early full-run success rate ≈ 99 %.
- **Reverse-reasoning** (teacher shown the label, justifies it) → guarantees decision↔label
  alignment; no reject-sampling discard on the noisy label.
- **Perfectly controlled comparison:** reuses the **exact same 3,047 prompts** as the
  template-SFT set (same multimodal input, same label) — only the rationale differs. Student
  LoRA SFT (same recipe, **via sparkq**), scored on the **same** 2025-H1 OOS eval + GBM ceiling.

### 3-way result (same 2025-H1 OOS, n=1000, SE≈0.032) — LANDED 2026-07-04
Corpus: 2,742 QC-passed Opus theses (98 % gate-pass, grounding 0.98, all 5 classes; the 207
of 3,000 that never distilled were rate-limit dropouts). Student LoRA SFT via sparkq (2 epochs,
same recipe as template).

| model | OOS IC | invalid | note |
|--|--:|--:|--|
| base (prompt-only) | **+0.205** | 49 | untrained; still the best |
| **Opus-distill-SFT** | **+0.171** | 0 ¹ | terse Opus theses |
| template-SFT 3,047 | +0.163 | 0 | templated rationale |
| *ref:* momentum / GBM / raw-return | 0.266 / 0.215 / **0.06** | — | proxy vs tradeable ceiling |

**Verdict — lands on pre-registered guardrail #2 ("input-bound, not rationale"):**
- distill 0.171 **> template 0.163** by only **+0.008 (~0.25 SE, not significant)** — the strongest
  possible teacher (Opus 4.8) barely moves IC over a formulaic template. Rationale *quality* is not
  the lever.
- distill 0.171 **< base 0.205** by −0.034 (~1 SE) — **no SFT beats the untrained base.** Consistent
  with the ceiling thesis: the base already sits near the input-extractable limit; SFT-to-a-noisy
  label is at best a lossy re-encoding.
- distillation's value here is **format/interpretability, not predictive skill** — and even that
  carried a cost (¹).

¹ **The thesis-first non-termination tax (literature-predicted).** At `max_new=200` the distill
model produced **489/1000 invalid** (no `[[[CLASS]]]` within budget) — it learned Opus's ~110-word
style, longer than the template's ~292-char one. At `max_new=512` invalid → **0** and IC 0.171, so
it was pure inference-budget truncation, not a format-learning failure. This is exactly the
`thesis → LABEL` ordering risk Wadhwa (EMNLP 2024) flags; the fix (label-first) is the top v3.1
recommendation in [`2026-07-03-distillation-v3-lit.md`](2026-07-03-distillation-v3-lit.md).

## How to read the result (guardrails set *before* seeing it)
- **If IC rises with data then plateaus below ~0.24:** data-scale helps template-SFT up
  to the input ceiling; the residual gap is under-extraction → distillation / better
  reward are the next levers (survey bottleneck A/B).
- **If IC is flat across 267→3047:** template-SFT is already saturated at 267; more data
  is not the lever (consistent with the input-bound-ceiling thesis).
- **If small-N collapses to one class and larger-N doesn't:** data-scale *repairs
  collapse* — the concrete value of scale even if peak IC is unchanged (links VRPO /
  bottleneck C). (A 20-sample validation slice already hinted at a SELL-skew at N=267;
  the full eval will confirm or dissolve it.)
- **#2:** Δ ≤ 0 would **replicate** the 14-ticker finding (news text doesn't break the
  ceiling at headline scale) now at 10× universe — a stronger negative result.
- Low train loss (267 reached eval-loss 0.03) is **train-fidelity, not OOS skill** — the
  whole point is that IC, not loss, is the ceiling-bound metric.
