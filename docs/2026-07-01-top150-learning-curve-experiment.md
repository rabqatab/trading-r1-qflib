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

## Results — PENDING

### #1 Learning curve (all on full-MM eval, 2025-H1 OOS, n≈1000)
| SFT examples | IC | ±SE | invalid % | class dist | note |
|--:|--:|--:|--:|--|--|
| 0 (base) | _pending_ | | | | |
| 267 | _pending_ | | | | |
| 1,000 | _pending_ | | | | |
| 3,047 | _pending_ | | | | |

### #2 Multimodal ceiling (base + best-SFT, +news vs price-only, same OOS)
| model | full-MM IC | price-only IC | Δ | verdict |
|--|--:|--:|--:|--|
| base | _pending_ | _pending_ | | |
| SFT-3047 | _pending_ | _pending_ (OOD caveat) | | |

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
