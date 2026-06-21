<h1 align="center">trading-r1-qflib</h1>

<p align="center">
  <em>Reimplementing Trading-R1 and benchmarking LLM vs quant-factor trading
  signals on a shared, look-ahead-safe qf-lib backtest (US equities).</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/engine-qf--lib%20(submodule)-orange" alt="qf-lib">
  <img src="https://img.shields.io/badge/tests-52%20passing-brightgreen" alt="tests">
</p>

---

## What this is

Three ways to turn information into US-equity trades, compared on **one**
evaluation harness so the numbers are actually comparable:

| # | Approach | Status |
|---|---|---|
| **#3** | **Quant factor** (12-1 momentum) | ✅ running |
| **#2** | **Prompt-only open-source LLM** (Qwen3-4B) → 5-class signal | ✅ **landed** (provenance-verified) |
| **#1** | **Trained Trading-R1** (Qwen-class, SFT → GRPO) | 🟡 SFT v0 trained + evaluated → **degenerate (all-HOLD)**; v1 + GRPO next |

Every approach emits the same thing — a **target-weight matrix
`[date × ticker]`** — which is run through a single
[qf-lib](https://github.com/quarkfin/qf-lib) event backtest and scored on
**CR / Sharpe / Hit-Rate / Max-Drawdown**. Look-ahead is blocked at three
layers (snapshot `as_of`, signal lag, next-bar execution).

Based on **Trading-R1** (Xiao et al., [arXiv:2509.11420](https://arxiv.org/abs/2509.11420));
code/data/weights are unreleased, so this is a *reimplementation*, not a rerun.

## Layout

```
trading-r1-qflib/
├── compare_lab/          # the comparison substrate (sub-project 1)
│   ├── snapshot.py       #   per-ticker, as-of, price+technical → LLM prompt
│   ├── providers/        #   equal_weight · momentum · llm  (all → weight matrix)
│   ├── llm_client.py     #   vLLM (OpenAI-compatible) client + response cache
│   ├── backtest.py       #   qf-lib bridge: weights → daily returns
│   ├── metrics.py        #   CR / SR / HR / MDD
│   ├── run_comparison.py #   CLI: run every provider, write the comparison
│   ├── analyze_results.py#   richer diagnostics (corr, decision dist) for the memo
│   ├── labeling.py       #   volatility 5-class labels (Algorithm S1) — SFT/RL target
│   ├── macro_pit.py      #   fix delivered macro release_date leak  → macro_pit.parquet
│   ├── insider_pit.py    #   recover insider txn-type from text     → *_pit.parquet
│   ├── fundamentals_pit.py# normalize the two revenue XBRL tags     → *_pit.parquet
│   ├── multimodal_context.py# PIT join of news/fundamentals/sentiment/macro
│   └── sft/              #   sub-project 2: build_dataset · train (LoRA) · README
├── validate_data.py      # data QC gates G1–G5 + scored quality (docs/DATA_QC_RUBRIC.md)
├── crawl_news.py         # Google-News-RSS news crawler (reproduces news.parquet)
├── qf-lib-harness/       # submodule → github.com/ico1036/qf-lib-harness (frozen, read-only)
├── data/                 # gitignored: prices, qflib_data_store/ (multi-modal), sft_adapter_v0/
├── docs/                 # DATA_STORE · DATA_QC_RUBRIC · DATA_REQUIREMENTS · memo · superpowers/{specs,plans}
└── tradingR1.pdf         # the paper
```

`compare_lab` reuses the `qf-lib-harness` submodule's data loaders and the
qf-lib engine; it never modifies the (frozen) harness core.

## Setup

```bash
git clone --recurse-submodules git@github.com:rabqatab/trading-r1-qflib.git
cd trading-r1-qflib
# (if you cloned without --recurse-submodules:)
git submodule update --init

uv sync   # builds qf-lib from the pinned fork; first run takes a few minutes
```

**Data** (`qf-lib-harness/data/prices.parquet`, gitignored — never committed):
build it once via the harness pipeline. Fast path (~5 min, S&P 500 only):

```bash
cd qf-lib-harness && uv run python research/fast_fetch_prices.py
```

or the full ~7k-ticker pipeline (`research/1_…` → `5_…`, 30 min–2 h).

## Run the comparison

```bash
# baselines only (no LLM, no GPU needed)
uv run python -m compare_lab.run_comparison --out compare_lab/output

# add the prompt-only LLM row (requires a vLLM endpoint; see Roadmap)
uv run python -m compare_lab.run_comparison --llm --out compare_lab/output
```

Writes `comparison.csv` + an interactive `equity.html`.

## Results so far

Out-of-sample **2024-01 → 2026-04**, 12 large-cap equities, weekly rebalance:

| Strategy | Cumulative | Sharpe | MaxDD |
|---|---|---|---|
| Equal-weight (market) | **+126 %** | **1.07** | 27.8 % |
| 12-1 momentum | +50 % | 0.66 | 19.6 % |
| **Prompt-only LLM** (Qwen3-4B) | +43 % | 0.71 | **14.8 %** |

In this bull run equal-weight leads on raw return; the prompt-only LLM **lags on
return but carries the lowest drawdown** and — the key finding — its daily
returns are the **least correlated** to the baselines (0.63–0.69 vs 0.90 between
the two quant strategies), i.e. a *differentiated* signal, not a momentum copy.
Full write-up + caveats: [`docs/2026-06-21-three-way-comparison-memo.md`](docs/2026-06-21-three-way-comparison-memo.md)
(`compare_lab/output/report.html` for the interactive version). The LLM
decisions are provenance-verified (12/12 cached replies reproduced by the live
model). **Robustness:** adding SPY/QQQ (14-ticker run, `compare_lab/output_14/`)
*lowers* the LLM's Sharpe (0.71→0.55) while it keeps the lowest drawdown — the
prompt-only signal is sensitive to universe composition (it goes long the ETFs
too), which is itself an argument for training (#1).

## Data

**The multi-modal point-in-time pull has landed** (2026-06-21,
`data/qflib_data_store/`): price · news · fundamentals · analyst & insider
sentiment · macro — 6 parquet files, each carrying a real publish/filing
timestamp. Full schema and coverage in [`docs/DATA_STORE.md`](docs/DATA_STORE.md).
Two delivered-data PIT defects were **found and fixed** (unit-tested): the macro
`release_date` leak → `macro_pit.parquet`, and the empty insider txn-type →
`sentiment_insider_pit.parquet`. The remaining job is **integration** (join
modalities into the snapshot by their PIT timestamp), not procurement.

| Modality | File | Coverage |
|---|---|---|
| price (14 tk + raw_close) | `prices.parquet` | 2015–2026 |
| news (headlines) | `news.parquet` | 12 eq × 2024-01→2025-06 |
| fundamentals (10-Q/K) | `fundamentals.parquet` | 8 concepts |
| sentiment (analyst) | `sentiment_analyst.parquet` | 12 eq |
| sentiment (insider) | `sentiment_insider_pit.parquet` ✅ | 12 eq (txn-type recovered) |
| macro (FRED, 8 series) | `macro_pit.parquet` ✅ | 2022–2026 (leak-fixed) |

Procurement history + per-source paper mapping:
[`docs/DATA_REQUIREMENTS.md`](docs/DATA_REQUIREMENTS.md). **Quality gates:**
[`docs/DATA_QC_RUBRIC.md`](docs/DATA_QC_RUBRIC.md) (G1–G5 + scored) — run
`uv run python validate_data.py` (currently all hard gates pass, weighted 98.6;
the macro leak is caught by a `G2_macro_release_lag` gate).

**Hard requirement:** every non-price item must be joined on its real
**publish/filing timestamp** ("available as of trading day *t*") — any
future-dated leakage invalidates the backtest.

## Roadmap

- **Sub-project 1 — comparison substrate + baselines** ✅ done
  (`docs/superpowers/specs/2026-06-15-compare-lab-eval-substrate-design.md`).
- **Task 11 — prompt-only LLM row** ✅ done — `Qwen3-4B-Instruct-2507` served
  single-node vLLM (`--enforce-eager`, BF16) on DGX Spark via `sparkq`; LLM row
  landed and provenance-verified.
- **Sub-project 2 — train Trading-R1** 🟢 in progress
  (`docs/superpowers/specs/2026-05-25-trading-r1-dgx-spark-design.md`):
  - ✅ `labeling.py` — volatility 5-class labels (Algorithm S1); distribution
    matches paper Table 2 (Strong-Sell→Strong-Buy: 3/12/38/32/15%).
  - ✅ SFT **v0** — LoRA on Qwen3-4B (`compare_lab/sft/`), trained on pre-2024
    data (leak-safe); eval token-acc ~80%. GB10 compatibility gate cleared.
  - 🔜 teacher distillation (Qwen3-32B) to replace templated rationale, then
    **GRPO** RL with the structure/evidence/decision rewards.
- **Data integration** ✅ news/fundamentals/sentiment/macro join the snapshot by
  PIT timestamp (`compare_lab/multimodal_context.py`, opt-in in `snapshot.py`);
  delivered-data defects fixed (`*_pit.parquet`). See
  [`docs/DATA_STORE.md`](docs/DATA_STORE.md). Next: feed it to the LLM row + GRPO.

## Testing

```bash
uv run python -m pytest -q   # 52 passing
```
