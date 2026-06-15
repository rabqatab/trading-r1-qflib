<h1 align="center">trading-r1-qflib</h1>

<p align="center">
  <em>Reimplementing Trading-R1 and benchmarking LLM vs quant-factor trading
  signals on a shared, look-ahead-safe qf-lib backtest (US equities).</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/engine-qf--lib%20(submodule)-orange" alt="qf-lib">
  <img src="https://img.shields.io/badge/tests-23%20passing-brightgreen" alt="tests">
</p>

---

## What this is

Three ways to turn information into US-equity trades, compared on **one**
evaluation harness so the numbers are actually comparable:

| # | Approach | Status |
|---|---|---|
| **#3** | **Quant factor** (12-1 momentum) | ✅ running |
| **#2** | **Prompt-only open-source LLM** (≈4B) → 5-class signal | 🔜 needs a served model |
| **#1** | **Trained Trading-R1** (Qwen-class, SFT + GRPO) | 🗺️ planned (separate spec) |

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
├── compare_lab/         # the comparison substrate (this repo's code)
│   ├── snapshot.py      #   per-ticker, as-of, price+technical → LLM prompt
│   ├── providers/       #   equal_weight · momentum · llm  (all → weight matrix)
│   ├── llm_client.py    #   vLLM (OpenAI-compatible) client + response cache
│   ├── backtest.py      #   qf-lib bridge: weights → daily returns
│   ├── metrics.py       #   CR / SR / HR / MDD
│   └── run_comparison.py#   CLI: run every provider, write the comparison
├── qf-lib-harness/      # submodule → github.com/ico1036/qf-lib-harness (frozen, reused read-only)
├── docs/superpowers/    # specs/ (design) + plans/ (implementation)
└── tradingR1.pdf        # the paper
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

Out-of-sample **2024-01 → 2026-04**, 12 large-cap equities
(SPY/QQQ excluded — the fast S&P-500 dataset has no ETFs):

| Strategy | Cumulative | Sharpe | MaxDD |
|---|---|---|---|
| Equal-weight (market) | **+126 %** | **1.07** | 27.8 % |
| 12-1 momentum | +50 % | 0.66 | **19.6 %** |

In this bull run equal-weight leads on return/Sharpe; momentum draws down less.
*(Baseline comparison only — the LLM row is not in yet.)*

## Roadmap

- **Sub-project 1 — comparison substrate + baselines** ✅ done
  (`docs/superpowers/specs/2026-06-15-compare-lab-eval-substrate-design.md`).
- **Task 11 — prompt-only LLM row** 🔜 serve a ≈4B model with single-node vLLM
  (`--enforce-eager`, BF16) on DGX Spark via `sparkq`, then `run_comparison --llm`.
- **Sub-project 2 — train Trading-R1** 🗺️ data → distillation → SFT → GRPO on
  DGX Spark (`docs/superpowers/specs/2026-05-25-trading-r1-dgx-spark-design.md`).

## Testing

```bash
uv run pytest -q   # 23 passing
```
