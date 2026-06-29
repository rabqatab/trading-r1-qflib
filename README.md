<h1 align="center">trading-r1-qflib</h1>

<p align="center">
  <em>Reimplementing Trading-R1 and benchmarking LLM vs quant-factor trading
  signals on a shared, look-ahead-safe qf-lib backtest (US equities).</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/engine-qf--lib%20(submodule)-orange" alt="qf-lib">
  <img src="https://img.shields.io/badge/tests-68%20passing-brightgreen" alt="tests">
</p>

---

## What this is

Three ways to turn information into US-equity trades, compared on **one**
evaluation harness so the numbers are actually comparable:

| # | Approach | Status |
|---|---|---|
| **#3** | **Quant factor** (12-1 momentum) | ✅ running |
| **#2** | **Prompt-only open-source LLM** (Qwen3-4B) → 5-class signal | ✅ **landed** (provenance-verified) |
| **#1** | **Trained Trading-R1** (Qwen-class, SFT → GRPO) | 🟢 **v1** keeper (MDD 7.9 %, NO_TAG 0 %) · **v2** distilled → regression · **GRPO** → Sharpe 0.58 but lost defense + echo · **multimodal** → parse not returns; collapses. **Fixed (latest):** prompt fix killed the echo (NO_TAG→2.6 %); anti-overfit + diversity-reward GRPO **dissolved the collapses** (Track B all-SELL→balanced) and recovered Sharpe 0.33→0.54 — methods work, but v1's defensive MDD 7.9 % still unbeaten |

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
│   ├── sft/              #   sub-project 2a: build_dataset · distill (teacher v2) · train (LoRA)
│   └── grpo/             #   sub-project 2b: build_dataset · rewards (§5.2) · train (TRL GRPO, GB10)
├── validate_data.py      # data QC gates G1–G5 + scored quality (docs/DATA_QC_RUBRIC.md)
├── crawl_news.py         # Google-News-RSS news crawler (reproduces news.parquet)
├── qf-lib-harness/       # submodule → github.com/ico1036/qf-lib-harness (frozen, read-only)
├── data/                 # gitignored: prices, qflib_data_store/ (multi-modal), sft_adapter_v0..v2/
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

# evaluate a *trained* adapter (SFT v1/v2, GRPO): serve it as a vLLM LoRA, then
# point the backtest at it. The cache key is the snapshot hash (model-agnostic),
# so each model MUST use its own VLLM_CACHE_DIR or it reuses another's replies.
VLLM_MODEL=sft-v1 VLLM_CACHE_DIR=compare_lab/.cache_sftv1 \
  uv run python -m compare_lab.run_comparison --llm --out compare_lab/output_sftv1
```

Writes `comparison.csv` + an interactive `equity.html`. Env knobs for the LLM
row: `VLLM_MODEL` (served id / LoRA name), `VLLM_CACHE_DIR` (per-model cache —
required), `VLLM_MAX_TOKENS` (default 2048; raise for long theses), and
`LLM_CONCURRENCY` (default 16 in-flight requests — vLLM batches them, ~13× over
serial).

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
    data (leak-safe); eval token-acc ~80% but **degenerate (all-HOLD)**.
  - ✅ SFT **v1** — completion-only loss + class balancing fixed the collapse
    (loss 0.53→0.089, `data/sft_adapter_v1/`, served `sft-v1`). Backtest
    (14-ticker, `output_sftv1/`): CR +29 %, Sharpe 0.53, **MDD 7.9 %** — most
    defensive in the set, **NO_TAG 0 %** (vs prompt-only 8.2 %). Fixed the
    collapse; didn't lift return.
  - ❌ SFT **v2** — teacher distillation (Qwen3-30B-A3B, reverse-reasoning §8;
    250 theses, `data/sft_adapter_v2/`, served `sft-v2`) **evaluated → regression
    vs v1**: CR +34 % but MDD 20.7 % (v1 7.9 %), Sharpe 0.46 (v1 0.53), and the
    verbose §8 style fails to terminate on **9.2 %** of inputs (NO_TAG, vs v1 0 %).
    The long distilled rationale hurt both risk and format — see the memo. Keep v1.
  - ⚠️ **GRPO** RL on the **v1** base (`compare_lab/grpo/`, TRL GRPOTrainer on
    GB10, decision-driven — structure/evidence reward a flat 0 on the terse v1
    base) **evaluated → mixed**: best of the trained models on **CR +37 % /
    Sharpe 0.58**, but lost v1's defense (MDD 21.6 % vs 7.9 %) and regressed parse
    to **10 % NO_TAG** (echoes the template menu instead of one class). v1 still
    the keeper; GRPO promising-but-unfinished (needs a parse guardrail + more
    epochs / higher entropy). Numbers + analysis in the memo.
- **Data integration** ✅ news/fundamentals/sentiment/macro join the snapshot by
  PIT timestamp (`compare_lab/multimodal_context.py`, opt-in in `snapshot.py`);
  delivered-data defects fixed (`*_pit.parquet`). See
  [`docs/DATA_STORE.md`](docs/DATA_STORE.md).
- ❌ **Gap-closing cycle 1 — multimodal SFT→GRPO** (train 2024 / eval 2025-H1,
  12-equity; spec+plan `docs/superpowers/{specs,plans}/2026-06-25-…`) **evaluated →
  did not close the gap.** OFF/ON ablation: multimodal helped *parse* (NO_TAG
  8.3→1.3 %) but not *returns* (both ~−6 %). Trained models collapsed — SFT-mm to
  near-all-SELL (all-cash, CR≈0), GRPO over-long (−10 %) — in a flat 2025-H1 window
  (equal-weight only +5.3 %). Caveats: thin ~900-tok snapshot (vs paper 15–23k),
  short noisy window, tiny train sets. Numbers + analysis in the memo.
  - **Lever 1 tested — richer context is *not* the bottleneck.** Enriched the
    snapshot (3-bucket news ≤50 + 12 sentiment events, ~900→~2k tok) and re-ran the
    prompt-only ablation: parse improved monotonically (NO_TAG 8.3→1.3→0.6 %) but
    **CR got *worse*** (−6.3→−8.4 %) — more context → more confident/aggressive bets,
    not better calls. Didn't retrain (de-risk).
  - **Lever 2 tested — regime isn't the excuse; it's method + training data.** On the
    same flat 2025-H1 window, price-only **SFT-v1 survives** (+0.6 %, MDD 9.3 % — only
    positive trained model), while the multimodal versions are *worse* than their
    price-only counterparts. The window is survivable; the multimodal models lack v1's
    cross-regime robustness (confounded by their 1-yr/290-ex training vs v1's 7-yr/4.2k).
  - ❌ **Lever 3a — deeper GRPO on v1 (vLLM, 2 epoch, num_gen 12, temp 1.2)** made it
    *worse*, not better (Sharpe 0.33 vs prior GRPO 0.58 / v1 0.53). RL depth is not the
    lever: entropy stuck ~0.028 (no exploration on the confident v1 base), and the 10 %
    NO_TAG is the model **copying the prompt's literal menu** `[[[A|B|C|D|E]]]` — a prompt
    bug, not an RL target. Real levers now: **exploration** (entropy in the objective /
    less-confident base) + **prompt design**. (Also verified vLLM GRPO rollout works on
    GB10, ~3× faster — so hardware was never the cap.) Next sub-cycle: lever 3b / prompt fix.
  - ✅ **Prompt fix** — dropped the copyable `[[[A|B|C|D|E]]]` menu from the prompt; a
    one-line change cut NO_TAG 8.3 %→2.6 % (menu echoes 100 %→0 %), what 3a's RL guardrail
    couldn't. Now the shared base for all further training.
  - 🟢 **Two parallel tracks (anti-overfit SFT + exploration GRPO)** — the collapses are
    over-fit-to-confidence (v1 entropy 0.028; mm-SFT all-SELL). Shared less-confident SFT
    recipe (dropout↑, fewer epochs) then RL with a diversity reward + lower KL on the
    price-only track; regularized SFT→GRPO on the multimodal track. Spec:
    `docs/superpowers/specs/2026-06-29-…`. ✅ **Both diagnosed causes fixed:** Track A
    (v1-reg) restored within-group decision diversity (StrongBuy collapse gone) and
    recovered Sharpe **0.33→0.54** (24–26); Track B (mm-reg) **dissolved the all-SELL
    collapse** (97 %→BUY 49/HOLD 28/SELL 12) — best multimodal result, edges v1 on the
    2025-H1 Sharpe. The methods work; v1's defensive MDD 7.9 % is still the ceiling to beat.
  - 🟢 **Stronger verifiable reward (graded continuous)** — replace the coarse 5×5
    decision matrix with `bet × clip(make_signal, ±3)` (dense, magnitude-aware; losing
    trades ×1.5). GRPO from the v1-reg base; **trained** (reward climbed −1.58→+0.33,
    the healthiest GRPO curve yet), **eval running**. First attempt to *beat* v1's ceiling
    rather than repair a failure — if the reward form doesn't break it, data is the real
    bottleneck. Spec: `docs/superpowers/specs/2026-06-29-graded-…`.

## Testing

```bash
uv run python -m pytest -q   # 52 passing
```
