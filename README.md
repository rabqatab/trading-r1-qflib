<h1 align="center">trading-r1-qflib</h1>

<p align="center">
  <em>Reimplementing Trading-R1 and benchmarking LLM vs quant-factor trading
  signals on a shared, look-ahead-safe qf-lib backtest (US equities).</em>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.11-blue" alt="Python 3.11">
  <img src="https://img.shields.io/badge/engine-qf--lib%20(submodule)-orange" alt="qf-lib">
  <img src="https://img.shields.io/badge/tests-70%20passing-brightgreen" alt="tests">
</p>

---

> 📄 **Final write-up (2026-07-18):** [*You Can't Model Your Way Past the Information Bound — But
> You Can Control the Tail*](docs/2026-07-18-writeup-two-layer-result.md) — the consolidated story
> of the whole arc: the prediction ceiling is information-bound (~0.06 raw IC), the residual alpha
> was universe selection (+0.096 → +0.004 on a PIT universe), and the one thing that survives
> every honesty upgrade is the RU-conformal CVaR exposure controller (8 streams, tail pinned,
> costs included). Includes the 7 measurement traps the next person should not re-discover.

## What this is

Three ways to turn information into US-equity trades, compared on **one**
evaluation harness so the numbers are actually comparable:

| # | Approach | One-line status |
|---|---|---|
| **#3** | **Quant factor** (12-1 momentum) | ✅ baseline |
| **#2** | **Prompt-only LLM** (Qwen3-4B, zero-shot) → 5-class signal | ✅ landed, provenance-verified — differentiated but universe-sensitive |
| **#1** | **Trained Trading-R1** (Qwen3-4B, SFT → GRPO) | 🟢 best LLM = graded-reward GRPO (bull-window Sharpe 0.93) — but it's a **bull-window long-bias, not prediction skill** (IC stuck ~0.2). See [Models at a glance](#models-at-a-glance). |

Every approach emits the same thing — a **target-weight matrix
`[date × ticker]`** — run through one
[qf-lib](https://github.com/quarkfin/qf-lib) event backtest, scored on
**CR / Sharpe / Hit-Rate / Max-Drawdown**. Look-ahead is blocked at three
layers (snapshot `as_of`, signal lag, next-bar execution).

Based on **Trading-R1** (Xiao et al., [arXiv:2509.11420](https://arxiv.org/abs/2509.11420));
code/data/weights are unreleased, so this is a *reimplementation*, not a rerun.
Full results + caveats: [memo](docs/2026-06-21-three-way-comparison-memo.md) ·
[honest-lens report](docs/2026-06-29-results-report.html) ·
[chronological log](docs/PROGRESS-2026-06-21.md).

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

# add the prompt-only LLM row (requires a vLLM endpoint)
uv run python -m compare_lab.run_comparison --llm --out compare_lab/output
```

Writes `comparison.csv` + an interactive `equity.html`. To evaluate a *trained*
adapter, see [Reproduce a model's evaluation](#reproduce-a-models-evaluation).
Env knobs for the LLM row: `VLLM_MODEL` (served id / LoRA name), `VLLM_CACHE_DIR`
(per-model cache — **required**, key is the model-agnostic snapshot hash),
`VLLM_MAX_TOKENS` (default 2048), `LLM_CONCURRENCY` (default 16, ~13× over serial).

## Models at a glance

**Shared input/output (every LLM model).** Each model reads one point-in-time
snapshot and emits a short thesis ending in a single 5-class tag.

- **Input** — ticker, last 15 days of OHLCV, technical indicators (SMA/RSI/MACD/…),
  optionally `+` a PIT-joined multimodal block (news/fundamentals/sentiment/macro).
  *Example (NVDA, 2017-01-24):* `Ticker: NVDA … Date|Open High Low Close Volume …
  Indicators: 50-sma 2.35, RSI 52.3, MACD 0.05 …`
- **Output** — `… the next-week posture is bullish … \n\n[[[BUY]]]`
  (one of `STRONG_SELL/SELL/HOLD/BUY/STRONG_BUY`).
- **Label / reward target** — deterministic, computed from *future* price (never an
  input): `make_signal` = vol-adjusted forward return over 3/7/15 d (Alg S1), cut
  into the 5 classes. The continuous value (e.g. `+1.29`) is the `signal` the graded
  reward and the IC metric use.

**The ladder** (OOS 2024-01→2026, 14-eq portfolio backtest; **IC** = rank-corr of the
predicted class vs the label, 0=chance):

| Model | What it is / how trained | CR | Sharpe | MDD | IC | Verdict |
|---|---|--:|--:|--:|--:|---|
| Equal-weight | market, no model | +143 % | 1.04 | 32 % | — | bull-run baseline |
| 12-1 Momentum | price factor, no model | +52 % | 0.70 | 20 % | — | quant baseline |
| **#2 Prompt-only** | Qwen3-4B, **zero-shot** (no training) | +43 %\* | 0.71\* | 14.8 %\* | — | differentiated, universe-sensitive |
| SFT **v0** | LoRA, full-seq loss, imbalanced data | — | — | — | — | ❌ degenerate (all-HOLD, all-cash) |
| **SFT v1** | LoRA + **completion-only loss + class balancing** | +29 % | 0.53 | **7.9 %** | 0.13 | ✅ keeper — most defensive, NO_TAG 0 % |
| SFT v2 | v1 + **teacher distillation** (Qwen3-30B, §8 theses) | +34 % | 0.46 | 21 % | 0.13 | ❌ regression (verbose → risk↑, 9 % no-tag) |
| GRPO (matrix) | RL on v1, asymmetric 5×5 decision reward | +37 % | 0.58 | 22 % | 0.19 | ⚠️ best CR, lost v1's defense + 10 % no-tag |
| Multimodal SFT/GRPO | v1 recipe + news/fund/sentiment input | ≈0 | <0 | — | 0.17 | ❌ collapsed (all-SELL / over-long) |
| Two-track (v1-reg / mm-reg) | anti-overfit SFT + **diversity reward, β=0** | +34 % | 0.54 | 18 % | 0.18 | ✅ repaired the collapses (→ ~v1 level) |
| **GRPO graded** ⭐ | RL on v1-reg, **continuous `bet × signal`** reward | **+53 %** | **0.93** | 11 % | 0.19 | 🟢 best bull-window return — *but see findings* |
| **GBM (tabular)** | gradient-boosting on technical features, **not an LLM** | — | **1.14**† | 12 %† | **0.24** | ⭐ best predictor *and* best net trader |

<sub>\* prompt-only is the **12-eq** run (other rows are the 14-eq portfolio); on the
14-eq universe its Sharpe falls to 0.55 (it goes long the added ETFs) — universe
sensitivity is itself an argument for training. † GBM = vectorised backtest, **net of
10 bps/side** costs, (3,7,15)-d weekly. Dead-end micro-experiments (deeper GRPO,
context-richness levers) are in the [memo](docs/2026-06-21-three-way-comparison-memo.md).</sub>

## What we learned (honest findings)

> **⏫ 2026-07 top-150 scale-up — the ceiling is information-bound at raw IC ~0.06, and no modelling
> beats it.** Full scoreboard: [`docs/2026-07-07-final-synthesis.md`](docs/2026-07-07-final-synthesis.md);
> why: [`docs/2026-07-03-why-the-ceiling.md`](docs/2026-07-03-why-the-ceiling.md); next work:
> [`docs/2026-07-06-post-ceiling-roadmap.md`](docs/2026-07-06-post-ceiling-roadmap.md).
>
> **Honest scoreboard — raw 7-day-return IC, same 2025-H1 OOS (proxy IC in parens):**
> | approach | type | raw IC |
> |--|--|--:|
> | **3-signal combo: mom+revision+PEAD** (rank mean) | **combination of new info** | **+0.096** (2025-H1; decade ~+0.01; **PIT universe +0.004 — universe-selection artifact**) |
> | **analyst revision momentum** (Finnhub) | **new information** | **+0.080** (window-unstable) |
> | PEAD (earnings-surprise drift, Finnhub) | new information | +0.068 (most stable) |
> | reject-sampled blind SFT | best extraction | +0.053 (0.261) |
> | *momentum, 1 feature (ref)* | baseline | +0.064 (0.266) |
> | distill v3.1 (label-first+filter) | extraction | +0.025 (0.228) |
> | base LLM / template / news headline / **+summary text** | — | **~0.00** (0.16–0.21) |
>
> Three refinements that **overturn** the "~0.24 ceiling" narrative below:
> - **The "0.24 ceiling" was model under-extraction of a smoothing-inflated proxy.** 1-feature linear
>   momentum hits proxy IC 0.266 (> GBM 0.215 > every LLM); on **raw returns it collapses to 0.064**
>   (matches Gu-Kelly-Xiu best-ML ρ≈0.063). Measured MI (KSG 0.034 nats) → the cap is **I(X;Y), not
>   the model**. **Report raw-return IC always — the proxy reads ~4× too good and is gameable** (v3.1
>   "beat" base+GBM on proxy but was ~0 on raw — a bearish-collapse mirage the raw check caught).
> - **No modelling lever beats the base on raw returns** — scale, Opus-4.8 distillation, reject-
>   sampling all move raw IC ~0→0.05. **Only new information outside the own-price path moved it more**
>   (analyst revisions, +0.080) — the data-processing inequality made concrete.
> - **Prose is useless to the LLM here; structure is not.** News headlines AND full article summaries
>   both gave raw IC ≈ 0 (the paper's core "rich text to the LLM" lever is a **clean null** at 150-tk);
>   the structured analyst *response* (revision) gave 0.080. → use the LLM as a structure-encoder.
> - **2026-07-17 — prediction plateaus, but risk is controllable** ([`docs/2026-07-17-cvar-conformal-control.md`](docs/2026-07-17-cvar-conformal-control.md)):
>   the RU-conformal online CVaR controller (arXiv:2606.00320) on the combo LS, 2017–2026: exposure-only
>   control lifts Sharpe 0.72→**0.92**, cuts maxDD 24.7%→**9.2%**, pins realized CVaR to target
>   (0.487% vs 0.500%) through COVID/2022/2025Q1 with zero recalibration; dominates fractional-Kelly.
> - **2026-07-18 — the ceiling survives everything; costs survived too** ([`docs/2026-07-18-remaining-items-results.md`](docs/2026-07-18-remaining-items-results.md)):
>   five parallel attacks on the prediction side (finer analyst endpoints, co-coverage links, rank
>   targets, FinBERT encoder — all lit-sweep-recommended) returned **zero usable IC**; the encoder arm
>   replicated the published large-cap null (embeddings ≤ shuffled placebo). The risk side delivered
>   again: **buy/hold banding + RU-CVaR survives 3bps costs at net Sharpe 0.73 with the tail still
>   pinned** (0.469% vs 0.5% target). ⚠️ half the net alpha needs same-close (MOC) execution.
> - **2026-07-18 (later) — PIT bounding test: the alpha was universe selection** ([`docs/2026-07-18-pit-bounding-test.md`](docs/2026-07-18-pit-bounding-test.md)):
>   on a point-in-time S&P membership universe (~472 names/day, free data) the decade LS goes
>   0.72 → 0.37 (∩ membership) → **−0.27** (full PIT cross-section); combo decade IC +0.012 → **+0.004**.
>   Not a delisting artifact (gap persists at 96-99% coverage): "today's top-150" is a look-ahead
>   winner filter and the combo only ranks within it. **Alpha layer: dead on an honest universe.
>   Risk layer: intact** (CVaR pinned even on the dead stream). Norgate only needed to *defend* a
>   PIT top-150-by-cap variant, not for this verdict.
> - **2026-07-18 (final batch) — the controller is the product** ([`docs/2026-07-18-free-reprocessing-batch.md`](docs/2026-07-18-free-reprocessing-batch.md)):
>   5 more free experiments. RU-conformal generality PROVEN: **8 loss streams, 2 targets, 10/10
>   pins, year-by-year (2020: static 3.74% → 1.02%), Sharpe ≥ static everywhere, maxDD −3–10×**;
>   spec'd weakness = crash-burst lag (the lit's regime-weighted fix passes only with in-sample
>   tuning — vanilla stays). Alpha side: broker-event drift null (all info in day 0), co-coverage
>   structurally closed (S&P bottom-tercile graph still 88–98% dense), one ~0.01 curiosity
>   (industry leader-follower, t≈3–4, 10/10 yrs positive on the PIT universe). Free-data alpha
>   work on this universe is exhausted.

1. **Prediction ≠ profit — and "reward optimized" was overclaimed.** The bull-window Sharpes
   (incl. graded's 0.93) are largely a *long-bias dividend* of a rising market — in the flat
   2025-H1 slice every LLM loses, and graded loses −4.3 % *despite a higher IC there*. And
   (Jiwoong's reward audit, [`docs/2026-06-30-jiwoong-reward-rebuttal.md`](docs/2026-06-30-jiwoong-reward-rebuttal.md))
   even the *reward* wasn't cleanly optimized: under the original 5×5 decision matrix the
   models' **mean reward stays negative and below a constant-policy baseline** (e.g. graded
   −0.28 vs best-const −0.06). So a positive IC/Sharpe ≠ "the model mastered the reward."
   Judge skill by IC, by mean-reward-vs-baseline, and by net-of-baseline returns across
   regimes — not raw bull Sharpe.
2. **There is a ~0.24 IC ceiling on the *tabular / quantified* input** (price + technical
   indicators + crude news *counts*) — i.e. the limit of what a strong **GBM** can extract.
   > ⚠️ **Refined 2026-07-03** (see the update box above): 0.24 is GBM's number *on the smoothed
   > proxy*; a linear momentum model reaches 0.266 (GBM under-extracts), and the *tradeable*
   > raw-return ceiling is **~0.06**. Read this as "GBM on the proxy," not the information limit.
   Four independent checks say *that* limit is the input, not the model: **(a) no overfit** —
   in-sample IC ≈ out-of-sample IC (can't beat 0.2 even on training data); **(b)
   feature-invariant** — 3× more features + cross-sectional + market-neutral target stay
   ~0.23; **(c) ensemble-null** — an LLM⊕GBM blend (corr 0.62) doesn't beat the GBM alone;
   **(d) oracle** — perfect labels print MDD 2.73 % (= the paper), so the drawdown gap is a
   *prediction* limit, not sizing.
   > **Scope caveat (important).** This ceiling is on *quantified* inputs. It does **not**
   > bound the paper's actual lever — an **LLM reasoning over raw news *text* + technicals**
   > (the paper cats them into a 15–23k-token prompt; a GBM can't read text and counts throw
   > the content away). **Tested as far as our data allows** (`MM_RICH`: every headline in a
   > 60d window, 60–250 per prompt, ~3–8k tok — the max headline-only data permits): the
   > rich-text prompt-only IC is **0.187 — still *below* price-only's 0.221** (zero-shot LLM
   > gets *distracted* by more text, not helped). Our prior *trained* multimodal also landed
   > ~0.19. So at our scale the news-text lever does **not** break the ceiling. The one thing
   > we genuinely can't test is the paper's full **article bodies** at 15–23k tok — our news
   > is **headlines-only** (Google-RSS); that, and a far larger universe, are the only
   > untested differences left.
3. **The cap is horizon- and cost-bound, not model-bound.** Shortening the label horizon
   to 2–5 d lifts achievable IC to 0.33, but the faster rebalancing it needs gives the
   gain back to transaction costs — net Sharpe peaks at the current 3/7/15-d weekly.
4. **A plain GBM beats every LLM here** (on tabular technical prediction the LLM under-uses
   precise numerics; GBM IC 0.24 / net SR 1.14 > every trained LLM). **But the GBM is only a
   valid proxy for the *input ceiling*, not for the LLM** — rigorously tested
   ([`docs/2026-06-30-gbm-llm-proxy-validity.md`](docs/2026-06-30-gbm-llm-proxy-validity.md)):
   GBM IC ≥ every LLM IC ✅ (so "input caps at 0.24" is a fair GBM claim), but it overestimates
   each LLM's IC (gap varies 3×), agrees with LLM decisions only ~30 %, and is right on 16–26 %
   of rows the LLM gets wrong — i.e. an *independent* model. So a GBM learning curve bounds the
   *signal*, **not the LLM's data-response** — that needs the LLM trained at several sizes.
   (RL narrows the gap: SFT extracts ~60 % of the ceiling, GRPO ~88 %.)
5. **The multimodal signal was news-driven.** It looked like an orthogonal regime-hedge
   (CV IC ~0.12, positive in 2025-H1) but a news-less walk-forward on fresh 2025-H2→2026
   data scores ~0.02 — and `news.parquet` is the one stale feed (ends 2025-06). Reviving it
   is **free** (extend `crawl_news.py`'s `MONTHS` range and re-run — Google-News-RSS, no key).
6. **vs the paper:** on its own window our best GRPO matches Sharpe (1.51 vs 1.57) but the
   **drawdown is 2× worse** (5.5 % vs 2.8 %) — the real gap is risk/regime control (= the
   oracle's prediction limit in 2d, not sizing).

**Net:** the simplest defensible model (SFT v1) is the most defensive; graded-reward GRPO
gives the best bull-window return; a non-LLM GBM is the best *tabular* predictor. The binding
constraint for *quantified* inputs is their predictive content (IC ~0.24) and transaction
cost — not reward shape, RL depth, model size, or hardware. **The one input lever not yet
fairly tested is the paper's: an LLM reasoning over full-scale raw news *text*** (see the
scope caveat above) — that reproduction is the current work.

## Reproduce a model's evaluation

```bash
# 1. serve a trained adapter as a vLLM LoRA, then point the backtest at it.
#    cache key = snapshot hash (model-agnostic) → each model needs its own cache dir.
VLLM_MODEL=sft-v1 VLLM_CACHE_DIR=compare_lab/.cache_sftv1 \
  uv run python -m compare_lab.run_comparison --llm --out compare_lab/output_sftv1

# 2. honest-lens analysis from cached decisions (no GPU):
uv run python -m compare_lab.eval_labels   .cache_sftv1 .cache_graded_full  # label-fidelity IC
uv run python -m compare_lab.compare_paper .cache_graded_full               # 1:1 vs the paper
uv run python -m compare_lab.make_report --out docs/2026-06-29-results-report.html
```

Trained adapters live in `data/sft_adapter_v{0,1,2}/` + the GRPO outputs (gitignored).
Training code: `compare_lab/sft/` (LoRA SFT, distillation) and `compare_lab/grpo/`
(TRL GRPOTrainer on DGX Spark GB10; rewards in `grpo/rewards.py`). Build the targets with
`labeling.py` (`make_signal` / 5-class labels).

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

## Status & next

Comparison substrate + baselines + prompt-only LLM ✅ done; the SFT→GRPO ladder is
trained and audited (see [Models](#models-at-a-glance)). The open lever is the
**input's predictive content** — every other knob (reward, RL depth, model size,
multimodal-as-fed) is capped by the IC ceiling. Candidate next steps: a fresh **news**
pull to revive the multimodal signal (free — extend `crawl_news.py`'s `MONTHS` range past
2025-06 and re-run; Google-News-RSS, no key), or productionising the GBM signal.
Design docs in `docs/superpowers/specs/`; full chronological log in
[`docs/PROGRESS-2026-06-21.md`](docs/PROGRESS-2026-06-21.md).

## Testing

```bash
uv run python -m pytest -q   # 70 passing
```
