# Data Requirements

> **Purpose:** a procurement checklist a data-sourcing coworker can act on.
> What we already have, what we need next, and the exact coverage + access
> method per source. Grounded in the paper's collection rules
> ([`trading-r1-paper-summary.md`](trading-r1-paper-summary.md) §2).

## TL;DR — what to get

| Priority | Item | Why | Access | Who |
|---|---|---|---|---|
| **P0** | SPY, QQQ daily OHLCV | complete the 14-ticker paper universe | yfinance (free) | self-serve |
| **P0** | (optional) full ~7k US-equity OHLCV | broader cross-section than S&P 500 | yfinance (free) | self-serve |
| **P1** | **FRED API key** | macro modality | free key | coworker |
| **P1** | **Finnhub API key** | company news + insider sentiment | free tier / paid for history | coworker |
| **P1** | **SimFin API key** | fundamentals (income/balance/cashflow, TTM) | free / paid | coworker |
| **P1** | SEC EDGAR access | fundamentals (10-Q/10-K) | free, no key | self-serve |
| **P2** | News archive (Finnhub history or vendor) | the paper's news is 30-day lookback per trading day across 18 months | paid tier likely | coworker |

**P0** unblocks finishing the current comparison substrate (sub-project 1).
**P1/P2** are for the LLM work — the prompt-only LLM (#2) and the Trading-R1
reimplementation (#1, sub-project 2).

---

## What we already have

- **Daily OHLCV**, S&P 500 constituents (503 tickers), 2015-01-02 → 2026-06-12,
  via yfinance (`qf-lib-harness/data/prices.parquet`, gitignored).
- Derived **technical indicators** computed locally with `stockstats` (no
  external source needed) — paper Table S2 set (SMA/EMA, MACD, RSI, KDJ, CCI,
  ROC, ATR, Bollinger, ADX, MFI, …).

Gap: the dataset has **no ETFs**, so the paper's SPY/QQQ are currently dropped.

---

## Target universe & window (for the LLM work)

- **Universe (14):** NVDA, MSFT, AAPL, META, AMZN, TSLA, BRK-B, JPM, LLY, JNJ,
  XOM, CVX, SPY, QQQ (paper Table S3).
- **Training window:** 2024-01-01 → 2025-05-31 (18 months, paper) — **extend to
  the latest available** if the source allows; more is better.
- **Evaluation holdout:** keep at least 2024-06 → 2024-08 (paper) untouched, plus
  our longer OOS 2024-01 → 2026-04 for the qf-lib comparison.

---

## Per-source detail (paper §2.1)

### 1. News  — `Finnhub` company-news API (+ Google News fallback)
- **What:** company headlines/snippets per ticker.
- **Coverage:** 30-day lookback for each trading day, time-bucketed
  (t−3..t / t−10..t−4 / t−30..t−11). Over 18 months × 14 tickers this is a lot
  of history → **likely needs a paid Finnhub tier** (free tier has a short
  history limit). Confirm history depth before buying.
- **Access:** Finnhub API key. Google News scraping is a fallback but brittle
  and ToS-sensitive — prefer a licensed API.
- **Deliver with:** the **publish timestamp** of every item (mandatory for
  point-in-time; see below).

### 2. Technical (price) — `Yahoo Finance`
- **What:** daily OHLCV. **Have it** (yfinance). Just add SPY/QQQ.

### 3. Fundamentals — `SimFin` API + `SEC EDGAR` (10-Q / 10-K)
- **What:** income statement / balance sheet / cash flow, quarterly + annual +
  TTM; key line items.
- **Access:** SimFin API key (free tier exists; paid for full history/coverage);
  SEC EDGAR is free (no key) but needs a descriptive `User-Agent` and respects
  fair-access rate limits.
- **Deliver with:** the **filing/publish date** per figure (only data published
  on or before the trading day may be used).

### 4. Sentiment — `Finnhub` insider sentiment/transactions + `Yahoo` analyst recs
- **What:** insider sentiment (90-day, fields `change`, `mspr`), insider
  transactions (≤25 most recent), analyst upgrades/downgrades (90-day trailing).
- **Access:** Finnhub API key (same as news); Yahoo `upgrades_downgrades`.
- Note: the paper **excludes social media** (Twitter etc.) — low signal value.

### 5. Macro — `FRED` API
- **What:** macro series (rates, inflation, etc.), 2-year history, monthly
  (first-of-month).
- **Access:** free FRED API key.

---

## How each source is used in the paper (purpose & training stage)

Trading-R1 trains a reasoning LLM with a **3-stage easy-to-hard curriculum**
(each stage = SFT warm-start → GRPO RL fine-tuning → self-distill augmentation):

- **Stage I — STRUCTURE:** organize the analysis into well-formed sections
  (fundamentals / technical / news / sentiment / macro / conclusion).
- **Stage II — CLAIMS:** make each point *evidence-grounded* — every bullet must
  pair an opinion with a **verbatim quote** and a **named source**
  (anti-hallucination).
- **Stage III — DECISION:** emit a 5-class call (Strong Sell … Strong Buy),
  rewarded against a volatility-adjusted ground-truth label.

All five modalities are assembled into **one prompt per (ticker, trading day)** —
the *input context* the model reasons over in every stage. Beyond that shared
input role, each source has a specific job:

| Modality (source) | Purpose in the paper | Which model / training step it feeds |
|---|---|---|
| **Technical — price** (Yahoo) | Input for the technical-analysis section **and** the source of the **5-class label**: the volatility-adjusted target is computed deterministically from price EMA returns (Algorithm S1). | Input (all stages) · **Decision label & RL decision reward (Stage III)** · distillation target check |
| **Technical — indicators** (stockstats, derived) | Technical-analysis section content (MACD/RSI/Bollinger/…). | Input (all stages) |
| **News** (Finnhub / Google) | Evidence for the news section; **quotable** headlines with source + date. | Input · **Evidence reward (Stage II)** — opinion+quote+source grounding · distillation |
| **Fundamentals** (SimFin / SEC EDGAR) | Fundamental-analysis section; **quotable** financial line items with filing date. | Input · **Evidence reward (Stage II)** · distillation |
| **Sentiment** (Finnhub insider, Yahoo analyst) | Sentiment & analyst-coverage sections; insider sentiment/transactions, up/downgrades. | Input · **Evidence reward (Stage II)** · distillation |
| **Macro** (FRED) | Macro-context section (rates, inflation, …). | Input · Evidence reward (Stage II) · distillation |

**Two consequences for how the data must look:**

1. **Price quality is label quality.** The training *answers* (5-class labels)
   and the RL *decision reward* are derived from price. Clean, complete,
   split/dividend-adjusted daily prices for the full window are non-negotiable.
2. **Text modalities must be quotable, not pre-scored.** Stage II rewards a
   bullet only when it carries a *verbatim quote* and a *named source*. So we
   need the **actual text snippets + source name + date** — raw articles /
   filing excerpts / disclosure text — **not** a single sentiment number. A
   sentiment score alone cannot be grounded and is unusable for the evidence
   reward.

> The prompt-only LLM baseline (#2) uses the *same assembled input* but no
> training — it just reads the snapshot and emits a 5-class call. So the input
> modalities are needed even before any training, to make #2 a fair comparison.

## Hard requirement: point-in-time integrity

Every non-price item **must carry its real publish/filing timestamp**, and the
deliverable must let us filter to "available on or before trading day *t*."
This is the single most important constraint — any future-dated leakage silently
invalidates the entire backtest. Concretely:

- News → article publish datetime.
- Fundamentals → SEC filing date (not the fiscal period end).
- Analyst/insider → event/disclosure date.

## Preferred delivery format

- One record per **(ticker, date, modality, item)** with a `published_at` field,
  as Parquet or JSONL. Raw API dumps are fine too — we can normalize — as long
  as timestamps are preserved.
- A note on **licensing / redistribution** for each source (can we use it for
  internal research? store it? share derived features?).

## Cost / licensing to confirm

- Finnhub & SimFin paid tiers (news/fundamentals history depth) — get quotes.
- News redistribution terms (even internal research storage can be restricted).
- FRED & EDGAR are free and research-friendly.
