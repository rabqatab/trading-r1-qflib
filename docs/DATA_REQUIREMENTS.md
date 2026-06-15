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
