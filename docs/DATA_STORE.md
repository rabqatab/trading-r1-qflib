# Data Store — multi-modal point-in-time dataset

> **Location:** `data/qflib_data_store/` (gitignored — never committed).
> **Received:** 2026-06-21 (`qflib_data_store.zip`, archived in `data/`).
> **Universe:** the paper's 14-ticker set (NVDA MSFT AAPL META AMZN TSLA BRK-B
> JPM LLY JNJ XOM CVX SPY QQQ). Non-price modalities cover the **12 equities**
> only (ETFs have no fundamentals/news/insider/analyst — expected).

This is the multi-modal input the LLM work needs (paper §2.1): the prompt-only
LLM (#2) and Trading-R1 (#1) reason over **news + fundamentals + sentiment +
macro**, not just price+technical. Every non-price file carries a real
publish/filing timestamp, so we can filter to "available as of trading day *t*".

## Files

| File | Rows | Modality | Source | PIT timestamp |
|---|---|---|---|---|
| `prices.parquet` | 40,348 | technical (price) | Yahoo (`auto_adjust`) | `date` |
| `news.parquet` | 11,886 | news | Google News RSS | `published_at` |
| `fundamentals.parquet` → **`fundamentals_pit.parquet`** | 2,156 | fundamentals | SEC EDGAR (XBRL) | `filing_date` (✅ revenue-normalized) |
| `sentiment_analyst.parquet` | 7,319 | sentiment (analyst) | Yahoo upgrades/downgrades | `gradedate` |
| `sentiment_insider.parquet` | 1,332 | sentiment (insider) | Finnhub insider txns | `start_date` |
| `macro.parquet` → **`macro_pit.parquet`** | 5,761 | macro | FRED | `release_date` (✅ leak-fixed) |

### `prices.parquet` — 14 tickers, 2015-01-02 → 2026-06-18
`date, ticker, Open, High, Low, Close, Volume, raw_close, dollar_volume`.
`Close` is split/dividend-adjusted; **`raw_close` is unadjusted** (new vs the
harness set — useful for label sanity and as-traded price). Includes SPY/QQQ.
**This supersedes the manual SPY/QQQ merge from task D** — it is the cleaner,
complete 14-ticker price set.

### `news.parquet` — 12 equities, 2024-01-01 → 2025-06-30 (18 mo)
`ticker, date, published_at, headline, source, url, url_hash`. **Headlines
only** (no article body). 571–1,656 items/ticker (NVDA/TSLA/AMZN richest).
Sources: Yahoo Finance, Motley Fool, Barron's, CNBC, Investopedia. The 18-month
window matches the paper's training span. `published_at` has intraday times →
clean PIT. *Quotability:* headline + source + date are present (enough for the
Stage-II evidence reward); article snippets are not.

### `fundamentals.parquet` — 12 equities, filings 2023-07 → 2026-05
Long format: `ticker, concept, value, fiscal_period, period_end, filing_date,
form, unit`. 8 concepts: Assets, CashAndCashEquivalents, Liabilities,
NetIncomeLoss, OperatingIncomeLoss, **Revenues** *and*
**RevenueFromContractWithCustomerExcludingAssessedTax** (two XBRL revenue tags —
**normalize to one revenue line** per ticker), StockholdersEquity. Forms: 10-Q
(1,542) + 10-K (614). `filing_date` = when it became public → use that, **not**
`period_end`, for PIT filtering.

### `sentiment_analyst.parquet` — 12 equities, 2011 → 2026
`gradedate, firm, tograde, fromgrade, action, pricetargetaction,
currentpricetarget, priorpricetarget, ticker`. Analyst rating changes + price
targets. `action ∈ {init, main, up, down, reit, …}`. Quotable (firm + to/from
grade + date).

### `sentiment_insider.parquet` — 12 equities, 2024-06 → 2026-06
`shares, value, url, text, insider, position, transaction, start_date,
ownership, ticker`. Insider transactions with a quotable `text`
("Stock Award(Grant) at price 0.00 per share."). `start_date` = disclosure/txn
date.

### `macro.parquet` — 8 series, 2022-01 → 2026-06
`series, date, value, release_date`. Series: **CPIAUCSL** (CPI), **UNRATE**
(unemployment), **FEDFUNDS**, **DGS2 / DGS10** (2y/10y Treasury), **T10Y2Y**
(10y–2y spread), **DEXUSEU** (USD/EUR), **VIXCLS** (VIX).

## Data-quality issues

1. **macro `release_date` == `date` (PIT LEAK) — ✅ FIXED.** FRED publishes with
   a lag (January CPI is released mid-February), but the delivered file set the
   release date to the reference-period date. **Fix:** `compare_lab/macro_pit.py`
   (`correct_release_dates`) rebuilds a leak-safe `release_date` — monthly series
   (CPI/UNRATE/FEDFUNDS) → conservative day of M+1 (15th/8th/3rd, at/after the
   real schedule), daily series → next business day. Output:
   **`macro_pit.parquet`** (0 leaks; monthly shifted ~38 d, daily ~1 d).
   **Use `macro_pit.parquet`, not `macro.parquet`,** for any snapshot. Unit-tested
   (`compare_lab/tests/test_macro_pit.py`). The exact fix (FRED/ALFRED vintage
   dates) needs an API key and is a later upgrade.
2. **insider `transaction` field empty for all 1,332 rows — ✅ FIXED.** The type
   lives in `text`; `compare_lab/insider_pit.py` (`parse_transaction`,
   `enrich_insider`) recovers a canonical `txn_type` and a coarse `direction`,
   written to **`sentiment_insider_pit.parquet`**. Distribution: SALE 538,
   GRANT 294, UNKNOWN 290, GIFT 114, EXERCISE 81, **PURCHASE 15** → direction
   NEUTRAL 779 / SELL 538 / **BUY 15**. Only open-market **Purchase** maps to BUY
   (grants/gifts/exercises carry no directional signal). Unit-tested
   (`compare_lab/tests/test_insider_pit.py`).
3. **Two revenue concepts in fundamentals — ✅ FIXED.** `Revenues` (total) and
   `RevenueFromContractWithCustomerExcludingAssessedTax` (ASC 606); some filings
   carry both (`Revenues` is the superset). `compare_lab/fundamentals_pit.py`
   (`normalize_revenue`) collapses them to one canonical **`Revenue`** line per
   (ticker, period_end, filing_date, fiscal_period), preferring `Revenues`, and
   adds a `concept_normalized` column (non-revenue concepts unchanged). Output:
   **`fundamentals_pit.parquet`** (2,156→2,075 rows; 312 Revenue rows;
   verified in all 81 conflicts `Revenues ≥ contract`). Unit-tested
   (`compare_lab/tests/test_fundamentals_pit.py`).

## How this plugs into the pipeline — ✅ implemented

`compare_lab/multimodal_context.py` (`MultiModalStore`) loads the `*_pit.parquet`
files and exposes per-`(ticker, as_of)` accessors that filter strictly on each
modality's own timestamp, then `render_sections()` emits a compact text block:
- news with `published_at <= as_of` (last 30 d),
- latest fundamentals (`fundamentals_pit.parquet`) with `filing_date <= as_of`,
- analyst (`gradedate`) / insider (`sentiment_insider_pit.parquet`, `start_date`)
  within 90 d `<= as_of`,
- macro (`macro_pit.parquet`) latest per series with `release_date <= as_of`.

`MarketSnapshotBuilder(ctx, multimodal=MultiModalStore())` appends these sections
to the price+technical snapshot (opt-in; price-only remains the default). Every
join uses the PIT timestamp, never the reference/period date — the single most
important rule (a future-dated row silently invalidates the backtest). Tests:
`test_multimodal_context.py` (per-modality PIT) + `test_snapshot.py` (wiring).
ETFs (SPY/QQQ) degrade gracefully — company sections render "none".
