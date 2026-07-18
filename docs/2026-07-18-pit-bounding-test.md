# PIT bounding test — the alpha was universe selection (2026-07-18)

> The free survivorship bounding re-test from [[2026-07-18-remaining-items-results]] §6.
> **Verdict: the decade LS result does not survive a point-in-time universe.** Gross Sharpe 0.72
> (today's top-150) → 0.37 (∩ PIT membership) → **−0.27** (full PIT still-listed cross-section).
> The combo's decade IC drops +0.012 → **+0.004**. The CVaR-control claim survives (it pins risk
> on a dead signal too); the alpha claim does not. **Norgate is NOT needed to reach this verdict.**

## Setup

- Universe: every S&P 500 member 2016-07→2026-05 (716 raw tickers → 566 canonical after ~25 rename
  mappings: FB→META, BLL→BALL, WLTW→WTW, ANTM→ELV, ABC→COR, CTL→LUMN, BBT→TFC, …).
  **81 true holes** (TWTR, ATVI, CELG, MON, XLNX, AGN, SIVB, FRC + 2025-26 M&A: WBA, ANSS, DFS,
  HES, JNPR, MRO, …) — no free price history.
- Prices: yfinance backfill → `data/yf_prices_sp500/prices.parquet` (485 tickers, 1.32M rows).
  Signals: same combo (mom10 + rec-revision + PEAD); Finnhub recs+earnings backfilled for the
  extras (635 files each now in the existing layouts). Names enter the cross-section only when
  PIT-member with data. Same LS/backtest conventions as [[2026-07-17-cvar-conformal-control]];
  arm A reproduces the known 0.72 (assert).
- Coverage (members-with-data/members): 87.6% (2017) rising to 99.3% (2026); arm C ≈ 472 names/day.

## Results

| arm | ann.ret | Sharpe | maxDD | 17-19 / 20-22 / 23-26 Sharpe |
|--|--:|--:|--:|--|
| A: top-150 as-is | +7.43% | **0.72** | 24.7% | −0.02 / 0.67 / 1.09 |
| B: 150 ∩ PIT membership | +2.94% | **0.37** | 23.4% | 0.32 / 0.01 / 0.82 |
| C: full PIT still-listed (~472/day) | −1.61% | **−0.27** | 29.7% | −0.63 / −0.63 / 0.43 |

- **Combo decade IC on C: +0.004** (vs +0.012 on A); per-year it is ~0 everywhere except 2026
  (+0.048). The 2024-25 "good years" on A (+0.034/+0.031) vanish on C (−0.005/+0.006).
- **Cost-realistic headline on C** (band 20/35, 3bps + borrow, RU-conformal): net Sharpe −0.30 →
  controlled **−0.15**, maxDD 14.1%, realized CVaR 0.364% — the controller still pins the tail;
  there is simply no alpha to control.
- **Bootstrap hole band** (50 draws, per-year matching drop): C Sharpe 5/50/95% = −0.32/−0.25/−0.18.
  Thinning noise is ~±0.07 — the 81 delisted holes cannot explain the ~1.0 A→C swing.

## Why this is decisive without Norgate

1. **The gap is not a missing-delisted-data artifact.** In 2023-26 coverage is 96-99% (essentially
   no holes) and C still runs at 0.43 vs A's 1.09. The inflation lives in *which names are in A*:
   "today's top-150 by size" is a look-ahead winner filter, and the combo signal is additionally
   mega-cap-specific (it ranks well *within* the winner set, not within the honest cross-section).
2. Classic delisting survivorship is the smaller half: PIT membership alone (B, still conditioned
   on today's 150) already halves the Sharpe; expanding to the honest opportunity set (C) does the
   rest. Bias direction of the remaining holes is net-unknown but bounded ~±0.1 by the band.
3. **Norgate ($346.50/6mo) is only needed for one thing now**: constructing "top-150 by trailing
   market cap, point-in-time" — the *defensible large-cap variant* of A that free data cannot build
   (no PIT caps, no delisted prices). Buy it only if we want to positively defend a large-cap-only
   strategy; the honest-universe verdict is already in.

## Standing after this test

- **Alpha layer: dead on an honest universe.** The combo's 0.096 (2025-H1) → decade +0.012
  (multi-year OOS) → **+0.004 (PIT universe)**. Each honesty upgrade removed most of what remained.
- **Risk layer: intact.** RU-conformal pinned realized CVaR on stream A (0.487%), on net-of-cost
  streams (0.469%), and on the dead PIT stream (0.364% — clipped exposure on a negative-drift
  stream). The control claim is loss-stream-agnostic, as the theory says.
- Open items now: (a) Norgate purchase — only if a PIT top-150-by-cap variant is wanted;
  (b) the MOC execution dependency (intraday data) — moot unless (a) revives an alpha claim.

Scripts: `compare_lab/fetch_pit_prices.py`, `compare_lab/fetch_pit_finnhub.py`,
`compare_lab/pit_bounding_backtest.py`. Data (gitignored): `data/yf_prices_sp500/` (+holes.json),
recs/earnings extras merged into existing `data/finnhub_recs/`, `data/finnhub_earnings_full/`.
