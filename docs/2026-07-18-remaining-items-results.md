# Remaining items — parallel execution results (2026-07-18)

> All six remaining items from [[2026-07-17-remaining-items-lit-sweep]] executed in one parallel
> batch (5 experiments + 1 scoping). **Score: 1 positive (cost gate PASSED), 3 negative, 1 blocked,
> 1 scoped.** Every prediction-side lever came back null/negative; the only thing that worked —
> again — is portfolio mechanics. The two-layer conclusion is now exhaustively confirmed.

## Scoreboard

| # | item | verdict | headline number |
|--|--|--|--|
| 1 | Cost haircut + turnover reduction | ✅ **PASSED** | banding 20/35 → net Sharpe 0.63 @3bps; **+ RU-CVaR = 0.73, CVaR pinned 0.469%** |
| 2 | Estimate-revision breadth | 🚫 **BLOCKED** (untestable) + fallback negative | Finnhub estimates = snapshots, no PIT history; ud fallback IC +0.003, dilutes combo |
| 3 | Co-coverage momentum | ❌ negative | IC +0.003–0.004; dense-graph structural failure |
| 4 | Rank target + horizon sweep | ❌ negative | no arm beats mom +0.064 / regGBM +0.063; rank halves variance, adds no mean |
| 5 | LLM-encoder ablation | ❌ **clean null** | every text arm ≤ price baseline; FinBERT+PCA below shuffled placebo |
| 6 | PIT universe scoping | 📋 scoped | free data fixes membership half; delisted prices = blocker → Norgate $346.50/6mo |

## 1. Costs (gate) — PASSED, with one warning ✅

`compare_lab/cost_haircut_backtest.py` (reuses `cvar_conformal_backtest` signals; gross unbuffered
Sharpe reproduces 0.735 ≈ 0.72 known). Cost = one-way turnover × {0,2,3,5}bps + 35bps/yr borrow
on the short book; drifted-weight turnover.

| arm | turnover/day | Sharpe @3bps | maxDD @3bps |
|--|--:|--:|--:|
| unbuffered daily quintile | 0.171 | 0.59 | 24.1% |
| **band 20/35 (NMV)** | **0.089** | **0.63** | **19.3%** |
| ema3 / ema5 / ema10 | 0.113/0.094/0.073 | 0.55/0.44/0.34 | — |
| band+ema5 | 0.060 | 0.42 | 13.3% |
| JT overlap K=5 | 0.083 | 0.28 | — |

- **Banding is the only mitigation that works**: halves turnover at zero gross cost (0.71 vs 0.72
  gross). EMA/JT lag a fast signal (mom10/PEAD) — alpha decays faster than costs shrink, exactly
  the Novy-Marx-Velikov ranking.
- **RU-conformal on the banded NET stream @3bps: Sharpe 0.73, maxDD 8.79%, realized CVaR 0.469%**
  (target 0.5%) — the control claim survives costs; net-of-cost lands where gross static-λ was.
- ⚠️ **Execution-lag warning**: trading at close t+1 instead of same close drops net Sharpe
  0.63→0.31. Half the net alpha is same-close execution — MOC-style execution is load-bearing.

## 2. Estimate-revision breadth — the hypothesis is UNTESTABLE with our data 🚫

`compare_lab/revision_breadth_ic.py` + `fetch_upgrade_downgrade.py`. Feasibility probe first:
- `/stock/eps-estimate`, `/stock/revenue-estimate`: **snapshot-per-fiscal-period, no as-of dates,
  no up/down counts** → PIT revision breadth cannot be backfilled. `/stock/price-target`: pure
  current snapshot. The lit sweep's most-promising signal needs I/B/E/S-style history we don't have.
  **Untested, not refuted.**
- Fallback `ud` (broker upgrade−downgrade counts, 63d, scaled by coverage; from
  `/stock/upgrade-downgrade` which DOES have dated history 2013→): IC +0.003 overall, corr 0.54
  with the existing rev signal, and combo4 (+ud) = 0.010 < combo3 = 0.011 — **dilutes**. Do not add.
- Kept asset: `data/finnhub_upgrade_downgrade/` — 58,185 dated broker events, 313 brokers, 150/150
  tickers; still useful for event-study uses (day-of-upgrade drift), not as a smoothed factor.

## 3. Co-coverage momentum — structurally dead in a 150-large-cap universe ❌

`compare_lab/cocoverage_momentum_ic.py` + `fetch_cocoverage_data.py`. Broker-co-coverage graph
(tf-idf broker weights, trailing 24m, yearly), CF momentum k∈{5,10,21}, self-check asserts pass.
- Mean IC **+0.003–0.004** (raw and own-reversal-residualized alike), sign-flips 5 of 10 years.
- **Why (structural)**: within 150 mega-caps mean graph degree is 121/149 even after tf-idf — the
  "connected-firm return" is ≈ a market average. Ali-Hirshleifer's effect lives in sparse-coverage
  full-CRSP cross-sections; our universe choice kills it by construction.
- Bonus: our key's tier **includes `/stock/supply-chain`** → `data/finnhub_supply_chain/` (477
  in-universe edges). Supply-chain neighbor momentum: +0.009 — better, but the graph is a current
  snapshot (look-ahead composition) so that's an optimistic ceiling. Both ≪ mom/rev/PEAD.

## 4. Rank target + horizon sweep — negative; ranking = winsorization ❌

`compare_lab/rank_target_sweep.py` (HistGBM, gbm_ceiling features; native lambdarank skipped —
no xgboost/lightgbm in env; leak-free asserts pass).
- Canonical 2025-H1 (pooled 1000 pts): mom10 **+0.064**, regression GBM **+0.063** (expanding
  window beats the old +0.042), rank arms +0.012..+0.038. Walk-forward means (11 half-year
  windows 2021–26): mom10 +0.011 > rank_h3 +0.007 > reg +0.006. **No rank arm beats anything.**
- Best training horizon h=3 (directionally Label-Horizon-Paradox, but inside noise).
- The one real effect: rank labels halve window-to-window IC std (0.018 vs 0.039) **without
  raising the mean** — cross-sectional ranking acts as label winsorization, not signal extraction.
- Methodological note: the canonical pooled-1000-pt metric mixes time-series + cross-sectional
  variation and flatters everything (mom10 daily-CS in 2025H1 is actually −0.011).

## 5. LLM-encoder ablation — clean null, published null replicated ❌

`compare_lab/encoder_ablation.py` (+ `data/.../finbert_embed_daily_top150.parquet`, 150k news
items embedded with ProsusAI/finbert on CPU — sparkq not needed, ~27min after batch-sorting).
2025-H1 daily CS rank-IC: price-only **+0.051** | +sentiment scalar +0.044 | +supervised
projection +0.037 | **+shuffled-embedding placebo +0.035** | +FinBERT-PCA32 +0.024.
- **Every text arm ≤ baseline, and real embeddings ≤ noise embeddings** — the drag is pure tree
  dilution. Replicates the arXiv:2606.29290 large-cap FinBERT-PCA null; does NOT confirm the
  literature's +0.005–0.01 for supervised projections (caveat: only ~1.5 months of news-covered
  train data — rules out "this setup," not the general claim).
- Closes roadmap D: with end-to-end SFT/GRPO ≈ −0.01 (earlier) and encoder+head ≈ 0 (now), text
  adds nothing over 16 technical indicators on this universe/horizon by either route.

## 6. PIT universe — scoped; the $ decision is framed 📋

Free constituent history downloaded (`data/sp500_constituents/`, 1996→2026-06). Measured exposure:
- Jan-2017: only **89/150** of our names were S&P members; **28/150 never were** (ADRs/speculative);
  31/150 have no 2017 price data (later IPOs). **~65–75 of the true 2017 top-150 are missing.**
- Delisted-price blocker: yfinance recovers ~50% of index leavers (renames recoverable; true
  delistings — CELG, MON, TWX, ATVI... — gone), and those names are where the bias lives.
- **Recommended path**: (a) free *bounding* re-test (PIT membership ∩ still-listed, sensitivity
  bands for the holes) — if the decade Sharpe moves materially (likely: early years are where both
  the bias and the combo's weakness sit), (b) buy **Norgate US Platinum** ($346.50/6mo ≈ $58/mo,
  cheapest tier with delisted prices + PIT membership). → **user decision.**

## What this batch means

Five parallel attacks on the prediction side — finer analyst data, cross-firm links, target
redesign, text encoders — produced zero usable IC. The information-bound ceiling
([[2026-07-03-why-the-ceiling]], [[2026-07-07-final-synthesis]]) has now survived every lever the
2023–2026 literature suggested. Meanwhile the risk/cost side delivered again: the strategy
survives realistic costs (banded, net Sharpe 0.73 under CVaR control with the tail still pinned).

**The project's final open questions are now exactly two, both non-modelling:**
1. the free PIT bounding re-test → Norgate decision (survivorship gate);
2. the same-close execution dependency (half the net alpha) — measurable only with intraday data.

Scripts: `cost_haircut_backtest.py`, `revision_breadth_ic.py`, `fetch_upgrade_downgrade.py`,
`cocoverage_momentum_ic.py`, `fetch_cocoverage_data.py`, `rank_target_sweep.py`,
`encoder_ablation.py`. Data (all gitignored): `finnhub_upgrade_downgrade/`, `finnhub_supply_chain/`,
`sp500_constituents/`, `finbert_embed_daily_top150.parquet`.
