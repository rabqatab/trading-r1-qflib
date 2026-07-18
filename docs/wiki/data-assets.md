---
title: "Data assets — what exists, where, and what it can support"
status: ledger
updated: 2026-07-18
---

# Data assets

All under `data/` (gitignored). Free-tier or already-licensed Finnhub; FINNHUB_KEY in `.env`
(never printed/committed). Reference docs: [DATA_STORE](../DATA_STORE.md),
[DATA_REQUIREMENTS](../DATA_REQUIREMENTS.md), [DATA_QC_RUBRIC](../DATA_QC_RUBRIC.md).

| asset | location | coverage | supports |
|--|--|--|--|
| Top-150 prices | `qflib_data_store_top150/prices_top150.parquet` | 2015→2026-05 | all top-150 backtests |
| PIT S&P prices | `yf_prices_sp500/prices.parquet` (+`volume.parquet`, `holes.json`) | 485 tickers, 2015→2026-05; 81 delisted holes | PIT-universe backtests |
| S&P constituents | `sp500_constituents/` | membership 1996→2026-06 | PIT membership masks |
| Broker rec events | `finnhub_upgrade_downgrade/` | 635 tickers, 185k dated events, 2013→ | event studies, co-coverage graphs |
| Analyst recs (consensus) | `finnhub_recs/` | 635 tickers | rev signal |
| Earnings calendar | `finnhub_earnings_full/` | 635 tickers, 2016→ | PEAD |
| Sector profiles | `finnhub_profiles/` | 635 tickers (42 coarse industries) | industry groupings |
| Supply-chain graph | `finnhub_supply_chain/` | top-150, snapshot (⚠️ no history — look-ahead composition) | upper-bound tests only |
| News + FinBERT embeddings | `news_top150_summ.parquet`, `qflib_data_store_top150/finbert_embed_daily_top150.parquet` | 150k items, 2024-11→2025-07 | encoder experiments |
| 14-eq multimodal store | `qflib_data_store/` (6 PIT parquets) | see [DATA_STORE](../DATA_STORE.md) | original paper-scale runs |

**Known inability list** (what this data CANNOT support — see
[measurement-traps](measurement-traps.md) #6):
- Point-in-time estimate revisions (Finnhub eps/revenue-estimate & price-target = snapshots).
- Delisted-name prices (81 holes; needs Norgate).
- PIT market caps / top-N-by-cap universes (needs Norgate).
- Analyst-level (vs broker-level) identity for co-coverage.
- Intraday anything.
