"""Normalize the two revenue XBRL tags into one canonical `Revenue` line.

The delivered `fundamentals.parquet` reports revenue under two concepts:
`Revenues` (total) and `RevenueFromContractWithCustomerExcludingAssessedTax`
(ASC 606 contract revenue). Most tickers use one; some filings carry both, where
`Revenues` is the superset (it includes non-contract revenue — insurance
investment income, energy other-income, etc.). We collapse revenue to a single
`Revenue` concept per (ticker, period_end, filing_date, fiscal_period),
**preferring `Revenues`**, and add a `concept_normalized` column to every row
(unchanged for non-revenue concepts). PIT stays on `filing_date`.

    uv run python -m compare_lab.fundamentals_pit      # writes *_pit.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REVENUE_TAGS: tuple[str, ...] = (
    "Revenues",
    "RevenueFromContractWithCustomerExcludingAssessedTax",
)
CANONICAL_REVENUE = "Revenue"
# lower rank wins on conflict — Revenues (total) preferred over contract revenue
_PREF = {"Revenues": 0, "RevenueFromContractWithCustomerExcludingAssessedTax": 1}
_KEY = ["ticker", "period_end", "filing_date", "fiscal_period"]

_STORE = Path(__file__).resolve().parents[1] / "data" / "qflib_data_store"


def normalize_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """Collapse the two revenue tags to one `Revenue` line; tag every row with
    `concept_normalized`."""
    out = df.copy()
    out["concept_normalized"] = out["concept"]
    is_rev = out["concept"].isin(REVENUE_TAGS)
    out.loc[is_rev, "concept_normalized"] = CANONICAL_REVENUE

    rev = out[is_rev].copy()
    rev["_pref"] = rev["concept"].map(_PREF)
    rev = (rev.sort_values("_pref")
              .drop_duplicates(subset=_KEY, keep="first")
              .drop(columns="_pref"))

    result = pd.concat([out[~is_rev], rev], ignore_index=True)
    return result.sort_values(["ticker", "period_end", "concept_normalized"]) \
                 .reset_index(drop=True)


def main() -> int:
    src = _STORE / "fundamentals.parquet"
    f = pd.read_parquet(src)
    out = normalize_revenue(f)
    dst = _STORE / "fundamentals_pit.parquet"
    out.to_parquet(dst, index=False)
    n_rev = int((out["concept_normalized"] == CANONICAL_REVENUE).sum())
    print(f"rows {len(f)} -> {len(out)}  (revenue rows collapsed to {n_rev})")
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
