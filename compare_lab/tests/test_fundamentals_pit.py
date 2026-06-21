"""Tests for fundamentals revenue-tag normalization.

The delivered fundamentals use two XBRL revenue tags — `Revenues` (total) and
`RevenueFromContractWithCustomerExcludingAssessedTax` (ASC 606 contract revenue).
Some filings carry both. We collapse them to one canonical `Revenue` line per
(ticker, period_end, filing_date, fiscal_period), preferring `Revenues` (the
total, a superset), and leave every non-revenue concept untouched. PIT stays on
`filing_date`.
"""
from __future__ import annotations

import pandas as pd

from compare_lab.fundamentals_pit import (
    CANONICAL_REVENUE,
    REVENUE_TAGS,
    normalize_revenue,
)

_CONTRACT = "RevenueFromContractWithCustomerExcludingAssessedTax"


def _row(ticker, concept, value, fp="2023Q2"):
    return {"ticker": ticker, "concept": concept, "value": value,
            "fiscal_period": fp, "period_end": pd.Timestamp("2023-06-30"),
            "filing_date": pd.Timestamp("2023-07-30"), "form": "10-Q",
            "unit": "USD"}


def test_single_tag_tickers_keep_their_value_as_revenue():
    df = pd.DataFrame([_row("AAPL", _CONTRACT, 100), _row("NVDA", "Revenues", 200)])
    out = normalize_revenue(df)
    aapl = out[out.ticker == "AAPL"].iloc[0]
    nvda = out[out.ticker == "NVDA"].iloc[0]
    assert aapl["concept_normalized"] == CANONICAL_REVENUE and aapl["value"] == 100
    assert nvda["concept_normalized"] == CANONICAL_REVENUE and nvda["value"] == 200


def test_conflict_prefers_total_revenues():
    df = pd.DataFrame([_row("XOM", _CONTRACT, 150), _row("XOM", "Revenues", 160)])
    out = normalize_revenue(df)
    xom = out[(out.ticker == "XOM") & (out.concept_normalized == CANONICAL_REVENUE)]
    assert len(xom) == 1
    assert xom.iloc[0]["value"] == 160  # Revenues (total), not the 150 contract line


def test_non_revenue_concepts_untouched():
    df = pd.DataFrame([_row("AAPL", "Assets", 999), _row("AAPL", _CONTRACT, 100)])
    out = normalize_revenue(df)
    assets = out[out.concept == "Assets"]
    assert len(assets) == 1
    assert assets.iloc[0]["concept_normalized"] == "Assets"
    assert assets.iloc[0]["value"] == 999


def test_real_file_collapses_revenue_and_preserves_filing_date():
    f = pd.read_parquet("data/qflib_data_store/fundamentals.parquet")
    out = normalize_revenue(f)
    rev = out[out["concept_normalized"] == CANONICAL_REVENUE]
    key = ["ticker", "period_end", "filing_date", "fiscal_period"]
    # at most one canonical Revenue row per filing-period
    assert rev.groupby(key).size().max() == 1
    # non-revenue rows are unchanged in count
    n_nonrev_in = int((~f["concept"].isin(REVENUE_TAGS)).sum())
    n_nonrev_out = int((~out["concept"].isin(REVENUE_TAGS) &
                        (out["concept_normalized"] != CANONICAL_REVENUE)).sum())
    assert n_nonrev_in == n_nonrev_out
    assert out["filing_date"].notna().all()
