"""Tests for recovering the insider transaction type from the `text` field.

The delivered sentiment_insider.parquet has an empty `transaction` column; the
type lives in free text ("Sale at price ...", "Stock Award(Grant) ...", ...).
We parse it to a canonical type and a coarse buy/sell/neutral direction (only an
open-market Purchase is genuinely bullish; grants/gifts/exercises are noise).
"""
from __future__ import annotations

import pandas as pd

from compare_lab.insider_pit import (
    DIRECTION,
    enrich_insider,
    parse_transaction,
)


def test_parse_known_phrasings():
    cases = {
        "Sale at price 215.73 per share.": "SALE",
        "Sale at price 217.66 - 222.38 per share.": "SALE",
        "Purchase at price 100.00 per share.": "PURCHASE",
        "Stock Award(Grant) at price 0.00 per share.": "GRANT",
        "Stock Gift at price 0.00 per share.": "GIFT",
        "Conversion of Exercise of derivative security at price 50 per share.": "EXERCISE",
    }
    for text, want in cases.items():
        assert parse_transaction(text) == want, (text, parse_transaction(text))


def test_empty_or_missing_is_unknown():
    assert parse_transaction("") == "UNKNOWN"
    assert parse_transaction("   ") == "UNKNOWN"
    assert parse_transaction(None) == "UNKNOWN"
    assert parse_transaction(float("nan")) == "UNKNOWN"


def test_direction_only_purchase_is_buy():
    assert DIRECTION["PURCHASE"] == "BUY"
    assert DIRECTION["SALE"] == "SELL"
    for t in ("GRANT", "GIFT", "EXERCISE", "UNKNOWN"):
        assert DIRECTION[t] == "NEUTRAL"


def test_enrich_real_file_fills_every_row():
    si = pd.read_parquet("data/qflib_data_store/sentiment_insider.parquet")
    out = enrich_insider(si)
    assert out["txn_type"].notna().all()
    assert set(out["txn_type"].unique()).issubset(set(DIRECTION))
    assert set(out["direction"].unique()).issubset({"BUY", "SELL", "NEUTRAL"})
    # known counts from the text distribution
    vc = out["txn_type"].value_counts()
    assert vc.get("SALE", 0) == 538
    assert vc.get("PURCHASE", 0) == 15
    assert vc.get("UNKNOWN", 0) == 290
