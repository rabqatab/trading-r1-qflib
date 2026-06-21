"""Recover the insider transaction type from the free-text `text` field.

The delivered `sentiment_insider.parquet` ships an empty `transaction` column;
the type is only in `text` ("Sale at price ...", "Stock Award(Grant) ...",
"Conversion of Exercise ...", "Purchase ...", "Stock Gift ..."). We map it to a
canonical `txn_type` and a coarse `direction`.

Signal note: only an open-market **Purchase** is a genuinely bullish insider
signal. Sales are noisy (diversification, taxes, 10b5-1 plans); grants, gifts,
and option exercises carry essentially no directional information — hence they
map to NEUTRAL, not SELL.

    uv run python -m compare_lab.insider_pit      # writes *_pit.parquet
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

# canonical type -> coarse direction
DIRECTION: dict[str, str] = {
    "PURCHASE": "BUY",
    "SALE": "SELL",
    "GRANT": "NEUTRAL",
    "GIFT": "NEUTRAL",
    "EXERCISE": "NEUTRAL",
    "UNKNOWN": "NEUTRAL",
}

_STORE = Path(__file__).resolve().parents[1] / "data" / "qflib_data_store"


def parse_transaction(text) -> str:
    """Map an insider `text` description to a canonical transaction type."""
    if text is None:
        return "UNKNOWN"
    if isinstance(text, float) and pd.isna(text):
        return "UNKNOWN"
    t = str(text).strip().lower()
    if not t:
        return "UNKNOWN"
    if t.startswith("sale"):
        return "SALE"
    if t.startswith("purchase"):
        return "PURCHASE"
    if "award" in t or "grant" in t:
        return "GRANT"
    if "gift" in t:
        return "GIFT"
    if "exercise" in t or "conversion" in t:
        return "EXERCISE"
    return "UNKNOWN"


def enrich_insider(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with `txn_type` and `direction` columns filled from text."""
    out = df.copy()
    out["txn_type"] = out["text"].map(parse_transaction)
    out["direction"] = out["txn_type"].map(DIRECTION)
    return out


def main() -> int:
    src = _STORE / "sentiment_insider.parquet"
    out = enrich_insider(pd.read_parquet(src))
    dst = _STORE / "sentiment_insider_pit.parquet"
    out.to_parquet(dst, index=False)
    print("txn_type:", out["txn_type"].value_counts().to_dict())
    print("direction:", out["direction"].value_counts().to_dict())
    print(f"wrote {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
