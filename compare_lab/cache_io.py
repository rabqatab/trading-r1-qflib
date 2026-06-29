"""Read cached vLLM replies into per-(ticker, date) 5-class decisions.

The response cache is one JSON per call: {"prompt": ..., "response": ...}. The
prompt carries `Ticker:` and `As of:`; the response ends with `[[[CLASS]]]`.
Shared by eval_labels / compare_paper / make_report.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

_TICKER = re.compile(r"^Ticker:\s*([A-Z.\-]+)", re.M)
_ASOF = re.compile(r"^As of:\s*(\d{4}-\d{2}-\d{2})", re.M)
_TAG = re.compile(r"\[\[\[\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)\s*\]\]\]")


def resolve(name_or_dir) -> Path:
    """'sftv1' | '.cache_sftv1' | a full path -> the cache directory."""
    p = Path(name_or_dir)
    if p.exists():
        return p
    base = Path(__file__).resolve().parent
    return base / (p.name if p.name.startswith(".cache_") else f".cache_{p.name}")


def read_decisions(cache_dir) -> pd.DataFrame:
    """DataFrame[ticker, date, pred]; pred = last [[[CLASS]]] or 'NO_TAG'."""
    rows = []
    for f in resolve(cache_dir).glob("*.json"):
        d = json.load(open(f))
        mt, md = _TICKER.search(d.get("prompt", "")), _ASOF.search(d.get("prompt", ""))
        if not (mt and md):
            continue
        tags = _TAG.findall(d.get("response", "") or "")
        rows.append({"ticker": mt.group(1), "date": pd.Timestamp(md.group(1)),
                     "pred": tags[-1] if tags else "NO_TAG"})
    return pd.DataFrame(rows)
