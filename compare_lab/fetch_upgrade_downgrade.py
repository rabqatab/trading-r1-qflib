"""Backfill Finnhub /stock/upgrade-downgrade (broker-level dated rec changes) for the
top-150 universe -> data/finnhub_upgrade_downgrade/{ticker}.json.

Probe findings (2026-07): eps-estimate / revenue-estimate return one consensus row per
fiscal period with NO as-of dates or up/down counts (snapshot per period -> revision
breadth NOT backfillable); price-target is a pure current snapshot. upgrade-downgrade
is the only endpoint with dated history (back to ~2015-01), so that's what we save.

The endpoint appears to cap responses near ~1000 rows (AAPL full-range query -> 997),
so any ticker returning >= 900 rows is refetched year-by-year and merged/deduped.

Resumable (skips existing files). Throttled ~1 call/sec (shared key).

    uv run python -m compare_lab.fetch_upgrade_downgrade
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_OUT = Path("data/finnhub_upgrade_downgrade")
_URL = "https://finnhub.io/api/v1/stock/upgrade-downgrade"
_FROM, _TO = "2014-01-01", "2026-07-17"
_CAP = 900  # near-cap threshold -> refetch by year


def _key() -> str:
    for line in Path(".env").read_text().splitlines():
        m = re.match(r"\s*FINNHUB_KEY\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    raise SystemExit("FINNHUB_KEY not found in .env")


def _get(key: str, symbol: str, frm: str, to: str) -> list[dict]:
    for attempt in range(4):
        r = requests.get(_URL, params={"symbol": symbol, "from": frm, "to": to,
                                       "token": key}, timeout=30)
        time.sleep(1.2)
        if r.status_code == 429:
            time.sleep(15)
            continue
        r.raise_for_status()
        return r.json()
    raise RuntimeError(f"429s exhausted for {symbol}")


def main() -> int:
    key = _key()
    _OUT.mkdir(parents=True, exist_ok=True)
    tickers = sorted(pd.read_parquet(_PRICES, columns=["ticker"])["ticker"].unique())
    for i, t in enumerate(tickers):
        p = _OUT / f"{t}.json"
        if p.exists():
            continue
        rows = _get(key, t, _FROM, _TO)
        if len(rows) >= _CAP:  # likely truncated -> refetch year-by-year, dedupe
            seen: dict[tuple, dict] = {}
            for yr in range(int(_FROM[:4]), int(_TO[:4]) + 1):
                for r in _get(key, t, f"{yr}-01-01", f"{yr}-12-31"):
                    seen[(r["gradeTime"], r["company"], r["toGrade"], r["action"])] = r
            rows = sorted(seen.values(), key=lambda r: r["gradeTime"])
        p.write_text(json.dumps(rows))
        print(f"[{i + 1}/{len(tickers)}] {t}: {len(rows)} rows", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
