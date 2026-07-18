"""PIT bounding re-test, stage 2: backfill Finnhub recommendation trends and earnings
calendar for the extra (non-top-150) PIT tickers that have price data.

Writes into the SAME layouts the decade backtest already reads:
  data/finnhub_recs/{T}.json           /stock/recommendation?symbol=T
  data/finnhub_earnings_full/{T}.json  /calendar/earnings?from=2016-01-01&to=2026-05-31&symbol=T
Resumable: skips existing files (so our 150 are never re-fetched). ~1.1 s throttle
(free tier 60/min). Empty responses are written as [] so reruns skip them too.

    uv run python -m compare_lab.fetch_pit_finnhub
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

_RECS = Path("data/finnhub_recs")
_EARN = Path("data/finnhub_earnings_full")
_YF = Path("data/yf_prices_sp500/prices.parquet")
SLEEP = 1.1


def _key() -> str:
    for line in Path(".env").read_text().splitlines():
        m = re.match(r"\s*FINNHUB_KEY\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    raise SystemExit("FINNHUB_KEY not found in .env")


def _get(url: str, params: dict) -> object | None:
    for attempt in range(4):
        try:
            r = requests.get(url, params=params, timeout=30)
            if r.status_code == 429:
                time.sleep(15)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"  retry {attempt}: {type(e).__name__}", flush=True)
            time.sleep(5 * (attempt + 1))
    return None


def main() -> int:
    key = _key()
    tickers = sorted(pd.read_parquet(_YF, columns=["ticker"])["ticker"].unique())
    _RECS.mkdir(exist_ok=True)
    _EARN.mkdir(exist_ok=True)
    jobs = []
    for t in tickers:
        if not (_RECS / f"{t}.json").exists():
            jobs.append(("rec", t))
        if not (_EARN / f"{t}.json").exists():
            jobs.append(("earn", t))
    print(f"[finnhub-pit] {len(tickers)} tickers, {len(jobs)} calls to make", flush=True)

    for i, (kind, t) in enumerate(jobs):
        if kind == "rec":
            d = _get("https://finnhub.io/api/v1/stock/recommendation",
                     {"symbol": t, "token": key})
            if d is not None:
                (_RECS / f"{t}.json").write_text(json.dumps(d))
        else:
            d = _get("https://finnhub.io/api/v1/calendar/earnings",
                     {"from": "2016-01-01", "to": "2026-05-31", "symbol": t, "token": key})
            if d is not None:
                rows = d.get("earningsCalendar", []) if isinstance(d, dict) else []
                (_EARN / f"{t}.json").write_text(json.dumps(rows))
        if i % 50 == 0:
            print(f"[finnhub-pit] {i}/{len(jobs)}", flush=True)
        time.sleep(SLEEP)

    nr = sum(1 for t in tickers if (_RECS / f"{t}.json").exists())
    ne = sum(1 for t in tickers if (_EARN / f"{t}.json").exists())
    print(f"[finnhub-pit] DONE recs {nr}/{len(tickers)}  earnings {ne}/{len(tickers)}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
