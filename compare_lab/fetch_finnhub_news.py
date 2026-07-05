"""Track B (#5): re-crawl company news with Finnhub's `summary` field (richer than headlines).

Our shipped news_top150.parquet is Google-News-RSS HEADLINES only. Finnhub `/company-news`
returns a `summary` (~150 char median, up to ~1.5k) with 98% coverage — the paper's actual news
source. This fetches it so we can test whether headline+summary text moves the IC over headlines.

Finnhub free tier: 60 calls/min, ~250 articles/call cap → chunk by month, sleep ~1.1s. Resumable:
one JSON per (ticker, month). Reads FINNHUB_KEY from .env (no python-dotenv → tiny parser).

    uv run python -m compare_lab.fetch_finnhub_news --from 2024-11-01 --to 2025-07-01
"""
from __future__ import annotations

import argparse
import json
import re
import time
from hashlib import sha1
from pathlib import Path

import pandas as pd
import requests

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_PARTS = Path("data/finnhub_news_parts")
_OUT = Path("data/qflib_data_store_top150/news_top150_summ.parquet")
_URL = "https://finnhub.io/api/v1/company-news"


def _key() -> str:
    for line in Path(".env").read_text().splitlines():
        m = re.match(r"\s*FINNHUB_KEY\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    raise SystemExit("FINNHUB_KEY not found in .env")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="frm", default="2024-11-01")
    ap.add_argument("--to", default="2025-07-01")
    ap.add_argument("--sleep", type=float, default=1.1)  # stay under 60/min
    args = ap.parse_args()
    key = _key()
    _PARTS.mkdir(parents=True, exist_ok=True)
    tickers = sorted(pd.read_parquet(_PRICES, columns=["ticker"])["ticker"].unique())
    months = pd.date_range(args.frm, args.to, freq="MS")
    print(f"[finnhub] {len(tickers)} tickers × {len(months)} months", flush=True)

    calls = 0
    for ti, t in enumerate(tickers):
        for m in months:
            frm = m.strftime("%Y-%m-%d")
            to = (m + pd.offsets.MonthEnd(1)).strftime("%Y-%m-%d")
            part = _PARTS / f"{t}_{m.strftime('%Y%m')}.json"
            if part.exists():
                continue
            for attempt in range(4):
                try:
                    r = requests.get(_URL, params={"symbol": t, "from": frm, "to": to,
                                                   "token": key}, timeout=20)
                    if r.status_code == 429:      # rate limited
                        time.sleep(5 * (attempt + 1)); continue
                    r.raise_for_status()
                    part.write_text(json.dumps(r.json()))
                    break
                except Exception:
                    time.sleep(3 * (attempt + 1))
            calls += 1
            time.sleep(args.sleep)
        if (ti + 1) % 10 == 0:
            print(f"[finnhub] {ti+1}/{len(tickers)} tickers, {calls} calls", flush=True)

    # assemble parquet
    rows = []
    for p in _PARTS.glob("*.json"):
        t = p.stem.split("_")[0]
        for a in json.loads(p.read_text()):
            ts = pd.to_datetime(a.get("datetime", 0), unit="s")
            rows.append({"ticker": t, "date": ts.normalize(), "published_at": ts,
                         "headline": a.get("headline", ""), "summary": a.get("summary", "") or "",
                         "source": a.get("source", ""), "url": a.get("url", ""),
                         "url_hash": sha1((a.get("url", "") or str(a.get("id"))).encode()).hexdigest()})
    df = pd.DataFrame(rows).drop_duplicates(subset=["ticker", "url_hash"])
    df = df[df["headline"].str.len() > 0].sort_values(["ticker", "published_at"])
    df.to_parquet(_OUT)
    nonempty = (df["summary"].str.len() > 0).mean()
    print(f"[finnhub] wrote {len(df)} rows ({df.ticker.nunique()} tickers) -> {_OUT} | "
          f"summary non-empty {100*nonempty:.0f}%", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
