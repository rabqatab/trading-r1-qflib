"""Backfill Finnhub data for the co-coverage momentum feature (Ali-Hirshleifer 2020):

  data/finnhub_upgrade_downgrade/{T}.json  — broker-level rec actions (upgrade-downgrade),
      fetched in two date chunks (default call caps ~1000 rows), merged + deduped.
  data/finnhub_supply_chain/{T}.json       — customer/supplier links (/stock/supply-chain).

Universe = top-150 store UNION the 485-ticker PIT S&P panel (yf_prices_sp500), so the
co-coverage graph can be rebuilt over the full PIT universe (cocoverage_pit_ic.py).
Upgrade-downgrade is fetched for every ticker; supply-chain only for the top-150.
A date chunk that returns >= _CAP rows likely hit Finnhub's ~1000-row cap and is
recursively split in half until under the cap.

Skips per-ticker files that already exist (another agent may be backfilling concurrently).
Reads FINNHUB_KEY from .env; throttles ~1.2 s/call (shared key).

    uv run python compare_lab/fetch_cocoverage_data.py
"""
from __future__ import annotations

import json
import re
import time
import urllib.request
from pathlib import Path

import pandas as pd

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_PIT = Path("data/yf_prices_sp500/prices.parquet")
_UD_DIR = Path("data/finnhub_upgrade_downgrade")
_SC_DIR = Path("data/finnhub_supply_chain")
_CHUNKS = [("2013-01-01", "2019-12-31"), ("2020-01-01", "2026-12-31")]
_CAP = 990          # near the observed ~1000-row per-call ceiling
_SLEEP = 1.2


def _key() -> str:
    for line in Path(".env").read_text().splitlines():
        m = re.match(r"\s*FINNHUB_KEY\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    raise SystemExit("FINNHUB_KEY not found in .env")


def _get(url: str, key: str):
    req = urllib.request.Request(url, headers={"X-Finnhub-Token": key})
    for attempt in range(5):
        try:
            with urllib.request.urlopen(req, timeout=30) as r:
                return json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            if e.code == 429:
                time.sleep(15 * (attempt + 1))
                continue
            raise
    raise RuntimeError(f"gave up on {url.split('?')[0]}")


def _fetch_ud_range(t: str, frm: str, to: str, key: str) -> list[dict]:
    """One upgrade-downgrade call; recursively halve the range if the ~1000-row
    cap is hit."""
    d = _get("https://finnhub.io/api/v1/stock/upgrade-downgrade"
             f"?symbol={t}&from={frm}&to={to}", key) or []
    time.sleep(_SLEEP)
    if len(d) < _CAP:
        return d
    lo, hi = pd.Timestamp(frm), pd.Timestamp(to)
    mid = lo + (hi - lo) / 2
    print(f"  cap hit ({len(d)} rows) {t} {frm}..{to} -> splitting", flush=True)
    return (_fetch_ud_range(t, frm, str(mid.date()), key)
            + _fetch_ud_range(t, str((mid + pd.Timedelta(days=1)).date()), to, key))


def main() -> int:
    key = _key()
    top150 = set(pd.read_parquet(_PRICES, columns=["ticker"]).ticker.unique())
    pit = set(pd.read_parquet(_PIT, columns=["ticker"]).ticker.unique())
    tickers = sorted(top150 | pit)
    _UD_DIR.mkdir(parents=True, exist_ok=True)
    _SC_DIR.mkdir(parents=True, exist_ok=True)

    for i, t in enumerate(tickers):
        p = _UD_DIR / f"{t}.json"
        if not p.exists():
            rows: dict[tuple, dict] = {}
            for frm, to in _CHUNKS:
                for r in _fetch_ud_range(t, frm, to, key):
                    rows[(r["gradeTime"], r["company"], r.get("action"))] = r
            out = sorted(rows.values(), key=lambda r: r["gradeTime"])
            p.write_text(json.dumps(out))
            print(f"[{i+1}/{len(tickers)}] ud {t}: {len(out)} rows", flush=True)

        if t not in top150:
            continue
        p = _SC_DIR / f"{t}.json"
        if not p.exists():
            d = _get(f"https://finnhub.io/api/v1/stock/supply-chain?symbol={t}", key)
            p.write_text(json.dumps(d))
            time.sleep(_SLEEP)
            print(f"[{i+1}/{len(tickers)}] sc {t}: {len((d or {}).get('data') or [])} links",
                  flush=True)
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
