"""Lead-lag stage 1a: fetch Finnhub /stock/profile2 for every ticker in the PIT
panel (top-150 store + yf extras) -> data/finnhub_profiles/{T}.json (T = panel
ticker name). finnhubIndustry is the industry key (coarse, ~60 buckets).
Resumable: skips existing files; empty responses written as {} so reruns skip.
~1.2 s throttle (key shared with another concurrent agent).

    uv run python -m compare_lab.fetch_finnhub_profiles
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import requests

_OUT = Path("data/finnhub_profiles")
_YF = Path("data/yf_prices_sp500/prices.parquet")
_P150 = Path("data/qflib_data_store_top150/prices_top150.parquet")
SLEEP = 1.2


def _key() -> str:
    for line in Path(".env").read_text().splitlines():
        m = re.match(r"\s*FINNHUB_KEY\s*=\s*(\S+)", line)
        if m:
            return m.group(1)
    raise SystemExit("FINNHUB_KEY not found in .env")


def _get(params: dict) -> object | None:
    for attempt in range(4):
        try:
            r = requests.get("https://finnhub.io/api/v1/stock/profile2",
                             params=params, timeout=30)
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
    tickers = sorted(
        set(pd.read_parquet(_YF, columns=["ticker"])["ticker"].unique())
        | set(pd.read_parquet(_P150, columns=["ticker"])["ticker"].unique()))
    _OUT.mkdir(exist_ok=True)
    jobs = [t for t in tickers if not (_OUT / f"{t}.json").exists()]
    print(f"[profiles] {len(tickers)} tickers, {len(jobs)} to fetch", flush=True)

    for i, t in enumerate(jobs):
        sym = t.replace("-", ".")  # Finnhub uses dot class notation (BRK.B)
        d = _get({"symbol": sym, "token": key})
        if d is not None:
            (_OUT / f"{t}.json").write_text(json.dumps(d))
        if i % 50 == 0:
            print(f"[profiles] {i}/{len(jobs)} ({t})", flush=True)
        time.sleep(SLEEP)

    have = [t for t in tickers if (_OUT / f"{t}.json").exists()]
    n_ind = sum(1 for t in have
                if json.loads((_OUT / f"{t}.json").read_text()).get("finnhubIndustry"))
    print(f"[profiles] DONE {len(have)}/{len(tickers)} files, "
          f"{n_ind} with finnhubIndustry", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
