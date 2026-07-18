"""Lead-lag stage 1b: prices.parquet has no volume, so refetch daily share Volume
for the same 485 yf tickers -> data/yf_prices_sp500/volume.parquet (long format:
date, ticker, Volume). Resumable one-parquet-per-chunk like fetch_pit_prices.
(Top-150 store already carries Volume, not refetched here.)

    uv run --with yfinance python -m compare_lab.fetch_yf_volume
"""
from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

_OUT_DIR = Path("data/yf_prices_sp500")
_PARTS = _OUT_DIR / "volume_parts"
_YF = _OUT_DIR / "prices.parquet"
START, END = "2015-01-01", "2026-06-01"
CHUNK, SLEEP = 50, 3.0


def main() -> int:
    import yfinance as yf

    targets = sorted(pd.read_parquet(_YF, columns=["ticker"])["ticker"].unique())
    print(f"[yf-volume] {len(targets)} tickers", flush=True)
    _PARTS.mkdir(parents=True, exist_ok=True)
    chunks = [targets[i:i + CHUNK] for i in range(0, len(targets), CHUNK)]
    for ci, ch in enumerate(chunks):
        part = _PARTS / f"chunk_{ci:03d}.parquet"
        if part.exists():
            print(f"[yf-volume] chunk {ci} exists, skip", flush=True)
            continue
        yf_syms = {t: t.replace(".", "-") for t in ch}
        df = yf.download(list(yf_syms.values()), start=START, end=END,
                         auto_adjust=True, progress=False, threads=True)["Volume"]
        if isinstance(df, pd.Series):
            df = df.to_frame(list(yf_syms.values())[0])
        empty = [s for s in yf_syms.values()
                 if s not in df.columns or df[s].dropna().empty]
        if empty:
            time.sleep(SLEEP)
            r = yf.download(empty, start=START, end=END, auto_adjust=True,
                            progress=False, threads=False)
            rc = r["Volume"] if "Volume" in r else pd.DataFrame()
            if isinstance(rc, pd.Series):
                rc = rc.to_frame(empty[0])
            for s in empty:
                if s in rc.columns and not rc[s].dropna().empty:
                    df[s] = rc[s]
        back = {v: k for k, v in yf_syms.items()}
        long = (df.rename(columns=back).rename_axis("date").reset_index()
                .melt(id_vars="date", var_name="ticker", value_name="Volume").dropna())
        long.to_parquet(part, index=False)
        print(f"[yf-volume] chunk {ci}/{len(chunks)}: {long['ticker'].nunique()}"
              f"/{len(ch)} tickers, {len(long)} rows", flush=True)
        time.sleep(SLEEP)

    allp = pd.concat([pd.read_parquet(p) for p in sorted(_PARTS.glob("chunk_*.parquet"))])
    allp["date"] = pd.to_datetime(allp["date"]).dt.tz_localize(None).dt.normalize()
    allp = allp.drop_duplicates(["date", "ticker"])
    allp.to_parquet(_OUT_DIR / "volume.parquet", index=False)
    print(f"[yf-volume] DONE {allp['ticker'].nunique()}/{len(targets)} tickers, "
          f"{len(allp)} rows -> volume.parquet", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
