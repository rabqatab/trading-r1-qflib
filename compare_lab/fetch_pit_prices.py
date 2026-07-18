"""PIT bounding re-test, stage 1: fetch daily adjusted closes for every ticker that
was an S&P 500 member at any point 2016-07 .. 2026-05 and is NOT in our top-150 store.

Membership = last snapshot <= date in data/sp500_constituents/sp500_history.csv.
Known ticker renames are applied first (RENAME below, old -> current listing); what
still comes back empty from yfinance is a true delisting hole (recorded in
data/yf_prices_sp500/holes.json). Resumable: one parquet part per ~50-ticker chunk,
skip-existing; final combine writes data/yf_prices_sp500/prices.parquet.

    uv run --with yfinance python -m compare_lab.fetch_pit_prices
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

_HIST = Path("data/sp500_constituents/sp500_history.csv")
_PRICES150 = Path("data/qflib_data_store_top150/prices_top150.parquet")
_OUT_DIR = Path("data/yf_prices_sp500")
_PARTS = _OUT_DIR / "parts"
START, END = "2015-01-01", "2026-06-01"
CHUNK, SLEEP = 50, 3.0

# old membership symbol -> current listing whose price series continues the company.
# Mergers where the old listing died (UTX-side of RTX, DWDP, TWX, ...) are NOT mapped:
# they are true holes. RTN->RTX per task spec.
RENAME = {
    "FB": "META", "BLL": "BALL", "WLTW": "WTW", "PKI": "RVTY", "FBHS": "FBIN",
    "ANTM": "ELV", "ABC": "COR", "RTN": "RTX",
    "CTL": "LUMN", "COG": "CTRA", "LB": "BBWI", "FISV": "FI", "FLT": "CPAY",
    "CDAY": "DAY", "RE": "EG", "TMK": "GL", "HRS": "LHX", "MYL": "VTRS",
    "SYMC": "GEN", "NLOK": "GEN", "KORS": "CPRI", "JEC": "J", "BBT": "TFC",
    "CBS": "PSKY", "VIAC": "PSKY", "PARA": "PSKY",
    "HCP": "DOC", "PEAK": "DOC", "WRK": "SW", "WYND": "TNL", "HFC": "DINO",
    "ADS": "BFH", "GPS": "GAP", "CHK": "EXE", "BK": "BNY", "SQ": "XYZ",
    "MMC": "MRSH",  # MRSH series verified continuous back to 2015
}
SKIP = {"UTX"}  # merged into RTX; RTN carries the RTX mapping per spec


def canon(t: str) -> str:
    return RENAME.get(t, t)


def membership_union(frm="2016-07-01", to="2026-05-31") -> set[str]:
    h = pd.read_csv(_HIST)
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date")
    sets = [h[h["date"] <= frm].iloc[-1]["tickers"]]
    sets += list(h[(h["date"] > frm) & (h["date"] <= to)]["tickers"])
    out: set[str] = set()
    for s in sets:
        out |= set(s.split(","))
    return out


def main() -> int:
    import yfinance as yf

    have150 = set(pd.read_parquet(_PRICES150, columns=["ticker"])["ticker"].unique())
    union = membership_union()
    targets = sorted({canon(t) for t in union if t not in SKIP} - {canon(t) for t in have150} - have150)
    print(f"[pit-prices] union {len(union)} -> canonical extra targets {len(targets)}", flush=True)

    _PARTS.mkdir(parents=True, exist_ok=True)
    chunks = [targets[i:i + CHUNK] for i in range(0, len(targets), CHUNK)]
    for ci, ch in enumerate(chunks):
        part = _PARTS / f"chunk_{ci:03d}.parquet"
        if part.exists():
            print(f"[pit-prices] chunk {ci} exists, skip", flush=True)
            continue
        yf_syms = {t: t.replace(".", "-") for t in ch}
        df = yf.download(list(yf_syms.values()), start=START, end=END,
                         auto_adjust=True, progress=False, threads=True)["Close"]
        if isinstance(df, pd.Series):
            df = df.to_frame(list(yf_syms.values())[0])
        # retry empties once, individually
        empty = [s for s in yf_syms.values() if s not in df.columns or df[s].dropna().empty]
        if empty:
            time.sleep(SLEEP)
            r = yf.download(empty, start=START, end=END, auto_adjust=True,
                            progress=False, threads=False)
            rc = r["Close"] if "Close" in r else pd.DataFrame()
            if isinstance(rc, pd.Series):
                rc = rc.to_frame(empty[0])
            for s in empty:
                if s in rc.columns and not rc[s].dropna().empty:
                    df[s] = rc[s]
        back = {v: k for k, v in yf_syms.items()}
        long = (df.rename(columns=back).rename_axis("date").reset_index()
                .melt(id_vars="date", var_name="ticker", value_name="Close").dropna())
        long.to_parquet(part, index=False)
        got = long["ticker"].nunique()
        print(f"[pit-prices] chunk {ci}/{len(chunks)}: {got}/{len(ch)} tickers, "
              f"{len(long)} rows", flush=True)
        time.sleep(SLEEP)

    allp = pd.concat([pd.read_parquet(p) for p in sorted(_PARTS.glob("chunk_*.parquet"))])
    allp["date"] = pd.to_datetime(allp["date"]).dt.tz_localize(None).dt.normalize()
    allp = allp.drop_duplicates(["date", "ticker"])
    allp.to_parquet(_OUT_DIR / "prices.parquet", index=False)
    got = set(allp["ticker"].unique())
    holes = sorted(set(targets) - got)
    (_OUT_DIR / "holes.json").write_text(json.dumps(holes, indent=1))
    print(f"[pit-prices] DONE: {len(got)}/{len(targets)} tickers with data, "
          f"{len(holes)} holes -> holes.json\nholes: {holes}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
