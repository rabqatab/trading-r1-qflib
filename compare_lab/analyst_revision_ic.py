"""Roadmap C#2: does analyst consensus-revision momentum carry OOS signal over price?

Analyst recommendation revisions are one of the few signals documented to work in LARGE caps
(not a small-cap/borrow-fee effect) and are NOT a function of the own-price path — so by DPI they
could add information over price/technical features. We test it on the SAME 2025-H1 OOS grid.

Signal per (ticker, as_of): consensus score = (SB*2+B-S-SS*2)/total from Finnhub recommendation,
strictly PIT (only months with period <= as_of). Two features:
  - level    : latest consensus score <= as_of
  - revision : score(latest) - score(3 months prior)  [revision momentum]
IC = Spearman(feature, target) where target = make_signal proxy AND raw 7-day return.

    uv run python -m compare_lab.analyst_revision_ic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.labeling import make_signal

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_RECS = Path("data/finnhub_recs")
_EVAL = Path("compare_lab/eval150/eval_mm.jsonl")


def _spear(a, b):
    a, b = pd.Series(a), pd.Series(b)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    return float(np.corrcoef(a.rank(), b.rank())[0, 1]) if a.nunique() > 1 and b.nunique() > 1 else float("nan")


def _rec_series(ticker: str) -> pd.DataFrame | None:
    p = _RECS / f"{ticker}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    if not d:
        return None
    df = pd.DataFrame(d)
    df["period"] = pd.to_datetime(df["period"])
    tot = df.strongBuy + df.buy + df.hold + df.sell + df.strongSell
    df["score"] = (df.strongBuy * 2 + df.buy - df.sell - df.strongSell * 2) / tot.replace(0, np.nan)
    return df.sort_values("period").reset_index(drop=True)


def main() -> int:
    tickers = sorted(pd.read_parquet(_PRICES, columns=["ticker"])["ticker"].unique())
    px = pd.read_parquet(_PRICES, columns=["date", "ticker", "Close"])
    px["date"] = pd.to_datetime(px["date"])
    piv = px.pivot(index="date", columns="ticker", values="Close").sort_index()
    sig = {t: make_signal(piv[t].dropna(), forward=True) for t in tickers}
    recs = {t: _rec_series(t) for t in tickers}
    have = sum(v is not None for v in recs.values())
    print(f"[revision] recommendation series for {have}/{len(tickers)} tickers")

    def raw7(t, d):
        s = piv[t].dropna()
        if d not in s.index:
            return None
        i = s.index.get_loc(d)
        return s.iloc[i + 7] / s.iloc[i] - 1 if i + 7 < len(s) else None

    ev = [json.loads(l) for l in _EVAL.open()]
    rows = []
    for e in ev:
        t, d = e["ticker"], pd.Timestamp(e["as_of"])
        r = recs.get(t)
        if r is None:
            continue
        pit = r[r["period"] <= d]                      # strictly point-in-time
        if len(pit) < 4:
            continue
        level = pit["score"].iloc[-1]
        revision = level - pit["score"].iloc[-4]       # ~3-month consensus change
        raw = raw7(t, d)
        rows.append({"level": level, "revision": revision,
                     "signal": e["signal"], "raw": raw})
    df = pd.DataFrame(rows)
    print(f"[revision] {len(df)} eval points with PIT recommendation history\n")
    print(f"{'feature':<20}{'IC vs make_signal':>20}{'IC vs RAW 7d':>16}")
    print("-" * 56)
    for feat in ("level", "revision"):
        print(f"{feat:<20}{_spear(df[feat], df['signal']):>+20.3f}{_spear(df[feat], df['raw']):>+16.3f}")
    # combine with a naive momentum feature? report correlation to price signal for independence
    print(f"\ncorr(revision, make_signal proxy) = {_spear(df['revision'], df['signal']):+.3f}  "
          f"(low & raw-IC>0 ⇒ genuinely new large-cap signal)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
