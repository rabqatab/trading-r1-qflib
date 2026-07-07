"""Roadmap C#2 extension + §5: PEAD + signal combination — can decorrelated weak signals lift
effective raw IC without any modelling?

Signals per (ticker, as_of) on the SAME 2025-H1 OOS grid, all strictly PIT:
  momentum : 10-day price return at as_of (the proven 0.064-raw baseline feature)
  revision : 3-month analyst consensus change (Finnhub /stock/recommendation) — raw IC 0.080
  pead     : latest EPS surprise% announced <= as_of (within 90d), Finnhub /calendar/earnings
             (announcement DATE known → PIT-safe post-earnings drift)
  combo    : equal-weight mean of available signal ranks (Granger: simple average is hard to beat)

Metric: Spearman vs RAW 7-day forward return (the honest one) + proxy for reference, plus a
Q1/Q2 temporal-stability split (the revision signal's known weakness).

    uv run python -m compare_lab.signal_combine_ic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_RECS = Path("data/finnhub_recs")
_EARN = Path("data/finnhub_earnings")
_EVAL = Path("compare_lab/eval150/eval_mm.jsonl")


def _spear(a, b):
    a, b = pd.Series(a, dtype=float), pd.Series(b, dtype=float)
    m = a.notna() & b.notna()
    a, b = a[m], b[m]
    return (float(np.corrcoef(a.rank(), b.rank())[0, 1]), int(m.sum())) \
        if a.nunique() > 1 and b.nunique() > 1 else (float("nan"), int(m.sum()))


def main() -> int:
    px = pd.read_parquet(_PRICES, columns=["date", "ticker", "Close"])
    px["date"] = pd.to_datetime(px["date"])
    piv = px.pivot(index="date", columns="ticker", values="Close").sort_index()

    # --- per-ticker PIT sources ---
    def recs(t):
        p = _RECS / f"{t}.json"
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

    def earns(t):
        p = _EARN / f"{t}.json"
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        rows = [{"date": pd.Timestamp(a["date"]),
                 "surp": (a["epsActual"] - a["epsEstimate"]) / abs(a["epsEstimate"])}
                for a in d if a.get("epsActual") is not None and a.get("epsEstimate")]
        return pd.DataFrame(rows).sort_values("date") if rows else None

    tickers = sorted(piv.columns)
    R = {t: recs(t) for t in tickers}
    E = {t: earns(t) for t in tickers}

    def raw7(t, d):
        s = piv[t].dropna()
        if d not in s.index:
            return np.nan
        i = s.index.get_loc(d)
        return s.iloc[i + 7] / s.iloc[i] - 1 if i + 7 < len(s) else np.nan

    def mom10(t, d):
        s = piv[t].dropna()
        if d not in s.index:
            return np.nan
        i = s.index.get_loc(d)
        return s.iloc[i] / s.iloc[i - 10] - 1 if i >= 10 else np.nan

    rows = []
    for e in (json.loads(l) for l in _EVAL.open()):
        t, d = e["ticker"], pd.Timestamp(e["as_of"])
        rec = R.get(t)
        rev = np.nan
        if rec is not None:
            pit = rec[rec["period"] <= d]
            if len(pit) >= 4:
                rev = pit["score"].iloc[-1] - pit["score"].iloc[-4]
        er = E.get(t)
        pead = np.nan
        if er is not None:
            pit = er[(er["date"] <= d) & (er["date"] >= d - pd.Timedelta(days=90))]
            if len(pit):
                pead = pit["surp"].iloc[-1]
        rows.append({"ticker": t, "date": d, "mom": mom10(t, d), "rev": rev,
                     "pead": pead, "raw": raw7(t, d), "sig": e["signal"]})
    df = pd.DataFrame(rows)
    # combo = mean of available ranks (NaN-tolerant)
    ranks = df[["mom", "rev", "pead"]].rank()
    df["combo"] = ranks.mean(axis=1, skipna=True)
    df["combo2"] = ranks[["mom", "rev"]].mean(axis=1, skipna=True)  # the two proven ones

    print(f"coverage: mom {df.mom.notna().sum()} | rev {df.rev.notna().sum()} | "
          f"pead {df.pead.notna().sum()} / {len(df)}\n")
    print(f"{'signal':<14}{'RAW 7d IC':>12}{'n':>6}   {'proxy IC':>10}")
    print("-" * 46)
    for c in ("mom", "rev", "pead", "combo2", "combo"):
        ic_r, n = _spear(df[c], df["raw"])
        ic_p, _ = _spear(df[c], df["sig"])
        print(f"{c:<14}{ic_r:>+12.3f}{n:>6}   {ic_p:>+10.3f}")
    # cross-correlations (independence check) + temporal stability on raw
    print("\nsignal cross-corr (rank): mom-rev %.3f | mom-pead %.3f | rev-pead %.3f" % (
        _spear(df["mom"], df["rev"])[0], _spear(df["mom"], df["pead"])[0],
        _spear(df["rev"], df["pead"])[0]))
    mid = df["date"].quantile(0.5)
    for c in ("mom", "rev", "pead", "combo"):
        a = df[df.date <= mid]; b = df[df.date > mid]
        print(f"  {c:<8} Q1 {_spear(a[c], a['raw'])[0]:+.3f} | Q2 {_spear(b[c], b['raw'])[0]:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
