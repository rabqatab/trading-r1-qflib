"""Shared-analyst co-coverage momentum (Ali-Hirshleifer JFE 2020 "connected-firm
momentum") on the top-150 universe.

Graph: edge w_ij = sum over brokers b that acted (upgrade/downgrade feed) on BOTH i
and j within a trailing 24-month window, weighted 1/#stocks b covers in that window
(tf-idf downweight of megabrokers, as in Ali-Hirshleifer). Recomputed yearly from the
window ending Dec 31 of the prior year (no look-ahead in the graph). Honest proxy
caveat: Finnhub exposes broker names ("company"), not individual analyst identity, so
this is broker-co-coverage, not analyst-co-coverage.

Feature: CF_k(i,t) = sum_j w_ij * r_j(t-k..t) / sum_j w_ij, j != i, k in {5,10,21}.
Eval: per-year (2017-07..) daily cross-sectional Spearman IC vs raw 7d forward return
(same convention as cvar_conformal_backtest), both raw and residualized on the stock's
own 5d return (cross-sectionally, per day) since 1-week own reversal contaminates.

Also: same neighbor-return feature on the Finnhub supply-chain graph (current snapshot
only — no history endpoint, so the graph itself is look-ahead; flagged in output).

    uv run python -m compare_lab.cocoverage_momentum_ic
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.cvar_conformal_backtest import build_signals

_UD = Path("data/finnhub_upgrade_downgrade")
_SC = Path("data/finnhub_supply_chain")
START = "2017-07-01"
KS = (5, 10, 21)
YEARS = range(2017, 2027)


def load_events(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        p = _UD / f"{t}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            rows.append((t, r["company"], pd.Timestamp(r["gradeTime"], unit="s")))
    return pd.DataFrame(rows, columns=["ticker", "broker", "date"])


def cocov_matrix(ev: pd.DataFrame, tickers: list[str], end_year: int,
                 tfidf: bool = True) -> np.ndarray:
    """w_ij from events in the trailing 24m window [end_year-2, end_year] (exclusive)."""
    lo, hi = pd.Timestamp(f"{end_year - 2}-01-01"), pd.Timestamp(f"{end_year}-01-01")
    win = ev[(ev.date >= lo) & (ev.date < hi)]
    idx = {t: i for i, t in enumerate(tickers)}
    cov: dict[str, set] = defaultdict(set)
    for t, b in zip(win.ticker, win.broker):
        cov[b].add(t)
    n = len(tickers)
    W = np.zeros((n, n))
    for b, ts in cov.items():
        if len(ts) < 2:
            continue
        u = 1.0 / len(ts) if tfidf else 1.0
        ii = [idx[t] for t in ts]
        for a in range(len(ii)):
            for c in range(a + 1, len(ii)):
                W[ii[a], ii[c]] += u
                W[ii[c], ii[a]] += u
    return W


def supply_chain_matrix(tickers: list[str]) -> np.ndarray:
    idx = {t: i for i, t in enumerate(tickers)}
    n = len(tickers)
    W = np.zeros((n, n))
    for t in tickers:
        p = _SC / f"{t}.json"
        if not p.exists():
            continue
        for link in (json.loads(p.read_text()) or {}).get("data") or []:
            j = link.get("symbol")
            if j in idx and j != t:
                W[idx[t], idx[j]] = 1.0
                W[idx[j], idx[t]] = 1.0
    return W


def neighbor_signal(Rk: pd.DataFrame, W_by_year: dict[int, np.ndarray]) -> pd.DataFrame:
    """CF(i,t) = sum_j W_ij Rk(j,t) / sum_{j valid} W_ij, using the year(t) graph."""
    out = pd.DataFrame(index=Rk.index, columns=Rk.columns, dtype=float)
    for yr, di in Rk.groupby(Rk.index.year).groups.items():
        W = W_by_year.get(yr)
        if W is None:
            continue
        Rv = Rk.loc[di].to_numpy()
        M = ~np.isnan(Rv)
        num = np.nan_to_num(Rv) @ W          # W symmetric, zero diag
        den = M.astype(float) @ W
        with np.errstate(invalid="ignore", divide="ignore"):
            out.loc[di] = np.where(den > 0, num / den, np.nan)
    return out


def _rank_z(df: pd.DataFrame) -> pd.DataFrame:
    r = df.rank(axis=1)
    return r.sub(r.mean(axis=1), axis=0).div(r.std(axis=1), axis=0)


def residualize(sig: pd.DataFrame, ctrl: pd.DataFrame) -> pd.DataFrame:
    """Per-day cross-sectional residual of rank(sig) on rank(ctrl)."""
    y, x = _rank_z(sig), _rank_z(ctrl)
    m = y.notna() & x.notna()
    ym, xm = y.where(m), x.where(m)
    beta = (ym * xm).mean(axis=1) / (xm * xm).mean(axis=1)
    return ym.sub(xm.mul(beta, axis=0))


def yearly_ic(sig: pd.DataFrame, fwd: pd.DataFrame) -> dict[int, float]:
    ics = sig.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)
    return ics.groupby(ics.index.year).mean().to_dict()


def main() -> int:
    piv, base_sigs = build_signals()
    tickers = list(piv.columns)
    dates = piv.index[piv.index >= START]
    fwd7 = (piv.shift(-7) / piv - 1).loc[dates]
    own5 = (piv / piv.shift(5) - 1).loc[dates]

    ev = load_events(tickers)
    n_files = sum((_UD / f"{t}.json").exists() for t in tickers)
    first = ev.groupby("ticker").date.min()
    print(f"upgrade-downgrade: {n_files}/150 files, {len(ev)} events, "
          f"{ev.broker.nunique()} distinct brokers")
    print(f"earliest event per ticker: median {first.median().date()}, "
          f"max (worst) {first.max().date()} ({first.idxmax()})")

    W_by_year = {yr: cocov_matrix(ev, tickers, yr) for yr in YEARS}

    # self-check: AAPL-MSFT 2024 weight, brute force over brokers vs matrix entry
    lo, hi = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
    win = ev[(ev.date >= lo) & (ev.date < hi)]
    cov = win.groupby("broker").ticker.agg(set)
    brute = sum(1.0 / len(s) for s in cov if "AAPL" in s and "MSFT" in s)
    ia, im = tickers.index("AAPL"), tickers.index("MSFT")
    assert np.isclose(brute, W_by_year[2024][ia, im]), (brute, W_by_year[2024][ia, im])
    print(f"self-check ok: w(AAPL,MSFT|2024 graph) = {brute:.4f} "
          f"(shared brokers: {sum('AAPL' in s and 'MSFT' in s for s in cov)})")
    deg = (W_by_year[2024] > 0).sum(axis=1)
    print(f"2024 graph density: mean degree {deg.mean():.0f}/149 (dense, as expected)\n")

    W_sc = supply_chain_matrix(tickers)
    sc_deg = (W_sc > 0).sum(axis=1)
    print(f"supply-chain graph (static snapshot, look-ahead caveat): "
          f"{int((W_sc > 0).sum() / 2)} edges, mean degree {sc_deg.mean():.1f}, "
          f"{int((sc_deg == 0).sum())} isolated tickers\n")

    graphs = {"cocov": W_by_year, "supply": {yr: W_sc for yr in YEARS}}
    feats: dict[str, pd.DataFrame] = {}
    for g, Ws in graphs.items():
        for k in KS:
            Rk = (piv / piv.shift(k) - 1).loc[dates]
            feats[f"{g}{k}"] = neighbor_signal(Rk, Ws)

    names = list(feats)
    print("per-year daily cross-sectional IC vs raw 7d fwd return")
    for label, tf in [("RAW", lambda s: s),
                      ("RESID on own 5d ret", lambda s: residualize(s, own5))]:
        cols = {n: yearly_ic(tf(feats[n]), fwd7) for n in names}
        print(f"\n[{label}]")
        print(f"{'year':<6}" + "".join(f"{n:>10}" for n in names))
        yrs = sorted({y for c in cols.values() for y in c})
        for y in yrs:
            print(f"{y:<6}" + "".join(f"{cols[n].get(y, float('nan')):>+10.3f}"
                                      for n in names))
        print(f"{'mean':<6}" + "".join(
            f"{np.nanmean(list(cols[n].values())):>+10.3f}" for n in names))

    print("\nmean daily cross-sectional Spearman corr of feature vs existing signals:")
    print(f"{'feat':<10}" + "".join(f"{b:>8}" for b in ("mom", "rev", "combo")) + f"{'own5':>8}")
    for n in names:
        row = f"{n:<10}"
        for b in ("mom", "rev", "combo"):
            row += f"{feats[n].rank(axis=1).corrwith(base_sigs[b].loc[dates].rank(axis=1), axis=1).mean():>+8.3f}"
        row += f"{feats[n].rank(axis=1).corrwith(own5.rank(axis=1), axis=1).mean():>+8.3f}"
        print(row)

    print("\ncaveats: broker-level (not analyst-level) co-coverage proxy; survivorship-"
          "biased today's-top-150 universe; supply-chain graph is a current snapshot "
          "(graph composition is look-ahead); dense graph within 150 large caps means "
          "the feature is close to a broker-weighted market factor.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
