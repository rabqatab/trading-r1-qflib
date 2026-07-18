"""Co-coverage momentum (Ali-Hirshleifer JFE 2020) re-test on the PIT S&P universe.

The top-150 test failed structurally: 121/149 mean degree -> neighbor return is the
market average. The literature says the effect lives where coverage is SPARSE, so we
re-test on the 485-extra-ticker PIT S&P panel (pit_bounding_backtest conventions),
focusing on the BOTTOM coverage tercile (#covering brokers, trailing 24 months).

Graph: trailing-24m broker-co-coverage over the FULL PIT universe, tf-idf broker
weights (1/#stocks covered), rebuilt yearly from the window ending Dec 31 of the
prior year (no look-ahead). Broker-level proxy caveat as before (no analyst ids).

Feature CF_k(i,t): co-coverage-weighted mean of neighbors' past k in {5,10,21} day
returns, own excluded. Directed variant: for targets in the bottom coverage tercile,
neighbors restricted to the TOP coverage tercile (attention flows visible->neglected).

Eval: per-year (2017-07..) daily cross-sectional Spearman vs raw 7d fwd return, PIT
members with price data only; raw and residualized on own past-5d return; on the full
PIT cross-section and separately on the bottom coverage tercile.

    uv run python -m compare_lab.cocoverage_pit_ic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.pit_bounding_backtest import membership_mask

_UD = Path("data/finnhub_upgrade_downgrade")
_P150 = Path("data/qflib_data_store_top150/prices_top150.parquet")
_YF = Path("data/yf_prices_sp500/prices.parquet")
START = "2017-07-01"
KS = (5, 10, 21)
YEARS = range(2017, 2027)


def load_panel() -> pd.DataFrame:
    px = pd.read_parquet(_P150, columns=["date", "ticker", "Close"])
    px["date"] = pd.to_datetime(px["date"])
    piv150 = px.pivot(index="date", columns="ticker", values="Close").sort_index()
    yf = pd.read_parquet(_YF)
    yf["date"] = pd.to_datetime(yf["date"])
    piv_x = yf.pivot(index="date", columns="ticker", values="Close").sort_index()
    piv_x = piv_x.reindex(piv150.index)
    piv_x = piv_x.drop(columns=piv_x.columns.intersection(piv150.columns))
    return pd.concat([piv150, piv_x], axis=1).sort_index(axis=1)


def load_events(tickers: list[str]) -> pd.DataFrame:
    rows = []
    for t in tickers:
        p = _UD / f"{t}.json"
        if not p.exists():
            continue
        for r in json.loads(p.read_text()):
            rows.append((t, r["company"], pd.Timestamp(r["gradeTime"], unit="s")))
    return pd.DataFrame(rows, columns=["ticker", "broker", "date"])


def broker_incidence(ev: pd.DataFrame, tickers: list[str],
                     end_year: int) -> np.ndarray:
    """0/1 broker x stock incidence over the trailing 24m window ending Jan 1 of
    end_year."""
    lo, hi = pd.Timestamp(f"{end_year - 2}-01-01"), pd.Timestamp(f"{end_year}-01-01")
    win = ev[(ev.date >= lo) & (ev.date < hi)]
    pairs = win[["broker", "ticker"]].drop_duplicates()
    bi = {b: i for i, b in enumerate(pairs.broker.unique())}
    ti = {t: i for i, t in enumerate(tickers)}
    B = np.zeros((len(bi), len(tickers)))
    B[pairs.broker.map(bi), pairs.ticker.map(ti)] = 1.0
    return B


def cocov_from_incidence(B: np.ndarray) -> np.ndarray:
    """W = B' diag(1/#stocks) B, zero diagonal (tf-idf broker co-coverage)."""
    nb = B.sum(axis=1)
    u = np.divide(1.0, nb, out=np.zeros_like(nb), where=nb > 0)
    W = B.T @ (B * u[:, None])
    np.fill_diagonal(W, 0.0)
    return W


def neighbor_signal(Rk: pd.DataFrame, W_by_year: dict[int, np.ndarray]) -> pd.DataFrame:
    """CF(i,t) = sum_j W_ji Rk(j,t) / sum_{j valid} W_ji, using the year(t) graph."""
    out = pd.DataFrame(index=Rk.index, columns=Rk.columns, dtype=float)
    for yr, di in Rk.groupby(Rk.index.year).groups.items():
        W = W_by_year.get(yr)
        if W is None:
            continue
        Rv = Rk.loc[di].to_numpy()
        M = ~np.isnan(Rv)
        num = np.nan_to_num(Rv) @ W
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


def yearly_ic(sig: pd.DataFrame, fwd: pd.DataFrame,
              mask: pd.DataFrame) -> dict[int, float]:
    s, f = sig.where(mask), fwd.where(mask)
    ics = s.rank(axis=1).corrwith(f.rank(axis=1), axis=1)
    return ics.groupby(ics.index.year).mean().to_dict()


def ic_block(title: str, feats: dict[str, pd.DataFrame], fwd: pd.DataFrame,
             own5: pd.DataFrame, mask: pd.DataFrame) -> None:
    names = list(feats)
    print(f"\n=== {title}: per-year daily cross-sectional IC vs raw 7d fwd ===")
    for label, tf in [("RAW", lambda s: s),
                      ("RESID on own 5d ret",
                       lambda s: residualize(s.where(mask), own5.where(mask)))]:
        cols = {n: yearly_ic(tf(feats[n]), fwd, mask) for n in names}
        print(f"[{label}]")
        print(f"{'year':<6}" + "".join(f"{n:>10}" for n in names))
        yrs = sorted({y for c in cols.values() for y in c})
        for y in yrs:
            print(f"{y:<6}" + "".join(f"{cols[n].get(y, float('nan')):>+10.3f}"
                                      for n in names))
        print(f"{'mean':<6}" + "".join(
            f"{np.nanmean(list(cols[n].values())):>+10.3f}" for n in names))


def main() -> int:
    piv = load_panel()
    tickers = list(piv.columns)
    dates = piv.index[piv.index >= START]
    M = membership_mask(piv.index, piv.columns)
    elig = (M & piv.notna()).loc[dates]
    piv_m = piv.where(M & piv.notna())
    fwd7 = (piv_m.shift(-7) / piv_m - 1).loc[dates]
    own5 = (piv / piv.shift(5) - 1).loc[dates]

    ev = load_events(tickers)
    n_files = sum((_UD / f"{t}.json").exists() for t in tickers)
    print(f"upgrade-downgrade: {n_files}/{len(tickers)} files, {len(ev)} events, "
          f"{ev.broker.nunique()} distinct brokers")

    B_by_year = {yr: broker_incidence(ev, tickers, yr) for yr in YEARS}
    W_by_year = {yr: cocov_from_incidence(B) for yr, B in B_by_year.items()}
    nbrok = {yr: pd.Series(B.sum(axis=0), index=tickers)
             for yr, B in B_by_year.items()}

    # self-check: AAPL-MSFT 2024 weight, brute force over brokers vs matrix entry
    lo, hi = pd.Timestamp("2022-01-01"), pd.Timestamp("2024-01-01")
    win = ev[(ev.date >= lo) & (ev.date < hi)]
    cov = win.groupby("broker").ticker.agg(set)
    brute = sum(1.0 / len(s) for s in cov if "AAPL" in s and "MSFT" in s)
    ia, im = tickers.index("AAPL"), tickers.index("MSFT")
    assert np.isclose(brute, W_by_year[2024][ia, im]), (brute, W_by_year[2024][ia, im])
    print(f"self-check ok: w(AAPL,MSFT|2024 graph) = {brute:.4f} "
          f"(shared brokers: {sum('AAPL' in s and 'MSFT' in s for s in cov)})")

    # coverage terciles (PIT: trailing-24m #brokers), among covered stocks
    bot_mask = pd.DataFrame(False, index=dates, columns=tickers)
    top_mask = pd.DataFrame(False, index=dates, columns=tickers)
    tercile_sets: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for yr in YEARS:
        nb = nbrok[yr]
        covered = nb[nb >= 1]
        lab = pd.qcut(covered.rank(method="first"), 3, labels=False)
        bot = np.isin(tickers, lab[lab == 0].index)
        top = np.isin(tickers, lab[lab == 2].index)
        tercile_sets[yr] = (bot, top)
        di = dates[dates.year == yr]
        bot_mask.loc[di] = bot
        top_mask.loc[di] = top
    bot_mask &= elig
    top_mask &= elig

    # density: member subgraph, full vs bottom tercile (old disaster: 121/149)
    print("\ngraph density on PIT members-with-data (deg = member neighbors):")
    print(f"{'year':<6}{'n_memb':>7}{'0brok':>7}{'med#b':>7}{'deg':>7}{'dens':>7}"
          f"{'| bot:med#b':>11}{'deg':>7}{'dens':>7}")
    for yr in YEARS:
        di = dates[dates.year == yr]
        if not len(di):
            continue
        memb = elig.loc[di].any().to_numpy()
        n = int(memb.sum())
        Wm = W_by_year[yr][np.ix_(memb, memb)]
        deg = (Wm > 0).sum(axis=1)
        nb = nbrok[yr].to_numpy()[memb]
        bot = tercile_sets[yr][0][memb]
        db, nbb = deg[bot], nb[bot]
        print(f"{yr:<6}{n:>7}{int((nb == 0).sum()):>7}"
              f"{np.median(nb[nb > 0]):>7.0f}{deg.mean():>7.0f}"
              f"{deg.mean() / (n - 1):>7.1%}{np.median(nbb):>11.0f}"
              f"{db.mean():>7.0f}{db.mean() / (n - 1):>7.1%}")

    # features: full graph + directed (neighbors restricted to top tercile)
    Wdir_by_year = {yr: W * tercile_sets[yr][1][:, None]  # zero non-top NEIGHBOR rows
                    for yr, W in W_by_year.items()}
    feats, dfeats = {}, {}
    for k in KS:
        Rk = (piv / piv.shift(k) - 1).loc[dates]
        feats[f"cf{k}"] = neighbor_signal(Rk, W_by_year)
        dfeats[f"dir{k}"] = neighbor_signal(Rk, Wdir_by_year)

    ic_block("FULL PIT cross-section", feats, fwd7, own5, elig)
    ic_block("BOTTOM coverage tercile (the honest test)", feats, fwd7, own5, bot_mask)
    ic_block("DIRECTED top-tercile neighbors -> bottom-tercile targets",
             dfeats, fwd7, own5, bot_mask)

    print("\ncaveats: broker-level (not analyst-level) co-coverage; delisted holes "
          "missing from the PIT panel; tercile labels use the trailing-24m window "
          "ending Dec 31 prior year (PIT); Finnhub feed depth thins pre-2016.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
