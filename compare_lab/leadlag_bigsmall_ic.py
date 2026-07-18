"""Big->small intra-industry lead-lag (Hou, RFS 2007) on the free PIT S&P universe.

Hypothesis: within an industry, returns of the biggest names lead the smaller ones
at ~weekly horizon. Honest expectations: post-publication decay + an all-S&P
universe (no true small caps) => small-tercile IC maybe 0.005-0.02, full
cross-section ~0. A null is a fine result.

Industry key: finnhubIndustry from data/finnhub_profiles/{T}.json (coarse, ~60
GICS-like buckets — the free granularity we have; noted honestly).
Size proxy: trailing-63d mean dollar volume (adjusted close x share volume);
no free PIT market caps. Volume: top-150 store + data/yf_prices_sp500/volume.parquet.

Signal (all inputs <= t): for PIT-member stock i on day t, leaders = top-3 by
trailing dollar volume within i's industry among that day's eligible members
(industry must have >=5 eligible members, else NaN). Signal = equal-weight
leaders' past-5d return, excluding i itself when i is a leader. Variants:
past-10d; leader-minus-own (sig5 - own past-5d); residualized on own mom10 +
own past-5d per day.

Eval: pit_bounding_backtest conventions — daily cross-sectional Spearman vs raw
7d fwd return on the PIT-eligible panel, per-year 2017-2026 from START
2017-07-01; full cross-section AND size-tercile slices (the effect should live
in the SMALL tercile); orthogonality vs own-mom10 and the mom/rev/pead combo.

    uv run python -m compare_lab.leadlag_bigsmall_ic
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_lab.cvar_conformal_backtest import START
from compare_lab.pit_bounding_backtest import (
    combo_from_raw, membership_mask, raw_signals,
)

_PROF = Path("data/finnhub_profiles")
_YF = Path("data/yf_prices_sp500/prices.parquet")
_VOL = Path("data/yf_prices_sp500/volume.parquet")
_P150 = Path("data/qflib_data_store_top150/prices_top150.parquet")
MIN_IND = 5          # min eligible members for an industry-day
N_LEAD = 3           # leaders per industry
DV_WIN, DV_MINP = 63, 40


# ---------------------------------------------------------------- panels
def build_panels() -> tuple[pd.DataFrame, pd.DataFrame]:
    """(piv_full, vol_full): adjusted closes + share volume, top-150 calendar."""
    t150 = pd.read_parquet(_P150, columns=["date", "ticker", "Close", "Volume"])
    t150["date"] = pd.to_datetime(t150["date"])
    piv150 = t150.pivot(index="date", columns="ticker", values="Close").sort_index()
    vol150 = t150.pivot(index="date", columns="ticker", values="Volume").sort_index()

    yf = pd.read_parquet(_YF)
    yf["date"] = pd.to_datetime(yf["date"])
    piv_x = yf.pivot(index="date", columns="ticker", values="Close").sort_index()
    piv_x = piv_x.reindex(piv150.index)
    vf = pd.read_parquet(_VOL)
    vf["date"] = pd.to_datetime(vf["date"])
    vol_x = vf.pivot(index="date", columns="ticker", values="Volume").sort_index()
    vol_x = vol_x.reindex(piv150.index)

    # drop yf columns duplicating a top-150 name (incl. class-share alias BRK.B/BRK-B)
    norm150 = {c.replace(".", "-") for c in piv150.columns}
    dup = [c for c in piv_x.columns if c.replace(".", "-") in norm150]
    piv_x = piv_x.drop(columns=dup)
    vol_x = vol_x.drop(columns=[c for c in dup if c in vol_x.columns])

    piv_full = pd.concat([piv150, piv_x], axis=1).sort_index(axis=1)
    vol_full = pd.concat([vol150, vol_x], axis=1).reindex(
        columns=piv_full.columns)
    return piv_full, vol_full


def industry_map(cols: pd.Index) -> pd.Series:
    out = {}
    for t in cols:
        p = _PROF / f"{t}.json"
        if not p.exists():
            continue
        ind = json.loads(p.read_text()).get("finnhubIndustry")
        if ind:
            out[t] = ind
    return pd.Series(out)


# ---------------------------------------------------------------- signal
def build_leadlag(piv: pd.DataFrame, elig: pd.DataFrame, dv63: pd.DataFrame,
                  ind: pd.Series, sample_date: pd.Timestamp,
                  ) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, list[str]]]:
    """sig5, sig10 (leaders' past-5d/10d mean return, self-excluded), plus the
    leader sets recorded on sample_date for the hand-check assert."""
    cols = piv.columns
    r5 = (piv / piv.shift(5) - 1).to_numpy()
    r10 = (piv / piv.shift(10) - 1).to_numpy()
    E = elig.to_numpy()
    DV = dv63.to_numpy()
    groups = {k: np.flatnonzero(cols.isin(g.index))
              for k, g in ind.groupby(ind)}
    sig5 = np.full(r5.shape, np.nan)
    sig10 = np.full(r5.shape, np.nan)
    ti0 = int(piv.index.searchsorted("2016-06-01"))  # warmup; eval starts 2017-07
    ti_samp = int(piv.index.get_loc(sample_date))
    rec: dict[str, list[str]] = {}
    for ti in range(ti0, len(piv.index)):
        for k, cidx in groups.items():
            sub = cidx[E[ti, cidx]]
            if sub.size < MIN_IND:
                continue
            ok = (np.isfinite(DV[ti, sub]) & np.isfinite(r5[ti, sub])
                  & np.isfinite(r10[ti, sub]))
            cand = sub[ok]
            if cand.size < 2:
                continue
            lead = cand[np.argsort(-DV[ti, cand])[:N_LEAD]]
            if ti == ti_samp:
                rec[k] = [cols[j] for j in lead]
            n = lead.size
            s5, s10 = r5[ti, lead].sum(), r10[ti, lead].sum()
            sig5[ti, sub], sig10[ti, sub] = s5 / n, s10 / n
            for c in lead:                       # self-excluded for leaders
                sig5[ti, c] = (s5 - r5[ti, c]) / (n - 1)
                sig10[ti, c] = (s10 - r10[ti, c]) / (n - 1)
    mk = lambda a: pd.DataFrame(a, index=piv.index, columns=cols)  # noqa: E731
    return mk(sig5), mk(sig10), rec


def residualize(sig: pd.DataFrame, x1: pd.DataFrame, x2: pd.DataFrame,
                ) -> pd.DataFrame:
    """Per-day cross-sectional OLS residual of sig on [1, x1, x2]."""
    Y, X1, X2 = sig.to_numpy(), x1.to_numpy(), x2.to_numpy()
    out = np.full(Y.shape, np.nan)
    for ti in range(Y.shape[0]):
        m = np.isfinite(Y[ti]) & np.isfinite(X1[ti]) & np.isfinite(X2[ti])
        if m.sum() < 20:
            continue
        A = np.column_stack([np.ones(m.sum()), X1[ti, m], X2[ti, m]])
        beta, *_ = np.linalg.lstsq(A, Y[ti, m], rcond=None)
        out[ti, m] = Y[ti, m] - A @ beta
    return pd.DataFrame(out, index=sig.index, columns=sig.columns)


# ---------------------------------------------------------------- eval
def daily_spearman(a: pd.DataFrame, b: pd.DataFrame,
                   idx: pd.DatetimeIndex) -> pd.Series:
    return a.loc[idx].rank(axis=1).corrwith(b.loc[idx].rank(axis=1), axis=1)


def ic_cols(sig: pd.DataFrame, fwd: pd.DataFrame, idx: pd.DatetimeIndex,
            masks: dict[str, pd.DataFrame | None]) -> pd.DataFrame:
    """Per-year IC table, one column per universe slice + mean/t rows."""
    out = {}
    for name, mk in masks.items():
        s = sig.where(mk) if mk is not None else sig
        f = fwd.where(mk) if mk is not None else fwd
        ics = daily_spearman(s, f, idx)
        col = ics.groupby(ics.index.year).mean()
        n = int(ics.notna().sum())
        col.loc["mean"] = ics.mean()
        col.loc["t"] = ics.mean() / ics.std() * np.sqrt(n) if n > 2 else np.nan
        out[name] = col
    return pd.DataFrame(out)


def main() -> int:
    piv, vol = build_panels()
    dates = piv.index
    ind = industry_map(piv.columns)
    M = membership_mask(dates, piv.columns)
    elig = M & piv.notna()
    dv63 = (piv * vol).rolling(DV_WIN, min_periods=DV_MINP).mean()

    # ---------- coverage
    known = piv.columns.intersection(ind.index)
    sizes = ind.value_counts()
    e_ind = elig[known]
    memb = e_ind.T.groupby(ind[known]).sum().T      # day x industry member counts
    pass5 = (memb >= MIN_IND)
    print(f"coverage: {len(known)}/{len(piv.columns)} tickers with finnhubIndustry, "
          f"{len(sizes)} industries (static sizes: median {sizes.median():.0f}, "
          f"max {sizes.max()} [{sizes.idxmax()}], "
          f"{(sizes >= MIN_IND).sum()} industries with >={MIN_IND} names)")
    row = pass5.loc[START:]
    print(f"per-day (from {START}): industries passing >={MIN_IND}: "
          f"{row.sum(axis=1).mean():.1f}, members covered: "
          f"{memb.where(pass5).sum(axis=1).loc[START:].mean():.0f}")

    # ---------- signal
    samp = dates[dates.searchsorted("2021-01-04")]
    sig5, sig10, rec = build_leadlag(piv, elig, dv63, ind, samp)
    r5own = piv / piv.shift(5) - 1
    mom10 = piv / piv.shift(10) - 1
    sigdiff = sig5 - r5own

    # hand-check leader construction on one industry-day
    big_ind = max(rec, key=lambda k: (ind == k).sum())
    members = [c for c in known if ind[c] == big_ind and elig.loc[samp, c]
               and np.isfinite(dv63.loc[samp, c])
               and np.isfinite(r5own.loc[samp, c])
               and np.isfinite(mom10.loc[samp, c])]
    expect = dv63.loc[samp, members].nlargest(N_LEAD).index
    assert set(expect) == set(rec[big_ind]), \
        f"leader mismatch {samp.date()} {big_ind}: {list(expect)} vs {rec[big_ind]}"
    print(f"assert OK: leaders {samp.date()} '{big_ind}' = {rec[big_ind]} "
          f"(top-{N_LEAD} dollar volume of {len(members)} members)")

    # ---------- eval
    piv_C = piv.where(elig)
    fwd7 = piv_C.shift(-7) / piv_C - 1
    idx = dates[(dates >= START) & (dates <= dates[-8])]
    pct_dv = dv63.where(elig).rank(axis=1, pct=True)
    masks = {"full": None, "small": pct_dv <= 1 / 3, "big": pct_dv > 2 / 3}

    print("\nper-year daily cross-sectional Spearman IC vs raw 7d fwd "
          "(small/big = bottom/top dollar-volume tercile):")
    tabs = {"sig5 (leaders past-5d)": sig5,
            "sig10 (leaders past-10d)": sig10,
            "diff (sig5 - own past-5d)": sigdiff,
            "resid5 (sig5 | mom10, own5d)": residualize(sig5, mom10, r5own),
            "resid10 (sig10 | mom10, own5d)": residualize(sig10, mom10, r5own)}
    for name, s in tabs.items():
        tab = ic_cols(s, fwd7, idx, masks)
        print(f"\n{name}")
        with pd.option_context("display.float_format", "{:+.4f}".format):
            print(tab.to_string())

    # ---------- orthogonality
    sigs_full = raw_signals(piv)
    combo = combo_from_raw(sigs_full, elig)
    print("\northogonality (mean daily cross-sectional Spearman with sig5):")
    for nm, other in [("own mom10", mom10.where(elig)),
                      ("own past-5d", r5own.where(elig)),
                      ("combo (mom/rev/pead)", combo)]:
        print(f"  {nm:<22}{daily_spearman(sig5, other, idx).mean():+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
