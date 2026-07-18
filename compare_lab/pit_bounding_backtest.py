"""PIT bounding re-test: how much does the decade combo-LS result move on a
point-in-time-membership universe of still-listed names?

Universe arms (all daily dollar-neutral top/bottom-quintile LS on the combo rank,
START 2017-07-01, same conventions as cvar_conformal_backtest):
  A  current top-150 as-is (must reproduce gross Sharpe 0.72 +- 0.05)
  B  top-150 INTERSECT point-in-time S&P membership each day
  C  full PIT still-listed universe (every member with price data that day)

For C additionally: per-year daily cross-sectional combo IC vs raw 7d fwd;
cost-realistic variant (band 20/35 entry/exit, 3 bps one-way + 35 bps/yr short
borrow) with ru_conformal() on the net stream = the PIT net controlled headline;
and a bootstrap sensitivity band for the delisted holes (drop a random matching
fraction of AVAILABLE names per year, 50 draws).

Prices: data/qflib_data_store_top150/prices_top150.parquet (our 150, adjusted)
      + data/yf_prices_sp500/prices.parquet (extra PIT names, yfinance adjusted).
Membership: data/sp500_constituents/sp500_history.csv (last snapshot <= date),
tickers translated through fetch_pit_prices.RENAME. Delisted holes -> NaN signals.

    uv run python -m compare_lab.pit_bounding_backtest
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from compare_lab.cvar_conformal_backtest import (
    BETA, START, build_signals, ls_returns, ru_conformal,
)
from compare_lab.cost_haircut_backtest import band_weights, stat_row, turnover
from compare_lab.fetch_pit_prices import SKIP, canon

_HIST = Path("data/sp500_constituents/sp500_history.csv")
_YF = Path("data/yf_prices_sp500/prices.parquet")
_RECS = Path("data/finnhub_recs")
_EARN = Path("data/finnhub_earnings_full")
HEADLINE_BPS = 3
BORROW_DAILY = 0.0035 * 0.5 / 252
N_BOOT, SEED = 50, 0


# ---------------------------------------------------------------- membership
def membership_mask(dates: pd.DatetimeIndex, columns: pd.Index) -> pd.DataFrame:
    """True where ticker (canonical symbol) is an S&P member on date (last
    snapshot <= date)."""
    h = pd.read_csv(_HIST)
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date").reset_index(drop=True)
    snap_idx = h["date"].searchsorted(dates, side="right") - 1
    sets = [frozenset(canon(t) for t in row.split(",") if t not in SKIP)
            for row in h["tickers"]]
    M = np.zeros((len(dates), len(columns)), dtype=bool)
    col_pos = {c: j for j, c in enumerate(columns)}
    prev = -1
    for i, si in enumerate(snap_idx):
        if si < 0:
            continue
        if si == prev:
            M[i] = M[i - 1]
            continue
        row = np.zeros(len(columns), dtype=bool)
        for t in sets[si]:
            j = col_pos.get(t)
            if j is not None:
                row[j] = True
        M[i] = row
        prev = si
    return pd.DataFrame(M, index=dates, columns=columns)


def member_universe_size(dates: pd.DatetimeIndex) -> pd.Series:
    """Total members per date (before any data-availability filter)."""
    h = pd.read_csv(_HIST)
    h["date"] = pd.to_datetime(h["date"])
    h = h.sort_values("date").reset_index(drop=True)
    snap_idx = h["date"].searchsorted(dates, side="right") - 1
    sizes = [len({canon(t) for t in h["tickers"].iloc[si].split(",") if t not in SKIP})
             if si >= 0 else 0 for si in snap_idx]
    return pd.Series(sizes, index=dates)


# ---------------------------------------------------------------- signals
def raw_signals(piv: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Same construction as cvar_conformal_backtest.build_signals but RAW values
    (no ranking) on an arbitrary price panel, so masks can be applied first."""
    dates = piv.index
    mom = piv / piv.shift(10) - 1

    rev = pd.DataFrame(index=dates, columns=piv.columns, dtype=float)
    for t in piv.columns:
        p = _RECS / f"{t}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text())
        if not d:
            continue
        df = pd.DataFrame(d)
        df["period"] = pd.to_datetime(df["period"])
        tot = df.strongBuy + df.buy + df.hold + df.sell + df.strongSell
        s = ((df.strongBuy * 2 + df.buy - df.sell - df.strongSell * 2)
             / tot.replace(0, np.nan))
        s.index = df["period"]
        s = s.sort_index()
        s = s[~s.index.duplicated(keep="last")]
        daily = s.reindex(dates, method="ffill")
        rev[t] = daily - daily.shift(63)

    pead = pd.DataFrame(index=dates, columns=piv.columns, dtype=float)
    for t in piv.columns:
        p = _EARN / f"{t}.json"
        if not p.exists():
            continue
        rows = [(pd.Timestamp(a["date"]),
                 (a["epsActual"] - a["epsEstimate"]) / abs(a["epsEstimate"]))
                for a in json.loads(p.read_text())
                if a.get("epsActual") is not None and a.get("epsEstimate")]
        if not rows:
            continue
        ev = pd.Series(dict(rows)).sort_index()
        ev = ev[~ev.index.duplicated(keep="last")]
        daily = ev.reindex(dates, method="ffill")
        last = pd.Series(ev.index, index=ev.index).reindex(dates, method="ffill")
        daily[(dates - last) > pd.Timedelta(days=90)] = np.nan
        pead[t] = daily
    return {"mom": mom, "rev": rev, "pead": pead}


def combo_from_raw(sigs: dict[str, pd.DataFrame],
                   mask: pd.DataFrame | None) -> pd.DataFrame:
    """Rank each raw signal within the (masked) daily cross-section, NaN-tolerant
    mean of ranks — identical maths to build_signals, but universe-aware."""
    ranks = []
    for s in (sigs["mom"], sigs["rev"], sigs["pead"]):
        v = s.where(mask) if mask is not None else s
        ranks.append(v.rank(axis=1))
    return pd.concat(ranks).groupby(level=0).mean()


# ---------------------------------------------------------------- reporting
def fmt(name: str, st: dict) -> str:
    return (f"{name:<34}{st['ann']:>+8.2%}{st['vol']:>8.2%}{st['sharpe']:>7.2f}"
            f"{st['mdd']:>8.2%}{st['cvar']:>+9.3%}")


def yearly_ic(sig: pd.DataFrame, piv: pd.DataFrame, idx: pd.DatetimeIndex) -> pd.Series:
    fwd7 = (piv.shift(-7) / piv - 1).loc[idx]
    out = {}
    for yr, ii in fwd7.groupby(fwd7.index.year).groups.items():
        ics = sig.loc[ii].rank(axis=1).corrwith(fwd7.loc[ii].rank(axis=1), axis=1)
        out[yr] = ics.mean()
    return pd.Series(out)


def main() -> int:
    # ---------- panels
    piv150, sigs150 = build_signals()          # untouched original path (arm A)
    yf = pd.read_parquet(_YF)
    yf["date"] = pd.to_datetime(yf["date"])
    piv_x = yf.pivot(index="date", columns="ticker", values="Close").sort_index()
    piv_x = piv_x.reindex(piv150.index)        # master calendar = top-150 dates
    dup = piv_x.columns.intersection(piv150.columns)
    if len(dup):
        piv_x = piv_x.drop(columns=dup)
    piv_full = pd.concat([piv150, piv_x], axis=1).sort_index(axis=1)
    dates = piv_full.index

    M = membership_mask(dates, piv_full.columns)
    has_px = piv_full.notna()
    elig = M & has_px                          # PIT member AND has price that day

    # ---------- coverage table
    n_members = member_universe_size(dates)
    n_avail = elig.sum(axis=1)
    cov = pd.DataFrame({"members": n_members, "with_data": n_avail})
    cov = cov.loc[START:]
    cov_y = cov.groupby(cov.index.year).mean().round(1)
    cov_y["coverage"] = (cov["with_data"] / cov["members"]).groupby(
        cov.loc[START:].index.year).mean()
    print("per-year PIT coverage (mean daily members-with-data / members):")
    print(f"{'year':<6}{'members':>9}{'with_data':>11}{'coverage':>10}")
    for yr, row in cov_y.iterrows():
        print(f"{yr:<6}{row['members']:>9.0f}{row['with_data']:>11.0f}"
              f"{row['coverage']:>10.1%}")
    miss_frac_y = (1 - cov_y["coverage"]).to_dict()

    # ---------- signals on the full panel
    sigs_full = raw_signals(piv_full)
    combo_A = sigs150["combo"]
    M150 = M[piv150.columns]
    combo_B = combo_from_raw(
        {k: sigs_full[k][piv150.columns] for k in ("mom", "rev", "pead")}, M150)
    combo_C = combo_from_raw(sigs_full, elig)

    # ---------- arms
    piv_B = piv150.where(M150)                 # non-member fwd returns excluded too
    piv_C = piv_full.where(elig)
    rA = ls_returns(piv150, combo_A)
    rB = ls_returns(piv_B, combo_B)
    rC = ls_returns(piv_C, combo_C)
    stA = stat_row(rA.to_numpy())
    assert abs(stA["sharpe"] - 0.72) < 0.05, \
        f"arm A gross Sharpe {stA['sharpe']:.3f} != 0.72 +- 0.05"

    print(f"\nbacktest {rA.index[0].date()} .. {rA.index[-1].date()} "
          f"({len(rA)} days); arm C mean names/day "
          f"{elig.loc[rC.index].sum(axis=1).mean():.0f}")
    hdr = f"{'arm (gross, zero-cost)':<34}{'ann.ret':>8}{'vol':>8}{'Sharpe':>7}{'maxDD':>8}{'CVaR.85':>9}"
    print(hdr)
    print("-" * len(hdr))
    print(fmt("A: top-150 as-is", stA))
    print(fmt("B: top-150 x PIT membership", stat_row(rB.to_numpy())))
    print(fmt("C: full PIT still-listed", stat_row(rC.to_numpy())))

    # ---------- per-year combo IC on C
    print("\nper-year daily cross-sectional combo IC (arm C universe, raw 7d fwd):")
    icC = yearly_ic(combo_C, piv_C, rC.index)
    icA = yearly_ic(combo_A, piv150, rA.index)
    print(f"{'year':<6}{'IC C':>8}{'IC A':>8}")
    for yr in icC.index:
        print(f"{yr:<6}{icC[yr]:>+8.3f}{icA.get(yr, np.nan):>+8.3f}")
    print(f"{'mean':<6}{icC.mean():>+8.3f}{icA.mean():>+8.3f}")

    # ---------- cost-realistic C: band 20/35 @3bps + borrow, then RU control
    pctC = combo_C.rank(axis=1, pct=True)
    W = band_weights(pctC)
    fwd1 = piv_C.shift(-1) / piv_C - 1
    g = (W * fwd1.fillna(0.0)).sum(axis=1).loc[rC.index]
    to = turnover(W, piv_full).loc[rC.index]
    net = (g - to * HEADLINE_BPS * 1e-4 - BORROW_DAILY).to_numpy()
    st_net = stat_row(net)
    lam, _ = ru_conformal(-net)
    st_ctrl = stat_row(lam * net)
    print(f"\ncost-realistic C (band 20/35, {HEADLINE_BPS} bps one-way, "
          f"35 bps/yr borrow; to/day {to.mean():.3f}):")
    print(fmt("C net (banded)", st_net))
    print(fmt("C net + RU-conformal  [HEADLINE]", st_ctrl))
    print(f"  mean lambda {lam.mean():.2f}, realized CVaR.85 {st_ctrl['cvar']:+.3%}")

    # ---------- bootstrap sensitivity: drop matching fraction of available names
    rng = np.random.default_rng(SEED)
    years = pd.Series(dates.year, index=dates)
    avail_y = {yr: elig.loc[years == yr].any().pipe(lambda s: s[s].index.to_numpy())
               for yr in miss_frac_y}
    sh = []
    for b in range(N_BOOT):
        drop = pd.DataFrame(False, index=dates, columns=piv_full.columns)
        for yr, f in miss_frac_y.items():
            avail = avail_y[yr]
            k = int(round(f * len(avail)))
            if k:
                sel = rng.choice(avail, size=k, replace=False)
                drop.loc[years == yr, sel] = True
        elig_b = elig & ~drop
        combo_b = combo_from_raw(sigs_full, elig_b)
        r_b = ls_returns(piv_full.where(elig_b), combo_b)
        sh.append(stat_row(r_b.to_numpy())["sharpe"])
    q = np.quantile(sh, [0.05, 0.25, 0.5, 0.75, 0.95])
    print(f"\nbootstrap hole-sensitivity on C ({N_BOOT} draws, per-year matching "
          f"drop fraction of AVAILABLE names):")
    print(f"  gross Sharpe quantiles  5%:{q[0]:.2f}  25%:{q[1]:.2f}  "
          f"50%:{q[2]:.2f}  75%:{q[3]:.2f}  95%:{q[4]:.2f}   "
          f"(point est {stat_row(rC.to_numpy())['sharpe']:.2f})")
    print("\nbias direction (unquantifiable with free data): missing acquired "
          "winners -> long leg understated; missing delisted losers -> short leg "
          "understated; net direction unknown.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
