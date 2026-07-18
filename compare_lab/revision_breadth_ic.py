"""Does Finnhub's finer analyst data add signal beyond the 63d rec-revision signal?

FEASIBILITY (probed 2026-07): /stock/eps-estimate and /stock/revenue-estimate return one
consensus row per FISCAL PERIOD with no as-of dates and no revised-up/down counts — a
current snapshot per period, so historical revision breadth CANNOT be backfilled
point-in-time. /stock/price-target is a pure current snapshot (skip backtest).
/stock/upgrade-downgrade DOES return dated broker-level history (back to ~2015-01) —
that's what we test here.

Signal "ud" (upgrade-downgrade breadth), per ticker, on the trading-day index:
    ud_t = (#up − #down over trailing 63 trading days)
           / (#distinct covering brokers over trailing 252 trading days)
where an event counts on the first trading day >= its gradeTime date, "covering" means
any record (up/down/main/init/reit), and ud is NaN when no broker touched the name in a
year. Evaluation follows the house convention (cvar_conformal_backtest): daily
cross-sectional Spearman of signal vs RAW 7-row forward return, per-year 2017-2026,
dates restricted to the combo LS index (starts 2017-07).

    uv run python -m compare_lab.revision_breadth_ic
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.cvar_conformal_backtest import build_signals, ls_returns

_UD = Path("data/finnhub_upgrade_downgrade")
WIN_UD = 63    # trailing trading-day window for up/down counts
WIN_COV = 252  # trailing trading-day window for #covering brokers


def _events(t: str, dates: pd.DatetimeIndex) -> pd.DataFrame | None:
    """Raw events -> (di, action, company) with di = first trading-day pos >= date."""
    p = _UD / f"{t}.json"
    if not p.exists():
        return None
    rows = json.loads(p.read_text())
    if not rows:
        return None
    ev = pd.DataFrame(rows)
    ev["date"] = pd.to_datetime(ev["gradeTime"], unit="s").dt.normalize()
    ev = ev[ev["date"] >= dates[0]]
    di = np.searchsorted(dates.values, ev["date"].values)
    ev = ev.assign(di=di)[di < len(dates)]
    return ev if len(ev) else None


def build_ud_signal(dates: pd.DatetimeIndex, columns) -> pd.DataFrame:
    sig = pd.DataFrame(index=dates, columns=columns, dtype=float)
    n = len(dates)
    for t in columns:
        ev = _events(t, dates)
        if ev is None:
            continue
        up = np.zeros(n)
        dn = np.zeros(n)
        np.add.at(up, ev.di[ev.action == "up"].to_numpy(), 1)
        np.add.at(dn, ev.di[ev.action == "down"].to_numpy(), 1)
        net63 = pd.Series(up - dn, index=dates).rolling(WIN_UD).sum()
        # distinct covering brokers over trailing WIN_COV days (two-pointer)
        evs = sorted(zip(ev.di, ev.company))
        cov = np.zeros(n)
        cnt: Counter = Counter()
        lo = hi = 0
        for i in range(n):
            while hi < len(evs) and evs[hi][0] <= i:
                cnt[evs[hi][1]] += 1
                hi += 1
            while lo < hi and evs[lo][0] < i - WIN_COV + 1:
                c = evs[lo][1]
                cnt[c] -= 1
                if not cnt[c]:
                    del cnt[c]
                lo += 1
            cov[i] = len(cnt)
        sig[t] = net63 / pd.Series(cov, index=dates).replace(0, np.nan)
    return sig


def _self_check(sig: pd.DataFrame, dates: pd.DatetimeIndex, t: str = "AAPL",
                day: str = "2024-06-28") -> None:
    """Brute-force breadth from raw JSON for one ticker-date; must match pipeline."""
    d0 = dates.get_loc(pd.Timestamp(day))
    ev = _events(t, dates)
    in63 = ev[(ev.di >= d0 - WIN_UD + 1) & (ev.di <= d0)]
    in252 = ev[(ev.di >= d0 - WIN_COV + 1) & (ev.di <= d0)]
    expected = ((in63.action == "up").sum() - (in63.action == "down").sum()) \
        / in252.company.nunique()
    got = sig.loc[pd.Timestamp(day), t]
    assert abs(got - expected) < 1e-12, (got, expected)
    print(f"self-check OK: {t} {day} breadth = {got:+.4f} "
          f"(up-down={((in63.action == 'up').sum() - (in63.action == 'down').sum()):+d}, "
          f"brokers={in252.company.nunique()})")


def _yearly_ic(sig: pd.DataFrame, fwd7: pd.DataFrame) -> pd.Series:
    out = {}
    for yr, idx in fwd7.groupby(fwd7.index.year).groups.items():
        s = sig.loc[idx]
        out[yr] = s.rank(axis=1).corrwith(fwd7.loc[idx].rank(axis=1), axis=1).mean()
    return pd.Series(out)


def _xcorr(a: pd.DataFrame, b: pd.DataFrame, idx) -> float:
    """Mean daily cross-sectional Spearman between two signals."""
    return a.loc[idx].rank(axis=1).corrwith(b.loc[idx].rank(axis=1), axis=1).mean()


def main() -> int:
    piv, sigs = build_signals()
    dates = piv.index
    ud = build_ud_signal(dates, piv.columns)
    _self_check(ud, dates)

    idx = ls_returns(piv, sigs["combo"]).index  # house eval dates (2017-07 ..)
    fwd7 = (piv.shift(-7) / piv - 1).loc[idx]

    ranks4 = [sigs[k].rank(axis=1) for k in ("mom", "rev", "pead")] + [ud.rank(axis=1)]
    combo4 = pd.concat(ranks4).groupby(level=0).mean()

    cols = {"mom": sigs["mom"], "rev": sigs["rev"], "pead": sigs["pead"], "ud": ud,
            "combo3": sigs["combo"], "combo4": combo4}
    tab = pd.DataFrame({k: _yearly_ic(v, fwd7) for k, v in cols.items()})
    tab.loc["ALL"] = pd.Series(
        {k: v.loc[idx].rank(axis=1).corrwith(fwd7.rank(axis=1), axis=1).mean()
         for k, v in cols.items()})
    print("\nper-year daily cross-sectional Spearman IC vs raw 7d fwd return "
          "(survivorship-biased top-150 universe):")
    print(tab.to_string(float_format=lambda x: f"{x:+.3f}"))

    print("\nmean daily cross-sectional Spearman between signals (2017-07..):")
    for name, other in (("rev", sigs["rev"]), ("mom", sigs["mom"]),
                        ("pead", sigs["pead"])):
        print(f"  corr(ud, {name:<4}) = {_xcorr(ud, other, idx):+.3f}")

    print("\nud coverage: mean names/day with signal ="
          f" {ud.loc[idx].notna().sum(axis=1).mean():.0f} / {len(piv.columns)}")
    print("caveats: same-day event inclusion (gradeTime is announce date, usually "
          "pre-market — matches house same-close convention); today's-top-150 universe "
          "(survivorship); Finnhub upgrade-downgrade history starts ~2015-01, may be "
          "incomplete broker coverage in early years; price-target & estimate-revision "
          "breadth untestable (snapshots only).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
