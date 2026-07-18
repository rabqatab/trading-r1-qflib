"""Event study: post-announcement drift after broker upgrades/downgrades on the
top-150 large caps (data/finnhub_upgrade_downgrade, 2013->2026; prices 2015->2026-05).

Design (JKKL JF 2004 framing: rec CHANGES carry info; large caps adjust fast):
  * events: action in {up, down} (init as a small third group; main/reit skipped)
  * timing: gradeTime epoch -> US/Eastern. 98.7% of stamps are 00:00 UTC =
    19/20:00 ET the previous day, i.e. after-close -> effective next day, which is
    exactly the stamp's UTC date. Rule: ET time > 16:00 -> effective next calendar
    day; else same day. day 0 = first trading day >= effective date.
  * abnormal return AR = stock daily close-to-close return - equal-weight mean of
    the 150-stock universe that day. Windows: AR(0), AR(+1), CAR(+1..+5),
    CAR(+1..+21); full non-NaN window required.
  * t-stats: cross-event iid SE and calendar-day(day-0)-clustered SE (events
    cluster by announcement day; overlap across days in the +21 window remains,
    so the clustered t is still mildly optimistic for CAR(+1..+21)).
  * cuts: sign; grade-jump >= 2 notches on a 5-point scale; broker-frequency
    tercile; 2015-2019 vs 2020-2026 (prices start 2015 -> pre-2015 events drop).
  * tradeable check: CAR(+1..+5) — day 0 is not capturable. Naive overlay: each
    day, long all tickers upgraded in the last 5 trading days (days +1..+5),
    short the recent downgrades, equal weight per side, r = (L - S)/2, cash when
    a side is empty. Zero costs, close-to-close.

caveats: today's-top-150 universe (survivorship), zero costs, same-close exec.

    uv run python -m compare_lab.broker_event_study
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401

_EVENTS = Path("data/finnhub_upgrade_downgrade")
_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")

_SCORE = {}
for g, s in [
    (("strong buy", "conviction buy", "top pick", "action list buy"), 2),
    (("buy", "overweight", "outperform", "market outperform", "sector outperform",
      "outperformer", "positive", "accumulate", "add", "long-term buy",
      "speculative buy", "moderate buy"), 1),
    (("hold", "neutral", "equal-weight", "equal weight", "market perform",
      "sector perform", "in-line", "in line", "peer perform", "perform",
      "sector weight", "market weight", "mixed", "fair value"), 0),
    (("underweight", "underperform", "sell", "reduce", "negative",
      "moderate sell", "sector underperform", "market underperform",
      "underperformer", "below average", "cautious"), -1),
    (("strong sell", "conviction sell"), -2),
]:
    for k in g:
        _SCORE[k] = s

WINDOWS = [("AR0", 0, 0), ("AR+1", 1, 1), ("CAR+1..+5", 1, 5), ("CAR+1..+21", 1, 21)]


def load_events() -> pd.DataFrame:
    rows = []
    for p in sorted(_EVENTS.glob("*.json")):
        rows += json.loads(p.read_text())
    ev = pd.DataFrame(rows)
    ev = ev[ev.action.isin(["up", "down", "init"])].copy()
    ts = pd.to_datetime(ev.gradeTime, unit="s", utc=True).dt.tz_convert("America/New_York")
    eff = ts.dt.normalize() + (ts.dt.hour >= 16) * pd.Timedelta(days=1)
    ev["eff_date"] = eff.dt.tz_localize(None)
    ev["notch"] = (ev.toGrade.str.lower().str.strip().map(_SCORE)
                   - ev.fromGrade.str.lower().str.strip().map(_SCORE)).abs()
    return ev


def main() -> int:
    px = pd.read_parquet(_PRICES, columns=["date", "ticker", "Close"])
    px["date"] = pd.to_datetime(px["date"])
    piv = px.pivot(index="date", columns="ticker", values="Close").sort_index()
    ret = piv.pct_change()
    ar = ret.sub(ret.mean(axis=1), axis=0)          # AR vs equal-weight-150
    dates = piv.index
    arv, cols = ar.to_numpy(), {t: j for j, t in enumerate(ar.columns)}

    ev = load_events()
    i0 = dates.searchsorted(ev.eff_date.to_numpy())
    ev["i0"] = i0
    ok = (i0 > 0) & (i0 < len(dates))
    ok &= (dates.to_numpy()[np.clip(i0, 0, len(dates) - 1)]
           - ev.eff_date.to_numpy()) <= np.timedelta64(5, "D")
    ev = ev[ok & ev.symbol.isin(cols)].copy()
    ev["day0"] = dates[ev.i0]

    # per-window event AR/CAR (NaN unless full window is in range and clean)
    for name, a, b in WINDOWS:
        v = np.full(len(ev), np.nan)
        for k, (j0, t) in enumerate(zip(ev.i0, ev.symbol)):
            if j0 + b < len(dates):
                w = arv[j0 + a: j0 + b + 1, cols[t]]
                if not np.isnan(w).any():
                    v[k] = w.sum()
        ev[name] = v

    # self-check: hand-computed AR0 for one known event (first AAPL downgrade)
    e = ev[(ev.symbol == "AAPL") & (ev.action == "down")].iloc[0]
    d0 = e.day0
    i = dates.get_loc(d0)
    hand = (piv.loc[d0, "AAPL"] / piv.iloc[i - 1]["AAPL"] - 1
            - (piv.loc[d0] / piv.iloc[i - 1] - 1).mean())
    assert np.isclose(hand, e.AR0, atol=1e-12), (hand, e.AR0)
    print(f"self-check OK: AAPL down {d0.date()} AR0 hand={hand:+.4%} == matrix={e.AR0:+.4%}\n")

    nb = ev[ev.action.isin(["up", "down"])].groupby("company").size()
    q1, q2 = nb.quantile([1 / 3, 2 / 3])
    ev["btier"] = ev.company.map(
        lambda c: "top" if nb.get(c, 0) > q2 else ("mid" if nb.get(c, 0) > q1 else "tail"))
    ev["late"] = ev.day0.dt.year >= 2020

    def row(label: str, g: pd.DataFrame) -> None:
        out = f"  {label:<28}{len(g):>6}"
        for name, *_ in WINDOWS:
            x = g[name].dropna()
            if len(x) < 10:
                out += f"{'--':>26}"
                continue
            m = x.mean()
            t_iid = m / (x.std(ddof=1) / np.sqrt(len(x)))
            cl = (x - m).groupby(g.loc[x.index, "day0"]).sum()
            t_cl = m / (np.sqrt((cl ** 2).sum()) / len(x))
            out += f"{m:>+9.2%} ({t_iid:>+5.1f}/{t_cl:>+5.1f})"
        print(out)

    hdr = f"  {'cut':<28}{'N':>6}" + "".join(f"{n + ' (t_iid/t_cl)':>26}" for n, *_ in WINDOWS)
    print(hdr + "\n" + "  " + "-" * (len(hdr) - 2))
    for act in ["up", "down", "init"]:
        g = ev[ev.action == act]
        row(f"{act.upper()} all", g)
        if act == "init":
            continue
        row(f"{act} notch>=2", g[g.notch >= 2])
        for tier in ["top", "mid", "tail"]:
            row(f"{act} broker-{tier}", g[g.btier == tier])
        row(f"{act} 2015-2019", g[~g.late])
        row(f"{act} 2020-2026", g[g.late])
    print("\n  (broker terciles by up/down event count: "
          f"top>{q2:.0f} ev, mid>{q1:.0f}, tail<= | notch on 5-pt grade scale)")

    # joint tradeable test: up-minus-down spread in CAR(+1..+5)
    u = ev.loc[ev.action == "up", "CAR+1..+5"].dropna()
    d = ev.loc[ev.action == "down", "CAR+1..+5"].dropna()
    diff = u.mean() - d.mean()
    se = np.sqrt(u.var(ddof=1) / len(u) + d.var(ddof=1) / len(d))
    print(f"\nup-minus-down CAR(+1..+5) spread: {diff:+.2%}  t_iid {diff / se:+.2f}"
          f"  (Nu={len(u)}, Nd={len(d)})")

    # naive overlay: hold days +1..+5 after day 0, long ups / short downs
    retv = ret.to_numpy()
    L = np.zeros((len(dates), len(cols)), bool)
    S = np.zeros_like(L)
    for act, M in [("up", L), ("down", S)]:
        for j0, t in zip(ev.i0[ev.action == act], ev.symbol[ev.action == act]):
            M[j0 + 1: j0 + 6, cols[t]] = True
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)  # empty legs -> NaN
        lr = np.nanmean(np.where(L, retv, np.nan), axis=1)
        sr = np.nanmean(np.where(S, retv, np.nan), axis=1)
    both = ~np.isnan(lr) & ~np.isnan(sr)                 # cash unless both legs live
    ls = pd.Series(np.where(both, (lr - sr) / 2, 0.0), index=dates)
    print("\nnaive 5-day overlay (long fresh ups / short fresh downs, gross 1, "
          "cash when a leg is empty, zero cost):")
    for lab, sl in [("full 2015-2026", slice(None)),
                    ("2015-2019", slice("2015", "2019")),
                    ("2020-2026", slice("2020", None))]:
        r, b = ls.loc[sl], pd.Series(both, index=dates).loc[sl]
        ann, vol = r.mean() * 252, r.std() * np.sqrt(252)
        print(f"  {lab:<16} ann {ann:>+7.2%}  vol {vol:>6.2%}  Sharpe {ann / vol:>+5.2f}"
              f"  active days {int(b.sum())}/{len(r)}")

    print("\ncaveats: survivorship (today's top-150), zero costs, close-to-close, "
          "overlapping +21 windows -> clustered t still mildly optimistic there.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
