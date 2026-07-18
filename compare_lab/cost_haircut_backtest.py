"""Cost-haircut backtest: does the combo LS survive realistic costs, and which
turnover-reduction arm is best?

Arms (all dollar-neutral LS on the combo rank, gross book = 1):
  1. unbuffered      daily top/bottom quintile (current behavior, baseline)
  2. band 20/35      enter at top/bottom 20 %, hold until exit of top/bottom 35 %
                     (Novy-Marx-Velikov banding), equal weight in held set daily
  3. ema{3,5,10}     EMA-smoothed cross-sectional pct rank, then unbuffered quintile
  4. band+ema5       banding applied to the EMA(5)-smoothed rank
  5. jt K=5          Jegadeesh-Titman overlap: 5 staggered sub-books, each rebalances
                     every 5th day, book = average of the five

Cost model:
  one-way turnover_t = 0.5 * sum_i |w_t,i - w_drift,i|, w_drift = w_{t-1}*(1+r_i,t)
                       renormalized to gross 1 (drift approximated with daily close
                       returns, no intraday path)
  net_t = gross_t - turnover_t * cost  (cost in {0,2,3,5} bps, charged per the spec's
          one-way-turnover formula) - short borrow 35 bps/yr on half the book, daily.

Execution-lag variant (best arm): weights from close t trade at close t+1
(one extra day shift) — measures how much alpha is same-close artifact.

Finally ru_conformal() exposure control on the best net arm at 3 bps (headline).
Approximation note: lambda control applied to the NET stream assumes trading cost
scales with exposure (turnover ~ position size), which is roughly right.

    uv run python -m compare_lab.cost_haircut_backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from compare_lab.cvar_conformal_backtest import (
    BETA, build_signals, ls_returns, ru_conformal,
)

COSTS_BPS = (0, 2, 3, 5)
BORROW_DAILY = 0.0035 * 0.5 / 252   # 35 bps/yr on the short half of the book
HEADLINE_BPS = 3


# ---------------------------------------------------------------- weight builders
def quintile_weights(sig: pd.DataFrame) -> pd.DataFrame:
    top = sig.ge(sig.quantile(0.8, axis=1), axis=0)
    bot = sig.le(sig.quantile(0.2, axis=1), axis=0)
    w = (0.5 * top.div(top.sum(axis=1), axis=0)
         - 0.5 * bot.div(bot.sum(axis=1), axis=0))
    return w.fillna(0.0)


def band_weights(pct: pd.DataFrame, enter=0.20, exit_=0.35) -> pd.DataFrame:
    """Enter leg at top/bottom `enter` pct-rank, hold until it exits top/bottom
    `exit_`. Equal weight within held set, renormalized daily. NaN rank drops."""
    P = pct.to_numpy()
    L = np.zeros(P.shape, bool)
    S = np.zeros(P.shape, bool)
    lo = np.zeros(P.shape[1], bool)
    sh = np.zeros(P.shape[1], bool)
    with np.errstate(invalid="ignore"):
        for i in range(len(P)):
            p = P[i]
            lo = (p >= 1 - enter) | (lo & (p >= 1 - exit_))
            sh = (p <= enter) | (sh & (p <= exit_))
            L[i], S[i] = lo, sh
    L = pd.DataFrame(L, index=pct.index, columns=pct.columns)
    S = pd.DataFrame(S, index=pct.index, columns=pct.columns)
    w = (0.5 * L.div(L.sum(axis=1), axis=0)
         - 0.5 * S.div(S.sum(axis=1), axis=0))
    return w.fillna(0.0)


def jt_weights(sig: pd.DataFrame, K=5) -> pd.DataFrame:
    """K staggered sub-books; sub-book k re-derives quintile weights every K-th day
    and holds them (un-drifted hold — approximation) until its next rebalance."""
    daily = quintile_weights(sig)
    pos = np.arange(len(sig))
    total = None
    for k in range(K):
        wk = daily.copy()
        wk.iloc[pos % K != k] = np.nan
        wk = wk.ffill().fillna(0.0)
        total = wk if total is None else total + wk
    return total / K


# ---------------------------------------------------------------- engine
def turnover(W: pd.DataFrame, piv: pd.DataFrame) -> pd.Series:
    ret1 = (piv / piv.shift(1) - 1).reindex(columns=W.columns)
    drift = W.shift(1) * (1 + ret1)
    g = drift.abs().sum(axis=1)
    drift = drift.div(g.where(g > 0, 1.0), axis=0).fillna(0.0)
    return 0.5 * (W - drift).abs().sum(axis=1)


def stat_row(r: np.ndarray) -> dict:
    ann, vol = r.mean() * 252, r.std() * np.sqrt(252)
    eq = np.cumprod(1 + r)
    mdd = (1 - eq / np.maximum.accumulate(eq)).max()
    loss = -r
    var = np.quantile(loss, BETA)
    return {"ann": ann, "vol": vol, "sharpe": ann / vol if vol > 0 else np.nan,
            "mdd": mdd, "cvar": loss[loss >= var].mean()}


def main() -> int:
    piv, sigs = build_signals()
    combo = sigs["combo"]
    fwd1 = piv.shift(-1) / piv - 1
    idx = ls_returns(piv, combo).index          # canonical backtest dates
    pct = combo.rank(axis=1, pct=True)

    arms: dict[str, pd.DataFrame] = {"unbuffered": quintile_weights(combo)}
    sm = {s: pct.ewm(span=s, min_periods=1).mean().where(combo.notna())
          for s in (3, 5, 10)}
    arms["band 20/35"] = band_weights(pct)
    for s, m in sm.items():
        arms[f"ema{s}"] = quintile_weights(m)
    arms["band+ema5"] = band_weights(sm[5].rank(axis=1, pct=True))
    arms["jt K=5"] = jt_weights(combo)

    # sanity: zero-cost unbuffered gross reproduces the known Sharpe ~0.72
    g0 = (arms["unbuffered"] * fwd1).sum(axis=1).loc[idx].to_numpy()
    sh0 = stat_row(g0)["sharpe"]
    assert abs(sh0 - 0.72) < 0.05, f"unbuffered gross Sharpe {sh0:.3f} != ~0.72"

    print(f"backtest {idx[0].date()} .. {idx[-1].date()}  ({len(idx)} days)  "
          f"[sanity: unbuffered gross Sharpe {sh0:.3f} ~ 0.72 OK]")
    print(f"short borrow 35 bps/yr on half book = {BORROW_DAILY*252:.3%}/yr; "
          "costs charged on one-way turnover; drifted-weight approximation.\n")

    hdr = (f"{'arm':<12}{'cost':>5}{'to/day':>8}{'ann.ret':>9}{'vol':>8}"
           f"{'Sharpe':>8}{'maxDD':>8}{'CVaR.85':>9}")
    print(hdr)
    print("-" * len(hdr))
    results = {}
    for name, W in arms.items():
        g = (W * fwd1).sum(axis=1).loc[idx]
        to = turnover(W, piv).loc[idx]
        for c in COSTS_BPS:
            net = (g - to * c * 1e-4 - BORROW_DAILY).to_numpy()
            st = stat_row(net)
            results[(name, c)] = {"to": to.mean(), **st}
            print(f"{name:<12}{c:>4}b{to.mean():>8.3f}{st['ann']:>+9.2%}"
                  f"{st['vol']:>8.2%}{st['sharpe']:>8.2f}{st['mdd']:>8.2%}"
                  f"{st['cvar']:>+9.3%}")
        print()

    # best net arm at the headline cost
    best = max(arms, key=lambda a: results[(a, HEADLINE_BPS)]["sharpe"])
    W = arms[best]
    g = (W * fwd1).sum(axis=1).loc[idx]
    to = turnover(W, piv).loc[idx]
    net = (g - to * HEADLINE_BPS * 1e-4 - BORROW_DAILY).to_numpy()

    # execution lag: trade one close later
    g_lag = (W.shift(1) * fwd1).sum(axis=1).loc[idx]
    net_lag = (g_lag - to * HEADLINE_BPS * 1e-4 - BORROW_DAILY).to_numpy()
    st, st_lag = stat_row(net), stat_row(net_lag)
    print(f"best net arm at {HEADLINE_BPS} bps: {best}  "
          f"(Sharpe {st['sharpe']:.2f}, to/day {to.mean():.3f})")
    print(f"execution lag t+1 close: Sharpe {st_lag['sharpe']:.2f} "
          f"(delta {st_lag['sharpe']-st['sharpe']:+.2f}), "
          f"ann {st_lag['ann']:+.2%} vs {st['ann']:+.2%}")

    # RU-conformal control on the best net stream (headline)
    lam, _ = ru_conformal(-net)
    ctrl = stat_row(lam * net)
    print(f"\nHEADLINE — RU-conformal on {best} net@{HEADLINE_BPS}bps: "
          f"Sharpe {ctrl['sharpe']:.2f}, maxDD {ctrl['mdd']:.2%}, "
          f"realized CVaR.85 {ctrl['cvar']:+.3%}, mean lambda {lam.mean():.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
