"""Compare position-sizing schemes on the SAME model decisions.

Holds the signal fixed (a model's cached 5-class calls) and varies only how those
calls become target weights, so the table isolates the sizing choice. Schemes:

  current   - deployed rule: BUY/STRONG_BUY -> equal 1/8, top-8 by strength, rest cash
  fixed     - tier value {SS:-2..SB:+2}, normalised to gross 1 (long-short)
  inv_vol   - tier value / trailing vol, gross 1 (equal risk contribution)
  mvo       - Sigma_shrunk^-1 . mu(tier), gross 1 (mean-variance, shrunk covariance)
  rank      - cross-sectional: long top-k / short bottom-k, dollar-neutral

NOT included: 'expected conviction' (needs per-class probabilities/logits, which the
text cache does not store -> would require re-inference with logprobs).

Engine: vectorised weight x next-bar return (weekly rebal, 1-day exec lag), one path
for all schemes; no commissions. NB qf-lib *does* support long-short (Exposure.SHORT),
but it sizes via a position-sizer + direction, not a free per-name weight vector, so
it is not a drop-in for these continuous-weight schemes; our backtest.py bridge is
also LONG/OUT only today. Use this for the relative sizing comparison; use the qf-lib
engine (backtest.py) for the commission-accurate headline portfolio number.

    uv run python -m compare_lab.sizing_compare sftv1 grpo
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.cache_io import read_decisions, resolve
from compare_lab.metrics import all_metrics

OOS_START = pd.Timestamp("2024-01-01")
TIER = {"STRONG_SELL": -2, "SELL": -1, "HOLD": 0, "BUY": 1, "STRONG_BUY": 2, "NO_TAG": 0}
VOL_WIN, RANK_K, MVO_SHRINK, MVO_RIDGE = 60, 3, 0.3, 1e-4


def _gross1(w: pd.Series) -> pd.Series:
    g = w.abs().sum()
    return w / g if g > 0 else w


def _w_market(s, **_):                        # benchmark: equal-weight all names, always long
    return pd.Series(1.0 / len(s), index=s.index)


def _w_current(s, **_):                       # long-only, 1/8 each, top-8, rest cash
    held = s[s >= 1].sort_values(ascending=False).head(8)
    w = pd.Series(0.0, index=s.index)
    w[held.index] = 1.0 / 8
    return w


def _w_fixed(s, **_):
    return _gross1(s.copy())


def _w_fixed_lo(s, **_):                      # long-only: drop the short leg
    return _gross1(s.clip(lower=0))


def _w_inv_vol(s, vol=None, **_):
    return _gross1(s / vol.reindex(s.index))


def _w_inv_vol_lo(s, vol=None, **_):
    return _gross1(s.clip(lower=0) / vol.reindex(s.index))


def _w_rank(s, k=RANK_K, **_):
    s = s.dropna()
    if len(s) < 2 * k:
        return pd.Series(0.0, index=s.index)
    order = s.sort_values()
    w = pd.Series(0.0, index=s.index)
    w[order.index[-k:]] = 1.0           # long top-k
    w[order.index[:k]] = -1.0           # short bottom-k
    return _gross1(w)


def _w_mvo(s, cov=None, **_):
    cols = [c for c in s.index if c in cov.index and np.isfinite(s[c])]
    if len(cols) < 2:
        return pd.Series(0.0, index=s.index)
    mu = s[cols].to_numpy(float)
    C = cov.loc[cols, cols].to_numpy(float)
    C = (1 - MVO_SHRINK) * C + MVO_SHRINK * np.diag(np.diag(C))   # shrink to diagonal
    C += MVO_RIDGE * np.eye(len(cols))                            # ridge for stability
    w = np.linalg.solve(C, mu)
    return _gross1(pd.Series(w, index=cols)).reindex(s.index).fillna(0.0)


SCHEMES = {"market_EW": _w_market, "current": _w_current,
           "fixed": _w_fixed, "fixed_LO": _w_fixed_lo,
           "inv_vol": _w_inv_vol, "inv_vol_LO": _w_inv_vol_lo,
           "mvo": _w_mvo, "rank": _w_rank}


def _weight_matrix(S: pd.DataFrame, R: pd.DataFrame, scheme) -> pd.DataFrame:
    W = pd.DataFrame(0.0, index=S.index, columns=S.columns)
    for d in S.index:
        hist = R.loc[R.index < d].tail(VOL_WIN)             # causal: strictly before d
        if len(hist) < 20:
            continue
        w = scheme(S.loc[d], vol=hist.std(), cov=hist.cov())
        W.loc[d] = w.reindex(S.columns).fillna(0.0)
    return W


def _returns(W: pd.DataFrame, R: pd.DataFrame, start=OOS_START) -> pd.Series:
    daily = W.reindex(R.index).ffill().shift(1)             # hold + next-bar execution
    r = (daily * R).sum(axis=1)
    return r[r.index >= start]


def run(cache) -> pd.DataFrame:
    dec = read_decisions(cache)
    dec = dec.assign(score=dec.pred.map(TIER))
    S = dec.pivot_table(index="date", columns="ticker", values="score", aggfunc="last")
    ctx = load_context(universe=sorted(dec.ticker.unique()))
    R = ctx.adj_close[S.columns].pct_change()
    start = S.index.min()                       # backtest from the cache's own first rebal
    rows = {}
    for name, fn in SCHEMES.items():
        r = _returns(_weight_matrix(S, R, fn), R, start=start)
        m = all_metrics(r)
        years = len(r) / 252.0
        cagr = (1 + m["CR"]) ** (1 / years) - 1 if years > 0 and m["CR"] > -1 else float("nan")
        rows[name] = {"CAGR%": cagr * 100, "SR": m["SR"], "MDD%": m["MDD"] * 100,
                      "CR%": m["CR"] * 100}
    return pd.DataFrame(rows).T


def main() -> int:
    for c in sys.argv[1:]:
        df = run(resolve(c))
        print(f"\n=== {resolve(c).name}  (OOS {OOS_START.date()}+, weekly, sizing comparison) ===")
        print(df.round(2).to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
