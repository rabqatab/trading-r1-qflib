"""Target-redesign sweep: cross-sectional RANK targets + label-horizon sweep vs the
regression GBM baseline (gbm_ceiling.py) on the top-150 universe.

Levers under test (2nd lit sweep — triple-barrier demoted, no replicated OOS evidence):
  * rank objectives (LambdaRankIC, arXiv:2605.00501): here the poor-man's version —
    regression on the cross-sectionally percentile-RANKED forward return. NOTE: neither
    xgboost nor lightgbm is installed in this env, so the native pairwise/lambdarank
    objective (arm c) is UNAVAILABLE; sklearn HistGradientBoosting has no rank loss.
  * Label-Horizon-Paradox (arXiv:2602.03395): train label horizon h in {3,5,7,10,15},
    always EVALUATED at 7d.

Eval protocol (identical for every arm): daily cross-sectional Spearman rank-IC of the
prediction vs the RAW 7-trading-day forward return, walk-forward — train on an expanding
window ending T (label-complete: last train date + h trading days < eval start), evaluate
the next 6 months, roll 2021H1 .. 2026H1. The 2025-H1 window is the canonical one
(momentum baseline +0.064 raw, regression GBM ~+0.042 raw).

Honest framing: the raw info ceiling is ~0.06 — the question is whether rank targets
extract MORE of it than regression, not whether they break it.

    uv run python -m compare_lab.rank_target_sweep
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from stockstats import StockDataFrame

import compare_lab  # noqa: F401
from compare_lab.snapshot import _INDICATORS  # the same 16 features as gbm_ceiling

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_CACHE = Path(os.environ.get(
    "RANK_SWEEP_CACHE",
    "/tmp/claude-1000/-home-alphabridge-Study-tradingR1-qflib/"
    "f4ef47e5-faf4-4073-9339-e7be58e5a29b/scratchpad/rank_sweep_features.parquet"))

HORIZONS = (3, 5, 7, 10, 15)   # training label horizons (trading days)
EVAL_H = 7                     # evaluation horizon, always 7d raw forward return
MIN_NAMES = 50                 # min cross-section size for a daily IC


def _daily_ic(pred: pd.DataFrame, fwd: pd.DataFrame) -> pd.Series:
    """Daily cross-sectional Spearman rank-IC (rows = dates)."""
    ics = pred.rank(axis=1).corrwith(fwd.rank(axis=1), axis=1)
    n = (pred.notna() & fwd.notna()).sum(axis=1)
    return ics[n >= MIN_NAMES]


def build_panel():
    """Long feature panel (date,ticker) x 16 indicators + raw/ranked fwd labels."""
    px = pd.read_parquet(_PRICES)
    px["date"] = pd.to_datetime(px["date"])
    piv = px.pivot(index="date", columns="ticker", values="Close").sort_index()

    # --- raw h-day forward returns, close(t) -> close(t+h trading days) ---
    fwd = {h: piv.shift(-h) / piv - 1 for h in HORIZONS}

    # --- assert: label construction is leak-free — fwd_h at t must equal the ---
    # --- compounded DAILY returns of days t+1 .. t+h only (nothing at <= t). ---
    ret1 = piv.pct_change()
    for h in (3, EVAL_H, 15):
        col = piv.columns[0]
        i = 1500
        manual = float(np.prod(1.0 + ret1[col].iloc[i + 1:i + h + 1].values) - 1.0)
        assert np.isclose(fwd[h][col].iloc[i], manual, atol=1e-12), \
            f"label leak check failed at h={h}: {fwd[h][col].iloc[i]} vs {manual}"

    # cross-sectional percentile rank per DAY (uses only same-day cross-section)
    rank = {h: fwd[h].rank(axis=1, pct=True) for h in HORIZONS}

    if _CACHE.exists():
        feats = pd.read_parquet(_CACHE)
    else:
        parts = []
        for t, g in px.groupby("ticker"):
            df = (g.set_index("date")[["Open", "High", "Low", "Close", "Volume"]]
                   .rename(columns=str.lower).sort_index().dropna())
            if len(df) < 250:
                continue
            sdf = StockDataFrame.retype(df.copy())
            for ind in _INDICATORS:
                _ = sdf[ind]
            F = sdf[_INDICATORS].copy()
            F["ticker"] = t
            parts.append(F)
        feats = pd.concat(parts).reset_index().rename(columns={"index": "date"})
        _CACHE.parent.mkdir(parents=True, exist_ok=True)
        feats.to_parquet(_CACHE)

    feats = feats.set_index(["date", "ticker"]).sort_index()
    for h in HORIZONS:
        feats[f"y_raw_{h}"] = fwd[h].stack()
        feats[f"y_rank_{h}"] = rank[h].stack()
    return feats, piv, fwd[EVAL_H]


def eval_windows(cal: pd.DatetimeIndex):
    """Half-year OOS windows 2021H1..2026H1 (last one partial to data end)."""
    out = []
    for y in range(2021, 2027):
        for lo, hi in ((f"{y}-01-01", f"{y}-06-30"), (f"{y}-07-01", f"{y}-12-31")):
            lo, hi = pd.Timestamp(lo), pd.Timestamp(hi)
            if lo > cal[-1]:
                continue
            out.append((lo, min(hi, cal[-1])))
    return out


def main() -> int:
    feats, piv, fwd7 = build_panel()
    cal = piv.index
    X_cols = list(_INDICATORS)
    windows = eval_windows(cal)

    arms = [("reg_raw_h7", "y_raw_7")] + \
           [(f"rank_h{h}", f"y_rank_{h}") for h in HORIZONS]
    mom10 = piv / piv.shift(10) - 1  # model-free momentum baseline

    results = {}   # (arm, window_label) -> mean daily rank-IC
    for lo, hi in windows:
        wlab = f"{lo.year}H{1 if lo.month == 1 else 2}"
        eval_days = cal[(cal >= lo) & (cal <= hi)]
        Xte = feats.loc[(slice(eval_days[0], eval_days[-1]), slice(None)), X_cols]

        results[("mom10", wlab)] = _daily_ic(mom10.loc[eval_days],
                                             fwd7.loc[eval_days]).mean()
        for arm, ycol in arms:
            h = int(ycol.rsplit("_", 1)[1])
            # label-complete training cutoff: last train date + h trading days < lo
            cut = cal[cal.searchsorted(lo) - h - 1]
            assert cal[cal.searchsorted(cut) + h] < lo, "train label overlaps eval"
            tr = feats.loc[(slice(None, cut), slice(None)), X_cols + [ycol]].dropna(
                subset=[ycol])
            gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                                max_depth=3, l2_regularization=1.0,
                                                random_state=0)
            gbm.fit(tr[X_cols], tr[ycol])
            pred = pd.Series(gbm.predict(Xte), index=Xte.index).unstack("ticker")
            ic = _daily_ic(pred, fwd7.loc[pred.index]).mean()
            results[(arm, wlab)] = ic
            print(f"  [{wlab}] {arm:<11} train_rows={len(tr):>7} "
                  f"cutoff={cut.date()} rank-IC={ic:+.4f}", flush=True)

    tab = pd.Series(results).unstack(0)
    order = ["mom10", "reg_raw_h7"] + [f"rank_h{h}" for h in HORIZONS]
    tab = tab[order]
    tab.loc["MEAN"] = tab.mean()

    pd.set_option("display.width", 200)
    print("\n=== daily cross-sectional rank-IC vs RAW 7d fwd return, walk-forward ===")
    print(tab.round(4).to_string())
    print("\ncanonical 2025H1 row above is directly comparable to: momentum +0.064, "
          "regression GBM +0.042 (gbm_ceiling protocol).")
    print("arm (c) native pairwise/lambdarank: SKIPPED — xgboost/lightgbm not "
          "installed; sklearn has no rank objective.")
    out = _CACHE.parent / "rank_sweep_results.csv"
    tab.to_csv(out)
    print(f"saved: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
