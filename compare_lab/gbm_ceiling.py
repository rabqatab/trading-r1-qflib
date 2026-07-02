"""Same-universe GBM ceiling for the top-150 2025-H1 OOS — contextualizes the LLM's IC.

VALID use of the GBM (unlike the earlier proxy misuse): it bounds the *extractable
signal from the same technical features*, on the *same* eval points, so "LLM IC vs GBM
IC" says how much of the achievable signal the templated-SFT LLM captured.

Leak-safety: features are the 16 backward-looking technical indicators the LLM snapshot
shows (strictly causal at as_of); GBM trains on **2024** (ticker, date -> forward signal),
predicts the **exact 1000 2025-H1 eval points** the LLM was scored on; forward signal is
the target only, never a feature (same as the LLM's label).

    uv run python -m compare_lab.gbm_ceiling
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from stockstats import StockDataFrame

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.labeling import make_signal
from compare_lab.snapshot import _INDICATORS  # the same 16 the LLM sees

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_EVAL = Path("compare_lab/eval150/eval_mm.jsonl")
_TRAIN_LO, _TRAIN_HI = pd.Timestamp("2024-01-01"), pd.Timestamp("2024-12-31")


def _spearman(x, y) -> float:
    x, y = pd.Series(x), pd.Series(y)
    rx, ry = x.rank(), y.rank()
    return float(np.corrcoef(rx, ry)[0, 1]) if x.nunique() > 1 and y.nunique() > 1 else float("nan")


def _indicators(ctx, ticker: str) -> pd.DataFrame:
    """16 indicators at every date (backward-looking -> full-series == causal-window value)."""
    df = pd.DataFrame({
        "open": ctx.open[ticker], "high": ctx.high[ticker], "low": ctx.low[ticker],
        "close": ctx.adj_close[ticker], "volume": ctx.volume[ticker],
    }).dropna()
    sdf = StockDataFrame.retype(df.copy())
    for ind in _INDICATORS:
        _ = sdf[ind]
    return sdf[_INDICATORS]


def main() -> int:
    tickers = sorted(pd.read_parquet(_PRICES, columns=["ticker"])["ticker"].unique())
    ctx = load_context(universe=tickers, parquet_path=_PRICES,
                       data_start=pd.Timestamp("2015-01-01"), data_end=pd.Timestamp("2026-05-08"))

    feats, sig = {}, {}
    for t in tickers:
        try:
            feats[t] = _indicators(ctx, t)
            sig[t] = make_signal(ctx.adj_close[t].dropna(), forward=True)
        except Exception:
            continue

    # --- training rows: all 2024 (ticker, date) with defined features + signal ---
    rows = []
    for t, F in feats.items():
        s = sig[t]
        idx = F.index[(F.index >= _TRAIN_LO) & (F.index <= _TRAIN_HI)]
        for d in idx:
            if d in s.index and pd.notna(s.loc[d]) and F.loc[d].notna().all():
                rows.append((*F.loc[d].values, s.loc[d]))
    train = np.array(rows, float)
    Xtr, ytr = train[:, :-1], train[:, -1]

    # --- test rows: the EXACT 1000 eval points the LLM was scored on ---
    ev = [json.loads(l) for l in _EVAL.open()]
    Xte, yte, kept = [], [], 0
    for e in ev:
        t, d = e["ticker"], pd.Timestamp(e["as_of"])
        F = feats.get(t)
        if F is None or d not in F.index or not F.loc[d].notna().all():
            continue
        Xte.append(F.loc[d].values); yte.append(e["signal"]); kept += 1
    Xte, yte = np.array(Xte, float), np.array(yte, float)

    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05,
                                        max_depth=3, l2_regularization=1.0,
                                        random_state=0)
    gbm.fit(Xtr, ytr)
    pred = gbm.predict(Xte)

    ic_cont = _spearman(pred, yte)
    # apples-to-apples: bucket GBM pred into 5 classes (same coarseness as the LLM output)
    q = pd.qcut(pred, 5, labels=False, duplicates="drop")
    ic_5cls = _spearman(q, yte)

    print(f"[gbm-ceiling] train rows (2024): {len(ytr)} | test points (2025-H1): {kept}")
    print(f"[gbm-ceiling] GBM continuous IC (input ceiling) : {ic_cont:+.3f}")
    print(f"[gbm-ceiling] GBM 5-class-bucketed IC (LLM-fair): {ic_5cls:+.3f}")
    print(f"[gbm-ceiling] LLM template-SFT 3047 IC          : +0.163  (for reference)")
    gap = ic_cont - 0.163
    print(f"[gbm-ceiling] extraction gap (GBM_cont - LLM)   : {gap:+.3f} "
          f"-> LLM captured {100*0.163/ic_cont:.0f}% of the continuous ceiling"
          if np.isfinite(ic_cont) and ic_cont > 0 else "")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
