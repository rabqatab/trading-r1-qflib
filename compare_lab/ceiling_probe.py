"""Why is the IC ceiling ~0.21 and model-invariant? — an information-theoretic probe.

Three experiments, same features / same 2024-train / same 1,000-point 2025-H1 OOS as
`gbm_ceiling.py`, to test whether the ceiling is the *input* (mutual information I(X;Y),
data-processing inequality) rather than any model's capacity:

  E1  MODEL-INVARIANCE : run 6 model families (linear, kNN, RF, GBM, MLP, + a 1-feature
      momentum baseline). If they all cap ~0.21, the ceiling is I(X;Y), not the model.
  E2  RESIDUAL AUDIT   : take GBM's OOS residual (signal − pred); train a 2nd model to
      predict it. residual-IC ≈ 0 ⇒ what GBM misses is irreducible noise, not missed signal.
  E3  BAYES / LABEL    : GBM in-sample (train) IC and R². If even memorizing the training set
      caps ~0.22 IC / low R², the label is NOT a deterministic function of the features —
      the cap is label noise, not under-fitting.

    uv run python -m compare_lab.ceiling_probe
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.gbm_ceiling import (_EVAL, _PRICES, _TRAIN_HI, _TRAIN_LO,
                                      _indicators, _spearman)
from compare_lab.labeling import make_signal
from compare_lab.snapshot import _INDICATORS


def _matrices():
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
    tr = []
    for t, F in feats.items():
        s = sig[t]
        for d in F.index[(F.index >= _TRAIN_LO) & (F.index <= _TRAIN_HI)]:
            if d in s.index and pd.notna(s.loc[d]) and F.loc[d].notna().all():
                tr.append((*F.loc[d].values, s.loc[d]))
    tr = np.array(tr, float)
    Xtr, ytr = tr[:, :-1], tr[:, -1]

    ev = [json.loads(l) for l in _EVAL.open()]
    Xte, yte, roc = [], [], []
    roc_i = _INDICATORS.index("close_10_roc")
    for e in ev:
        F = feats.get(e["ticker"]); d = pd.Timestamp(e["as_of"])
        if F is None or d not in F.index or not F.loc[d].notna().all():
            continue
        v = F.loc[d].values
        Xte.append(v); yte.append(e["signal"]); roc.append(v[roc_i])
    return Xtr, ytr, np.array(Xte, float), np.array(yte, float), np.array(roc, float)


def main() -> int:
    Xtr, ytr, Xte, yte, roc = _matrices()
    print(f"train rows {len(ytr)} | test points {len(yte)}\n")
    sc = StandardScaler().fit(Xtr)
    Ztr, Zte = sc.transform(Xtr), sc.transform(Xte)

    # ---- E1: model-invariance ----
    gbm = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=3,
                                        l2_regularization=1.0, random_state=0).fit(Xtr, ytr)
    models = {
        "momentum (close_10_roc, 1 feat)": ("naive", roc),
        "Ridge (linear)": Ridge(alpha=10.0).fit(Ztr, ytr),
        "kNN (k=100)": KNeighborsRegressor(n_neighbors=100).fit(Ztr, ytr),
        "RandomForest": RandomForestRegressor(n_estimators=300, max_depth=6, n_jobs=-1,
                                              random_state=0).fit(Xtr, ytr),
        "GBM (HistGBDT)": gbm,
        "MLP (64,32)": MLPRegressor(hidden_layer_sizes=(64, 32), alpha=1e-2, max_iter=300,
                                    random_state=0).fit(Ztr, ytr),
    }
    print("=== E1  MODEL-INVARIANCE (IC on the same 1,000 OOS points) ===")
    for name, m in models.items():
        pred = m[1] if isinstance(m, tuple) else (
            m.predict(Zte) if name.startswith(("Ridge", "kNN", "MLP")) else m.predict(Xte))
        print(f"  {name:<34} IC {_spearman(pred, yte):+.3f}")

    # ---- E2: residual audit ----
    res_tr = ytr - gbm.predict(Xtr)
    res_te = yte - gbm.predict(Xte)
    r2 = HistGradientBoostingRegressor(max_iter=400, learning_rate=0.05, max_depth=3,
                                       l2_regularization=1.0, random_state=1).fit(Xtr, res_tr)
    print("\n=== E2  RESIDUAL AUDIT (can any model predict what GBM misses?) ===")
    print(f"  IC(2nd-model, GBM OOS residual) = {_spearman(r2.predict(Xte), res_te):+.3f}"
          "   (≈0 ⇒ residual is irreducible noise, not missed signal)")

    # ---- E3: Bayes / label-noise ----
    ptr = gbm.predict(Xtr)
    ss_res = float(np.sum((ytr - ptr) ** 2)); ss_tot = float(np.sum((ytr - ytr.mean()) ** 2))
    print("\n=== E3  BAYES / LABEL-NOISE (is the label a function of the features?) ===")
    print(f"  GBM in-sample (train) IC  = {_spearman(ptr, ytr):+.3f}  (memorizing 36k rows still caps here)")
    print(f"  GBM in-sample R²          = {1 - ss_res / ss_tot:+.3f}  (low ⇒ features explain little variance)")
    print(f"  OOS IC                    = {_spearman(gbm.predict(Xte), yte):+.3f}")

    # ---- E6: smoothing inflation — predict RAW forward return vs the smoothed proxy ----
    # make_signal is a 3-day-EMA, overlapping-horizon, vol-adjusted proxy; raw next-week
    # return has none of that autocorrelation. If IC collapses on the raw target, the
    # "predictability" is largely a smoothing artifact, not tradeable return skill.
    tickers = sorted(pd.read_parquet(_PRICES, columns=["ticker"])["ticker"].unique())
    ctx = load_context(universe=tickers, parquet_path=_PRICES,
                       data_start=pd.Timestamp("2015-01-01"), data_end=pd.Timestamp("2026-05-08"))
    ev = [json.loads(l) for l in _EVAL.open()]
    raw, keep = [], []
    for i, e in enumerate(ev):
        px = ctx.adj_close.get(e["ticker"])
        if px is None:
            continue
        px = px.dropna(); d = pd.Timestamp(e["as_of"])
        if d not in px.index:
            continue
        pos = px.index.get_loc(d)
        if pos + 7 >= len(px):
            continue
        raw.append(px.iloc[pos + 7] / px.iloc[pos] - 1.0); keep.append(i)
    raw = np.array(raw, float)
    # align the two best predictors + the label to the kept rows
    mom_k = roc[keep]; sig_k = yte[keep]; gbm_k = gbm.predict(Xte)[keep]
    print("\n=== E6  SMOOTHING INFLATION (raw 7-day forward return vs the make_signal proxy) ===")
    print(f"  IC(momentum, make_signal proxy) = {_spearman(mom_k, sig_k):+.3f}")
    print(f"  IC(momentum, RAW 7d return)     = {_spearman(mom_k, raw):+.3f}")
    print(f"  IC(GBM,      RAW 7d return)     = {_spearman(gbm_k, raw):+.3f}")
    print("  (big drop on RAW ⇒ the proxy's predictability is inflated by EMA smoothing)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
