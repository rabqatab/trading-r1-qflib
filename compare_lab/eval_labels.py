"""Label-fidelity eval: how well does a model reproduce OUR label (the perf proxy)?

The chain: real performance -> [our forward Sharpe-like label = proxy] -> model.
This script measures axis ① **label-fidelity** (did the model learn the proxy),
regime-independent, from cached replies only (no inference). It reads axis ②
**proxy-validity** (does the proxy make money) from the sibling backtest CSV, and
places each model in the 2x2.

    uv run python -m compare_lab.eval_labels .cache_sftv1 .cache_v1reggrpo_full
    # (bare names resolve under compare_lab/; sibling backtest = .cache_X -> output_X)
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401  (path setup)
from alpha_lab.core import load_context
from compare_lab.cache_io import read_decisions, resolve
from compare_lab.grpo.rewards import DECISION_MATRIX, INVALID_DECISION_PENALTY
from compare_lab.labeling import CLASSES, make_labels, make_signal

IDX = {c: i for i, c in enumerate(CLASSES)}           # STRONG_SELL=0 .. STRONG_BUY=4


def _wcorr(x, y, w):
    """Weighted Pearson correlation (on ranks -> weighted Spearman)."""
    mx, my = np.average(x, weights=w), np.average(y, weights=w)
    cov = np.average((x - mx) * (y - my), weights=w)
    vx, vy = np.average((x - mx) ** 2, weights=w), np.average((y - my) ** 2, weights=w)
    return float(cov / (vx * vy) ** 0.5) if vx > 0 and vy > 0 else float("nan")


def _attach_truth(df: pd.DataFrame) -> pd.DataFrame:
    """Add ground-truth label + raw proxy signal at each (ticker, date)."""
    ctx = load_context(universe=sorted(df["ticker"].unique()))
    lab, sig = {}, {}
    for t in df["ticker"].unique():
        px = ctx.adj_close[t].dropna()
        lab[t] = make_labels(px, forward=True)
        sig[t] = make_signal(px, forward=True)
    df = df.copy()
    df["label"] = [lab[t].get(d) for t, d in zip(df.ticker, df.date)]
    df["signal"] = [sig[t].get(d) for t, d in zip(df.ticker, df.date)]
    return df


def _backtest(cache_dir: Path) -> dict | None:
    out = cache_dir.parent / cache_dir.name.replace(".cache_", "output_")
    csvf = out / "comparison.csv"
    if not csvf.exists():
        return None
    for r in csv.DictReader(open(csvf)):
        if r["provider"] == "llm_prompt_only":
            return {k: float(r[k]) for k in ("CR", "SR", "HR", "MDD")}
    return None


def evaluate(cache_dir) -> dict:
    cache_dir = resolve(cache_dir)
    df = _attach_truth(read_decisions(cache_dir))
    n = len(df)
    notag = float((df.pred == "NO_TAG").mean())
    dist = (df.pred.value_counts(normalize=True)).to_dict()

    # ① fidelity, scored by OUR OWN asymmetric decision matrix (rewards.py §5.2) —
    # the same scoring the model is trained against. Every row with a defined label
    # counts; NO_TAG / echo gets the invalid penalty (it's a real failure).
    d = df[df.label.notna()].copy()
    def _mr(pred, true):
        return INVALID_DECISION_PENALTY if pred == "NO_TAG" else DECISION_MATRIX[pred][true]
    matrix_reward = float(d.apply(lambda r: _mr(r.pred, r.label), axis=1).mean())
    # reference points on the SAME label set: perfect=+1.0; best single constant call;
    # always-HOLD; so matrix_reward is readable on a scale, not in a vacuum.
    labels = d.label.tolist()
    best_const = max(sum(DECISION_MATRIX[c][t] for t in labels) / len(labels)
                     for c in CLASSES)
    hold_const = sum(DECISION_MATRIX["HOLD"][t] for t in labels) / len(labels)

    # ①' the LABEL-GENERATION score view: rank-IC of predicted class vs the actual
    # forward Sharpe-like signal (make_signal). This IS the score the label is cut
    # from — Spearman is invariant to the quantile cut, so it scores against the
    # raw signal directly. tail_ic = same, but weighted so the extremes (which the
    # asymmetric quantiles {.03,.15,.53,.85} carve thin) count more.
    g = df[(df.pred != "NO_TAG") & df.label.notna() & df.signal.notna()].copy()
    g["pi"], g["li"] = g.pred.map(IDX), g.label.map(IDX)
    acc = float((g.pi == g.li).mean())
    omae = float((g.pi - g.li).abs().mean())            # ordinal distance
    rank_corr = float(g["pi"].corr(g["signal"], method="spearman"))   # pooled IC
    # per-ticker IC (labels are ranked per-ticker-over-time), then averaged
    per_t = [x.pi.corr(x.signal, method="spearman")
             for _, x in g.groupby("ticker") if len(x) > 30]
    per_t = [c for c in per_t if pd.notna(c)]
    ic_perticker = float(sum(per_t) / len(per_t)) if per_t else float("nan")
    # tail-weighted IC: weight = 2*|rank(signal)-0.5| (∈[0,1]), extremes ~1, middle ~0
    rs = g.signal.rank(pct=True)
    w = (2 * (rs - 0.5)).abs()
    rp = g.pi.rank(pct=True)
    tail_ic = _wcorr(rp.to_numpy(), rs.to_numpy(), w.to_numpy())
    by_class = g.groupby("pred")["signal"].mean().reindex(CLASSES)
    # monotonicity only over buckets the model actually uses (≥20 preds) — tiny
    # end buckets are noise and falsely break a strict is_monotonic check.
    used = g.pred.value_counts()
    solid = [c for c in CLASSES if used.get(c, 0) >= 20]
    monotone = bool(by_class.reindex(solid).is_monotonic_increasing)
    conf = pd.crosstab(g.label, g.pred).reindex(index=CLASSES, columns=CLASSES,
                                                fill_value=0)
    return {"n": n, "n_scored": len(g), "notag": notag, "dist": dist,
            "ic": rank_corr, "ic_perticker": ic_perticker, "tail_ic": tail_ic,
            "matrix_reward": matrix_reward, "best_const": best_const,
            "hold_const": hold_const,
            "acc": acc, "ordinal_mae": omae, "rank_corr": rank_corr,
            "monotone": monotone, "by_class_signal": by_class, "confusion": conf,
            "backtest": _backtest(cache_dir)}


def _quadrant(r: dict) -> str:
    # learned the proxy = positive rank-IC vs the label-generation signal
    # (does the predicted class rank with the actual forward Sharpe-like score).
    fidelity_ok = r["ic"] is not None and r["ic"] > 0.05
    bt = r["backtest"]
    bt_ok = bt is not None and bt["SR"] > 0.5
    f = "learned proxy" if fidelity_ok else "did NOT learn proxy"
    b = "?" if bt is None else ("backtest OK" if bt_ok else "backtest weak")
    diag = {(True, True): "✅ good proxy, well learned",
            (True, False): "⚠️ learned it, but PROXY/regime is the problem -> fix label",
            (False, True): "🍀 lucky / proxy too loose",
            (False, False): "❌ MODEL/training failure -> fix training"}[(fidelity_ok, bt_ok)]
    return f"[{f} | {b}] {diag}"


def _report(name: str, r: dict) -> None:
    print(f"\n{'='*70}\n{name}   (n={r['n']}, scored={r['n_scored']}, "
          f"NO_TAG={r['notag']*100:.1f}%)")
    print("  decision dist:",
          {k: f"{v*100:.0f}%" for k, v in sorted(r["dist"].items(),
                                                 key=lambda x: -x[1])})
    print(f"  ① fidelity vs LABEL signal (rank-IC):  pooled={r['ic']:+.3f}"
          f"  per-ticker={r['ic_perticker']:+.3f}  tail-weighted={r['tail_ic']:+.3f}"
          f"   (0=chance, 1=perfect)")
    print(f"     side views: acc={r['acc']*100:.1f}%  ordinal_MAE={r['ordinal_mae']:.2f}"
          f"  | training-reward(§5.2 matrix)={r['matrix_reward']:+.3f}"
          f" (best-const {r['best_const']:+.3f})  monotone={r['monotone']}")
    print("     mean realized signal by predicted class (want ↑ SS→SB):")
    print("    ", {c: (None if pd.isna(v) else round(v, 3))
                   for c, v in r["by_class_signal"].items()})
    bt = r["backtest"]
    print("  ② backtest :", "none" if bt is None
          else {k: round(v, 3) for k, v in bt.items()})
    print("  2x2 ->", _quadrant(r))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("caches", nargs="+", help=".cache_* dir(s) (bare or path)")
    args = ap.parse_args()
    for c in args.caches:
        p = resolve(c)
        if not p.exists():
            print(f"skip (not found): {c}", file=sys.stderr)
            continue
        _report(p.name, evaluate(p))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
