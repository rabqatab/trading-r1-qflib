"""Rich diagnostics for the 3-way comparison (for the results memo, task B).

Reuses the exact run_comparison pipeline but, beyond the 4 headline metrics,
also reports: positive-day counts (to sanity-check HR), pairwise return
correlation, the LLM decision distribution from the disk cache, and dumps the
aligned OOS daily-return + equity series to CSV for plotting.

Run with the LLM weights already cached (instant):
    uv run python -m compare_lab.analyze_results --out compare_lab/output
"""
from __future__ import annotations

import argparse
import glob
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.backtest import run_backtest
from compare_lab.config import OOS_START, OOS_END, REBAL_FREQ, UNIVERSE
from compare_lab.llm_client import VLLMClient
from compare_lab.metrics import all_metrics
from compare_lab.providers.equal_weight import EqualWeightProvider
from compare_lab.providers.llm import LLMProvider
from compare_lab.providers.momentum import MomentumProvider
from compare_lab.run_comparison import _available_universe, _rebal_dates

_TAG = re.compile(r"\[\[\[\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)\s*\]\]\]", re.I)


def _decision_distribution() -> Counter:
    c: Counter = Counter()
    for f in glob.glob("compare_lab/.cache/*.json"):
        if "smoke" in f:
            continue
        m = _TAG.findall(json.load(open(f))["response"])
        c[m[-1].upper() if m else "NO_TAG"] += 1
    return c


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_lab/output")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = _available_universe(UNIVERSE)
    ctx = load_context(universe=universe)
    rebal = _rebal_dates(ctx.adj_close.index)

    providers = [EqualWeightProvider(), MomentumProvider(),
                 LLMProvider(VLLMClient())]
    series: dict[str, pd.Series] = {}
    for p in providers:
        w = p.weights(ctx, rebal)
        r = run_backtest(w, ctx)
        oos = r[(r.index >= pd.Timestamp(OOS_START)) & (r.index < pd.Timestamp(OOS_END))]
        series[p.name] = oos

    rets = pd.DataFrame(series).dropna(how="all")
    equity = (1.0 + rets.fillna(0.0)).cumprod()
    rets.to_csv(out / "oos_daily_returns.csv")
    equity.to_csv(out / "oos_equity.csv")

    print(f"universe ({len(universe)}): {', '.join(universe)}")
    print(f"OOS: {OOS_START.date()} -> {OOS_END.date()}  rebal={REBAL_FREQ}  "
          f"trading days={len(rets)}")
    print("\n=== headline metrics ===")
    for name, s in series.items():
        m = all_metrics(s)
        pos = int((s.dropna() > 0).sum()); tot = int(s.dropna().shape[0])
        print(f"{name:16s} CR={m['CR']:+.3f} SR={m['SR']:.3f} "
              f"HR={m['HR']:.4f} ({pos}/{tot} up-days) MDD={m['MDD']:.3f}")

    print("\n=== pairwise daily-return correlation ===")
    print(rets.corr().round(3).to_string())

    print("\n=== LLM decision distribution (cached) ===")
    dist = _decision_distribution()
    tot = sum(dist.values())
    for k in ["STRONG_BUY", "BUY", "HOLD", "SELL", "STRONG_SELL", "NO_TAG"]:
        v = dist.get(k, 0)
        print(f"  {k:12s} {v:4d}  ({100*v/tot:.1f}%)")
    print(f"  total decisions: {tot}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
