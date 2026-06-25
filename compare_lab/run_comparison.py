"""CLI: run every provider through the qf-lib backtest and report metrics.

Usage:
    uv run python -m compare_lab.run_comparison [--llm] [--out DIR]

Without --llm only the equal-weight and momentum baselines run (no server
needed). With --llm the prompt-only LLM provider also runs (requires a vLLM
endpoint reachable at config/base_url, served on DGX Spark via sparkq).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context, PRICES_PATH
from compare_lab.backtest import run_backtest
from compare_lab.config import OOS_START, OOS_END, REBAL_FREQ, UNIVERSE
from compare_lab.providers.equal_weight import EqualWeightProvider
from compare_lab.providers.momentum import MomentumProvider
from compare_lab.report import build_html, build_table


def _available_universe(universe: tuple[str, ...]) -> tuple[str, ...]:
    """Filter universe to tickers present in the local prices.parquet."""
    _tickers = pd.read_parquet(PRICES_PATH, columns=["ticker"])["ticker"].unique()
    available = set(_tickers)
    filtered = tuple(t for t in universe if t in available)
    missing = set(universe) - set(filtered)
    if missing:
        print(f"[compare_lab] WARNING: tickers not in prices.parquet (skipped): "
              f"{sorted(missing)}")
    return filtered


def _rebal_dates(daily_index: pd.DatetimeIndex, start=OOS_START,
                 end=OOS_END) -> pd.DatetimeIndex:
    anchors = pd.date_range(start, end, freq=REBAL_FREQ)
    snapped = [daily_index[daily_index <= a][-1] for a in anchors
               if (daily_index <= a).any()
               and daily_index[daily_index <= a][-1] >= pd.Timestamp(start)]
    return pd.DatetimeIndex(sorted(set(snapped)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="include prompt-only LLM")
    ap.add_argument("--out", default="compare_lab/output")
    ap.add_argument("--oos-start", default=None, help="override OOS_START (YYYY-MM-DD)")
    ap.add_argument("--oos-end", default=None, help="override OOS_END (YYYY-MM-DD)")
    ap.add_argument("--multimodal", action="store_true",
                    help="LLM row sees news/fundamentals/sentiment/macro (implies 12-eq)")
    ap.add_argument("--universe-mm", action="store_true",
                    help="use the 12-equity multimodal universe (no SPY/QQQ)")
    args = ap.parse_args()

    from compare_lab.config import MM_OOS_END, MM_OOS_START, UNIVERSE_MM
    oos_start = pd.Timestamp(args.oos_start) if args.oos_start else (
        MM_OOS_START if args.multimodal else OOS_START)
    oos_end = pd.Timestamp(args.oos_end) if args.oos_end else (
        MM_OOS_END if args.multimodal else OOS_END)

    universe = _available_universe(
        UNIVERSE_MM if (args.universe_mm or args.multimodal) else UNIVERSE)
    ctx = load_context(universe=universe)
    rebal = _rebal_dates(ctx.adj_close.index, oos_start, oos_end)

    providers = [EqualWeightProvider(), MomentumProvider()]
    if args.llm:
        from compare_lab.llm_client import VLLMClient
        from compare_lab.providers.llm import LLMProvider
        mm = None
        if args.multimodal:
            from compare_lab.multimodal_context import MultiModalStore
            mm = MultiModalStore()
        providers.append(LLMProvider(VLLMClient(), multimodal=mm))

    results: dict[str, pd.Series] = {}
    for p in providers:
        print(f"[compare_lab] {p.name}: computing weights ...")
        w = p.weights(ctx, rebal)
        stats = getattr(p, "parse_stats", None)
        if stats and stats.get("total"):
            print(f"[compare_lab] {p.name}: parse no-tag rate "
                  f"{stats['no_tag_rate']:.1%} ({stats['no_tag']}/{stats['total']})")
        print(f"[compare_lab] {p.name}: backtesting ...")
        returns = run_backtest(w, ctx)
        oos = returns[(returns.index >= pd.Timestamp(oos_start))
                      & (returns.index < pd.Timestamp(oos_end))]
        results[p.name] = oos

    table = build_table(results)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "comparison.csv", index=False)
    build_html(results, out / "equity.html")
    print(table.to_string(index=False))
    print(f"[compare_lab] wrote {out / 'comparison.csv'} and {out / 'equity.html'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
