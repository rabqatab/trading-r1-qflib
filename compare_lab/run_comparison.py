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


def _rebal_dates(daily_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    anchors = pd.date_range(OOS_START, OOS_END, freq=REBAL_FREQ)
    snapped = [daily_index[daily_index <= a][-1] for a in anchors
               if (daily_index <= a).any()]
    return pd.DatetimeIndex(sorted(set(snapped)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="include prompt-only LLM")
    ap.add_argument("--out", default="compare_lab/output")
    args = ap.parse_args()

    universe = _available_universe(UNIVERSE)
    ctx = load_context(universe=universe)
    rebal = _rebal_dates(ctx.adj_close.index)

    providers = [EqualWeightProvider(), MomentumProvider()]
    if args.llm:
        from compare_lab.llm_client import VLLMClient
        from compare_lab.providers.llm import LLMProvider
        providers.append(LLMProvider(VLLMClient()))

    results: dict[str, pd.Series] = {}
    for p in providers:
        print(f"[compare_lab] {p.name}: computing weights ...")
        w = p.weights(ctx, rebal)
        print(f"[compare_lab] {p.name}: backtesting ...")
        returns = run_backtest(w, ctx)
        oos = returns[(returns.index >= pd.Timestamp(OOS_START))
                      & (returns.index < pd.Timestamp(OOS_END))]
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
