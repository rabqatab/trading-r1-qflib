"""Build the 2025-H1 OOS eval set for the top-150 universe (learning-curve + MM-ceiling).

The coauthor's SFT/GRPO bundle is all 2024 (train AND val), so the shipped val is
an in-window holdout, not time-OOS. This builds a fresh time-OOS grid on 2025-H1
(post the 2024 training window, still inside the news window that ends 2025-06-30),
using the SAME snapshot code that built the training prompts -> exact train/eval parity.

Two prompt variants per (ticker, as_of), same sample:
  - mm         : full multimodal (price+tech + NEWS/FUNDAMENTALS/SENTIMENT/MACRO), matches training
  - priceonly  : price+technical only (multimodal=None)

Truth = make_signal(adj_close, forward=True) at each (ticker, as_of) -> the continuous
vol-adjusted proxy the 5-class label is cut from. IC downstream = rank-corr(pred_class, signal).

    uv run python -m compare_lab.build_oos150_eval --n 1000 --out compare_lab/eval150
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401  (sys.path setup for alpha_lab)
from alpha_lab.core import load_context
from compare_lab.labeling import make_signal
from compare_lab.multimodal_context import MultiModalStore
from compare_lab.snapshot import MarketSnapshotBuilder

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_STORE = Path("data/qflib_store_top150_canonical")

# Instruction header — copied verbatim from the coauthor's training prompts so the
# eval prompt is byte-identical in structure to what the model was trained on.
_INSTRUCTION = (
    "You are a disciplined equity analyst. Based only on the price and technical "
    "data below, decide a 5-class trading signal for the next week. Choose exactly "
    "one of: STRONG_BUY, BUY, HOLD, SELL, STRONG_SELL. End your reply with that "
    "single choice on its own line wrapped in triple brackets, for example [[[BUY]]]."
)


def _anchors(index: pd.DatetimeIndex, start: str, end: str) -> pd.DatetimeIndex:
    """Weekly (W-FRI) rebalance anchors snapped to the last trading day <= anchor."""
    snapped = []
    for a in pd.date_range(start, end, freq="W-FRI"):
        prior = index[index <= a]
        if len(prior):
            snapped.append(prior[-1])
    return pd.DatetimeIndex(sorted(set(snapped)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000, help="subsample size (fixed seed)")
    ap.add_argument("--start", default="2025-01-03")
    ap.add_argument("--end", default="2025-06-27")
    ap.add_argument("--out", default="compare_lab/eval150")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tickers = sorted(pd.read_parquet(_PRICES, columns=["ticker"])["ticker"].unique())
    print(f"[eval] universe: {len(tickers)} tickers")
    ctx = load_context(universe=tickers, parquet_path=_PRICES,
                       data_start=pd.Timestamp("2015-01-01"),
                       data_end=pd.Timestamp("2026-05-08"))
    mm = MultiModalStore(store_dir=_STORE)
    builder_mm = MarketSnapshotBuilder(ctx, multimodal=mm)
    builder_px = MarketSnapshotBuilder(ctx, multimodal=None)

    # forward signal truth per ticker (NaN where undefined, e.g. last 15 bars)
    signal = {t: make_signal(ctx.adj_close[t].dropna(), forward=True) for t in tickers}

    anchors = _anchors(ctx.adj_close.index, args.start, args.end)
    print(f"[eval] {len(anchors)} weekly anchors {anchors[0].date()}..{anchors[-1].date()}")

    # full grid of (ticker, as_of) with a defined truth signal
    grid = [(t, a) for t in tickers for a in anchors
            if a in signal[t].index and pd.notna(signal[t].loc[a])]
    rng = np.random.default_rng(args.seed)
    if len(grid) > args.n:
        idx = rng.choice(len(grid), size=args.n, replace=False)
        grid = [grid[i] for i in sorted(idx)]
    print(f"[eval] building {len(grid)} eval points (both variants)...")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    f_mm = (out / "eval_mm.jsonl").open("w")
    f_px = (out / "eval_priceonly.jsonl").open("w")
    n_written = 0
    for t, a in grid:
        p_mm = builder_mm.build(t, a)
        p_px = builder_px.build(t, a)
        if "no data" in p_mm or "no data" in p_px:
            continue
        base = {"ticker": t, "as_of": a.strftime("%Y-%m-%d"),
                "signal": float(signal[t].loc[a])}
        for f, prompt in ((f_mm, p_mm), (f_px, p_px)):
            rec = dict(base)
            rec["prompt"] = [{"role": "user",
                              "content": f"{_INSTRUCTION}\n\n{prompt}"}]
            f.write(json.dumps(rec) + "\n")
        n_written += 1
    f_mm.close()
    f_px.close()
    print(f"[eval] wrote {n_written} points -> {out}/eval_mm.jsonl, eval_priceonly.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
