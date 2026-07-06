"""Track B (#5) eval builder: headline-only vs headline+summary, SAME Finnhub news.

Clean ablation of whether article-summary text raises IC over headlines. Both variants use the
same 2025-H1 OOS grid, same Finnhub summary store (news = news_top150_summ.parquet), same snapshot
code — the ONLY difference is whether each news line carries its `summary`. Eval on the base model
(prompt-only) so there's no training confound: a pure test of incremental text information.

    uv run python -m compare_lab.build_oos150_summ_eval --n 1000
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.build_oos150_eval import _INSTRUCTION, _anchors
from compare_lab.labeling import make_signal
from compare_lab.multimodal_context import MultiModalStore
from compare_lab.snapshot import MarketSnapshotBuilder

_PRICES = Path("data/qflib_data_store_top150/prices_top150.parquet")
_STORE = Path("data/qflib_store_top150_summ")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--start", default="2025-01-03")
    ap.add_argument("--end", default="2025-06-27")
    ap.add_argument("--out", default="compare_lab/eval150_summ")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    tickers = sorted(pd.read_parquet(_PRICES, columns=["ticker"])["ticker"].unique())
    ctx = load_context(universe=tickers, parquet_path=_PRICES,
                       data_start=pd.Timestamp("2015-01-01"), data_end=pd.Timestamp("2026-05-08"))
    mm = MultiModalStore(store_dir=_STORE)
    # two builders share ctx+store; the only difference is with_summary at render time
    b_hl = MarketSnapshotBuilder(ctx, multimodal=mm)      # headline-only
    b_su = MarketSnapshotBuilder(ctx, multimodal=mm)
    b_su._mm_summary = True                               # flag consumed below via monkey render

    signal = {t: make_signal(ctx.adj_close[t].dropna(), forward=True) for t in tickers}
    anchors = _anchors(ctx.adj_close.index, args.start, args.end)
    grid = [(t, a) for t in tickers for a in anchors
            if a in signal[t].index and pd.notna(signal[t].loc[a])]
    rng = np.random.default_rng(args.seed)
    if len(grid) > args.n:
        grid = [grid[i] for i in sorted(rng.choice(len(grid), args.n, replace=False))]
    print(f"[summ-eval] {len(grid)} points, 2 variants (headline / +summary)", flush=True)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    f_hl = (out / "eval_headline.jsonl").open("w")
    f_su = (out / "eval_summary.jsonl").open("w")
    n = 0
    for t, a in grid:
        # render both by calling build with the store's with_summary toggled
        mm._render_summary = False
        p_hl = _build_with_summary(b_hl, mm, t, a, False)
        p_su = _build_with_summary(b_su, mm, t, a, True)
        if "no data" in p_hl or "no data" in p_su:
            continue
        base = {"ticker": t, "as_of": a.strftime("%Y-%m-%d"), "signal": float(signal[t].loc[a])}
        for f, prompt in ((f_hl, p_hl), (f_su, p_su)):
            rec = dict(base)
            rec["prompt"] = [{"role": "user", "content": f"{_INSTRUCTION}\n\n{prompt}"}]
            f.write(json.dumps(rec) + "\n")
        n += 1
    f_hl.close(); f_su.close()
    print(f"[summ-eval] wrote {n} points -> {out}/eval_headline.jsonl, eval_summary.jsonl", flush=True)
    return 0


def _build_with_summary(builder, mm, ticker, as_of, with_summary: bool) -> str:
    """Rebuild the snapshot, forcing render_sections(with_summary=...)."""
    df = builder._window(ticker, as_of)
    if df.empty:
        return f"Ticker {ticker}: no data."
    from stockstats import StockDataFrame
    from compare_lab.snapshot import _INDICATORS, _OUTPUT_WINDOW, _abbrev
    sdf = StockDataFrame.retype(df.copy())
    for ind in _INDICATORS:
        _ = sdf[ind]
    recent = sdf.tail(_OUTPUT_WINDOW)
    lines = [f"Ticker: {ticker}",
             f"As of: {pd.Timestamp(as_of).date()} (showing last {len(recent)} trading days)",
             "", "Date | Open High Low Close Volume"]
    for d, row in recent.iterrows():
        lines.append(f"{d.strftime('%Y-%m-%d')} | {_abbrev(row['open'])} {_abbrev(row['high'])} "
                     f"{_abbrev(row['low'])} {_abbrev(row['close'])} {_abbrev(row['volume'])}")
    last = recent.iloc[-1]
    lines += ["", "Indicators (latest):"]
    for ind in _INDICATORS:
        lines.append(f"  {ind}: {_abbrev(last.get(ind, float('nan')))}")
    lines.append("")
    lines.append(mm.render_sections(ticker, as_of, with_summary=with_summary))
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
