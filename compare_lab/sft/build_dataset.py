"""Build the SFT v0 dataset: (snapshot prompt) -> (templated thesis + decision).

v0 design (roadmap Week 3): the rationale is *templated*, not teacher-distilled —
the goal is to teach the model (a) the exact output format and (b) to align its
5-class call with the deterministic volatility label. A stronger teacher
(Qwen3-32B) distillation is Phase 1.

Leakage guard: training examples are drawn from BEFORE the qf-lib OOS window
(2024-01-02). We sample (ticker, day) pairs in [TRAIN_START, TRAIN_END] so the
SFT model never sees labels from the evaluation period.

Output: JSONL, one chat record per line:
    {"messages": [{"role": "user", "content": <header+snapshot>},
                  {"role": "assistant", "content": <thesis ... [[[CLASS]]]>}]}

    uv run python -m compare_lab.sft.build_dataset --out compare_lab/sft/data
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime
from hashlib import sha1
from pathlib import Path

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.config import (MM_TRAIN_END, MM_TRAIN_START, UNIVERSE,
                                 UNIVERSE_MM)
from compare_lab.labeling import make_labels
from compare_lab.multimodal_context import MultiModalStore
from compare_lab.providers.llm import _PROMPT_HEADER
from compare_lab.run_comparison import _available_universe
from compare_lab.snapshot import MarketSnapshotBuilder

TRAIN_START = datetime(2017, 1, 3)
TRAIN_END = datetime(2023, 12, 29)      # strictly before the 2024-01 OOS window
SAMPLE_EVERY = 5                        # ~every 5th trading day per ticker

_DIRECTION = {
    "STRONG_BUY": ("strongly bullish", "a high-conviction long"),
    "BUY": ("bullish", "a long"),
    "HOLD": ("neutral", "no position change"),
    "SELL": ("bearish", "trimming exposure"),
    "STRONG_SELL": ("strongly bearish", "a high-conviction reduction"),
}
_IND = re.compile(r"^\s{2}(\w[\w_]*): ([\-\d\.napbmk]+)$", re.M)


def _indicators(snapshot: str) -> dict[str, str]:
    return {k: v for k, v in _IND.findall(snapshot)}


def _thesis(label: str, ind: dict[str, str], seed: str) -> str:
    """Templated, value-grounded thesis consistent with the label."""
    tone, action = _DIRECTION[label]
    rsi = ind.get("rsi_14", "na")
    macd = ind.get("macd", "na")
    sma50 = ind.get("close_50_sma", "na")
    sma200 = ind.get("close_200_sma", "na")
    # light deterministic variation so completions aren't identical strings
    v = int(sha1(seed.encode()).hexdigest(), 16) % 3
    openers = [
        "Reading the price and technical structure,",
        "On a disciplined read of the tape,",
        "Weighing trend against momentum,",
    ]
    body = (
        f"the medium-term trend is {tone}: price sits relative to its 50-day "
        f"({sma50}) and 200-day ({sma200}) moving averages, RSI(14) is {rsi}, "
        f"and MACD reads {macd}. "
    )
    close = (
        f"Net of trend and momentum, the next-week posture is {tone}, warranting "
        f"{action}.\n\n[[[{label}]]]"
    )
    return f"{openers[v]} {body}{close}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_lab/sft/data")
    ap.add_argument("--every", type=int, default=SAMPLE_EVERY)
    ap.add_argument("--balance", action="store_true",
                    help="down-sample dominant classes (anti-HOLD-collapse)")
    ap.add_argument("--multimodal", action="store_true",
                    help="append news/fundamentals/sentiment/macro (12-eq, 2024 window)")
    ap.add_argument("--limit", type=int, default=0, help="cap records (smoke)")
    ap.add_argument("--train-start", default=None)
    ap.add_argument("--train-end", default=None)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = _available_universe(UNIVERSE_MM if args.multimodal else UNIVERSE)
    ctx = load_context(universe=universe)
    builder = MarketSnapshotBuilder(
        ctx, multimodal=MultiModalStore() if args.multimodal else None)
    start = pd.Timestamp(args.train_start
                         or (MM_TRAIN_START if args.multimodal else TRAIN_START))
    end = pd.Timestamp(args.train_end
                       or (MM_TRAIN_END if args.multimodal else TRAIN_END))

    records = []
    dist: dict[str, int] = {}
    for t in universe:
        labels = make_labels(ctx.adj_close[t].dropna(), forward=True)
        window = labels[(labels.index >= start) & (labels.index <= end)].dropna()
        for d in window.index[:: args.every]:
            label = window.loc[d]
            snap = builder.build(t, d)
            if snap.endswith("no data."):
                continue
            ind = _indicators(snap)
            user = _PROMPT_HEADER + snap
            assistant = _thesis(label, ind, seed=f"{t}-{d.date()}")
            records.append({"label": label, "messages": [
                {"role": "user", "content": user},
                {"role": "assistant", "content": assistant},
            ]})
            dist[label] = dist.get(label, 0) + 1
        if args.limit and len(records) >= args.limit:
            break

    if args.balance:
        # cap each class at the 2nd-smallest class count (down-sample the
        # dominant HOLD/BUY so the decision token isn't a constant-HOLD prior).
        counts = sorted(dist.values())
        cap = counts[1] if len(counts) > 1 else counts[0]
        by_cls: dict[str, list] = {}
        for r in records:
            by_cls.setdefault(r["label"], []).append(r)
        records = [r for cls in sorted(by_cls)
                   for r in by_cls[cls][:cap]]          # stable (date-ordered)
        dist = {k: min(v, cap) for k, v in dist.items()}
        print(f"[balance] cap={cap} per class -> {len(records)} records")

    records = [{"messages": r["messages"]} for r in records]

    # deterministic split (90/10) by record hash
    train, val = [], []
    for r in records:
        h = int(sha1(r["messages"][0]["content"].encode()).hexdigest(), 16) % 10
        (val if h == 0 else train).append(r)

    for name, rows in [("train", train), ("val", val)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"universe={len(universe)}  total={len(records)}  "
          f"train={len(train)}  val={len(val)}")
    print("label distribution:", {k: dist[k] for k in sorted(dist)})
    print(f"wrote {out}/train.jsonl, {out}/val.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
