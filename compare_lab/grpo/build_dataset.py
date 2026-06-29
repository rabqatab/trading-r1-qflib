"""Build the GRPO prompt dataset (paper §5, our v1-base RL).

Each row is the SAME label-free inference prompt the student sees at eval
(snapshot → thesis), carrying the deterministic volatility `label` so the
decision reward can score the generated call. Pre-2024 (`2017-01..2023-12`,
leak-safe vs the 2024-2026 OOS), price+technical input — matching the SFT input
so GRPO isolates *RL on the reward* from any input change.

Built on node 1 (needs qf-lib-harness for snapshots/labels), then rsync'd to
node 2 as plain jsonl for training.

    uv run python -m compare_lab.grpo.build_dataset --n 300 --out compare_lab/grpo/data
"""
from __future__ import annotations

import argparse
import json
from hashlib import sha1
from pathlib import Path

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.config import (MM_TRAIN_END, MM_TRAIN_START, UNIVERSE,
                                 UNIVERSE_MM)
from compare_lab.labeling import make_signal
from compare_lab.multimodal_context import MultiModalStore
from compare_lab.providers.llm import _PROMPT_HEADER
from compare_lab.run_comparison import _available_universe
from compare_lab.sft.distill import TRAIN_END, TRAIN_START, _balanced_examples
from compare_lab.snapshot import MarketSnapshotBuilder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_lab/grpo/data")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--multimodal", action="store_true",
                    help="append news/fundamentals/sentiment/macro (12-eq, 2024 window)")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = _available_universe(UNIVERSE_MM if args.multimodal else UNIVERSE)
    ctx = load_context(universe=universe)
    builder = MarketSnapshotBuilder(
        ctx, multimodal=MultiModalStore() if args.multimodal else None)
    start = MM_TRAIN_START if args.multimodal else TRAIN_START
    end = MM_TRAIN_END if args.multimodal else TRAIN_END

    # continuous vol-adjusted signal per ticker (the value the label is cut from),
    # carried per-example for the graded GRPO reward.
    signals = {t: make_signal(ctx.adj_close[t].dropna(), forward=True) for t in universe}

    records = []
    for t, d, label in _balanced_examples(ctx, universe, args.n, start=start, end=end):
        snap = builder.build(t, d)
        if snap.endswith("no data."):
            continue
        sig = float(signals[t].get(d, float("nan")))
        if sig != sig:                       # NaN guard (shouldn't happen where label valid)
            continue
        records.append({
            "prompt": [{"role": "user", "content": _PROMPT_HEADER + snap}],
            "label": label,
            "signal": sig,
        })

    train, val = [], []
    for r in records:
        h = int(sha1(r["prompt"][0]["content"].encode()).hexdigest(), 16) % 10
        (val if h == 0 else train).append(r)
    for name, rows in [("train", train), ("val", val)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    dist: dict[str, int] = {}
    for r in records:
        dist[r["label"]] = dist.get(r["label"], 0) + 1
    print(f"built {len(records)} prompts  train={len(train)} val={len(val)}")
    print(f"label distribution: {dict(sorted(dist.items()))}")
    print(f"wrote {out}/train.jsonl, {out}/val.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
