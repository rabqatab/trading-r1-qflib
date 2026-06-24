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
from compare_lab.config import UNIVERSE
from compare_lab.providers.llm import _PROMPT_HEADER
from compare_lab.run_comparison import _available_universe
from compare_lab.sft.distill import _balanced_examples
from compare_lab.snapshot import MarketSnapshotBuilder


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_lab/grpo/data")
    ap.add_argument("--n", type=int, default=300)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    universe = _available_universe(UNIVERSE)
    ctx = load_context(universe=universe)
    builder = MarketSnapshotBuilder(ctx)

    records = []
    for t, d, label in _balanced_examples(ctx, universe, args.n):
        snap = builder.build(t, d)
        if snap.endswith("no data."):
            continue
        records.append({
            "prompt": [{"role": "user", "content": _PROMPT_HEADER + snap}],
            "label": label,
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
