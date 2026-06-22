"""Teacher distillation for SFT v2 (paper §3.6 / §4.3, our local-distill variant).

Reverse-reasoning distillation: a strong local teacher (Qwen3-30B-A3B) is shown
the snapshot AND the deterministic volatility label, and asked to write a §8-format
XML thesis that *justifies that call*, grounding each bullet in the actual data
with *italic quotes* and `backtick sources` (so it scores on the GRPO evidence
reward). The training pair stored is (label-free inference prompt → thesis): the
student learns to produce the reasoning + decision from the snapshot alone.

No reject sampling (the volatility label is ground truth), but we record the
structure/decision reward per thesis so low-quality ones can be filtered.

Pre-2024 window (leak-safe vs the 2024-2026 eval); price+technical input (the
multi-modal store only starts 2024), matching the v0/v1 SFT input so v2 isolates
*teacher reasoning vs template*.

    VLLM_BASE_URL=... VLLM_MODEL=teacher uv run python -m compare_lab.sft.distill \
        --out compare_lab/sft/data_v2 --n 250
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from hashlib import sha1

import pandas as pd

import compare_lab  # noqa: F401
from alpha_lab.core import load_context
from compare_lab.config import UNIVERSE
from compare_lab.grpo.rewards import decision_reward, structure_reward
from compare_lab.labeling import make_labels
from compare_lab.llm_client import VLLMClient
from compare_lab.providers.llm import _PROMPT_HEADER
from compare_lab.run_comparison import _available_universe
from compare_lab.snapshot import MarketSnapshotBuilder

TRAIN_START = datetime(2017, 1, 3)
TRAIN_END = datetime(2023, 12, 29)

_TEACHER_INSTR = (
    "You are a senior equity analyst. The volatility-adjusted ground-truth call "
    "for this name on this date is [[[{label}]]]. Write a structured investment "
    "thesis that JUSTIFIES exactly this call using ONLY the data below.\n"
    "Strict format:\n"
    "- 5 to 7 analysis sections, each a markdown header (## TREND, ## MOMENTUM, "
    "## VOLATILITY, ## RISK, ## SETUP, ...).\n"
    "- Each section: one **bold** intro sentence, then 4-7 bullets. Each bullet = "
    "an opinion of 15-90 words, then the evidence as an *italic quote* of an actual "
    "value from the data, then the `source` in backticks (e.g. `RSI(14)`, `MACD`, "
    "`50-day SMA`).\n"
    "- Finish with `## CONCLUSION` and a final line containing exactly "
    "[[[{label}]]].\n\nData:\n{snapshot}\n"
)


def _balanced_examples(ctx, universe, n):
    """Pick ~n (ticker, date, label) triples, balanced across the 5 classes."""
    per_class: dict[str, list] = {}
    for t in universe:
        labels = make_labels(ctx.adj_close[t].dropna(), forward=True)
        win = labels[(labels.index >= pd.Timestamp(TRAIN_START))
                     & (labels.index <= pd.Timestamp(TRAIN_END))].dropna()
        for d in win.index[::7]:                      # thin to keep it spread out
            per_class.setdefault(win.loc[d], []).append((t, d, win.loc[d]))
    cap = max(1, n // 5)
    out = []
    for cls in sorted(per_class):
        # deterministic stride sample within the class
        items = per_class[cls]
        stride = max(1, len(items) // cap)
        out.extend(items[::stride][:cap])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="compare_lab/sft/data_v2")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--max-tokens", type=int, default=1600)
    args = ap.parse_args()
    from pathlib import Path
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)

    universe = _available_universe(UNIVERSE)
    ctx = load_context(universe=universe)
    builder = MarketSnapshotBuilder(ctx)
    client = VLLMClient()                              # teacher endpoint via env

    triples = _balanced_examples(ctx, universe, args.n)
    records, struct_scores, dec_ok = [], [], 0
    for t, d, label in triples:
        snap = builder.build(t, d)
        if snap.endswith("no data."):
            continue
        prompt = _TEACHER_INSTR.format(label=label, snapshot=snap)
        key = "distill-" + sha1(f"{t}{d}{label}".encode()).hexdigest()[:12]
        thesis = client.complete(prompt, key=key)
        struct_scores.append(structure_reward(thesis))
        dec_ok += int(decision_reward(thesis, label) > 0)   # teacher emitted the right call
        # store the LABEL-FREE inference prompt -> thesis pair
        user = _PROMPT_HEADER + snap
        records.append({"messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": thesis},
        ]})

    train, val = [], []
    for r in records:
        h = int(sha1(r["messages"][0]["content"].encode()).hexdigest(), 16) % 10
        (val if h == 0 else train).append(r)
    for name, rows in [("train", train), ("val", val)]:
        with open(out / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    n = len(records)
    print(f"distilled {n} theses  train={len(train)} val={len(val)}")
    print(f"teacher decision-correct: {dec_ok}/{n} "
          f"({100*dec_ok/max(n,1):.0f}%)  mean structure reward: "
          f"{sum(struct_scores)/max(len(struct_scores),1):.2f}")
    print(f"wrote {out}/train.jsonl, {out}/val.jsonl")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
