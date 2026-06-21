"""Quick probe: does the SFT v0 LoRA shift decisions vs the base model?

Samples cached base-model replies, re-asks the same prompts to the served
`sft-v0` LoRA, and prints the base->SFT decision-shift counts. A cheap sanity
check before committing to the full ~1.6k-call SFT backtest: if SFT just mirrors
base, the full run tells us little.

    VLLM_MODEL=sft-v0 uv run python -m compare_lab.tests.probe_sft_vs_base [N]
"""
from __future__ import annotations

import glob
import json
import sys
from collections import Counter

import compare_lab  # noqa: F401
from compare_lab.llm_client import _default_transport, DEFAULT_BASE_URL
from compare_lab.providers.llm import parse_decision

_ORDER = ["STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY"]


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    files = sorted(f for f in glob.glob("compare_lab/.cache/*.json")
                   if "smoke" not in f)
    stride = max(1, len(files) // n)
    sample = files[::stride][:n]
    call = _default_transport(DEFAULT_BASE_URL, "sft-v0")

    base_dist, sft_dist, shifts = Counter(), Counter(), Counter()
    changed = 0
    for f in sample:
        d = json.load(open(f))
        base = parse_decision(d["response"])
        sft = parse_decision(call(d["prompt"]))
        base_dist[base] += 1
        sft_dist[sft] += 1
        shifts[(base, sft)] += 1
        changed += (base != sft)
    print(f"sampled {len(sample)}  changed {changed} ({100*changed/len(sample):.0f}%)")
    print("base dist:", {k: base_dist.get(k, 0) for k in _ORDER})
    print("sft  dist:", {k: sft_dist.get(k, 0) for k in _ORDER})
    print("notable shifts (base -> sft):")
    for (b, s), c in shifts.most_common(10):
        if b != s:
            print(f"  {b} -> {s}: {c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
