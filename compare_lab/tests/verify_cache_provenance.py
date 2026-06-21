"""Provenance check: do the pre-existing cached LLM replies match the live model?

The disk cache is keyed by snapshot hash (model-agnostic), so a prior session's
replies are reused silently. Since the client samples at temperature 0 (greedy),
the live served model should reproduce the same *parsed 5-class decision* for the
same prompt. We re-ask a random sample through the live endpoint and compare.

Match rate ~100% => cache was produced by the same model => safe to trust.
Low match rate => cache is from a different model/config => regenerate.

Usage (endpoint up, cache populated):
    uv run python -m compare_lab.tests.verify_cache_provenance [N]
"""
from __future__ import annotations

import glob
import json
import sys

import compare_lab  # noqa: F401
from compare_lab.llm_client import _default_transport, DEFAULT_BASE_URL, DEFAULT_MODEL
from compare_lab.providers.llm import parse_decision


def main() -> int:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    files = sorted(f for f in glob.glob("compare_lab/.cache/*.json") if "smoke" not in f)
    # deterministic stride sample across the whole set (no RNG needed)
    stride = max(1, len(files) // n)
    sample = files[::stride][:n]
    call = _default_transport(DEFAULT_BASE_URL, DEFAULT_MODEL)

    match = mismatch = 0
    rows = []
    for f in sample:
        d = json.load(open(f))
        cached = parse_decision(d["response"])
        live = parse_decision(call(d["prompt"]))
        ok = cached == live
        match += ok
        mismatch += (not ok)
        if not ok:
            rows.append((f.split("/")[-1], cached, live))
    print(f"model={DEFAULT_MODEL}  base={DEFAULT_BASE_URL}")
    print(f"sample={len(sample)}  match={match}  mismatch={mismatch}  "
          f"rate={100*match/len(sample):.1f}%")
    for fn, c, l in rows:
        print(f"  MISMATCH {fn}: cached={c} live={l}")
    return 0 if mismatch == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
