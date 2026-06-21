"""Manual smoke test: verify the served LLM emits a parseable 5-class decision.

Runs the exact provider chain for a few (ticker, date) pairs against the live
vLLM endpoint, then prints the raw reply + parsed class. Confirms we will NOT
get an all-HOLD degenerate signal before the ~1.4k-call full run.

Usage (with the endpoint up on localhost:8000):
    uv run python -m compare_lab.tests.smoke_llm_endpoint
"""
from __future__ import annotations

import compare_lab  # noqa: F401  (path bootstrap)
from alpha_lab.core import load_context
from compare_lab.config import OOS_START, OOS_END
from compare_lab.llm_client import VLLMClient
from compare_lab.providers.llm import _PROMPT_HEADER, parse_decision
from compare_lab.snapshot import MarketSnapshotBuilder

PROBES = [("NVDA", "2024-03-15"), ("XOM", "2024-09-20"), ("TSLA", "2025-01-17")]


def main() -> int:
    ctx = load_context(universe=("NVDA", "XOM", "TSLA"))
    builder = MarketSnapshotBuilder(ctx)
    client = VLLMClient()  # uses VLLM_BASE_URL / VLLM_MODEL env or defaults

    n_parsed = 0
    for ticker, date in PROBES:
        snap = builder.build(ticker, date)
        reply = client.complete(_PROMPT_HEADER + snap, key=f"smoke-{ticker}-{date}")
        decision = parse_decision(reply)
        had_tag = "[[[" in reply
        n_parsed += int(had_tag)
        print("=" * 70)
        print(f"{ticker} @ {date}  ->  parsed={decision}  tag_present={had_tag}")
        print(f"reply chars={len(reply)}")
        print("--- reply tail ---")
        print(reply[-400:])
    print("=" * 70)
    print(f"RESULT: {n_parsed}/{len(PROBES)} replies contained an explicit [[[...]]] tag")
    return 0 if n_parsed == len(PROBES) else 1


if __name__ == "__main__":
    raise SystemExit(main())
