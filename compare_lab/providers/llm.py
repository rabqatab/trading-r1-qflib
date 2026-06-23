"""LLMProvider (#2) - prompt-only LLM signal mapped to target weights.

For each (rebal date, ticker): build a snapshot, ask the LLM for a 5-class
decision, map {STRONG_BUY, BUY} -> held (else flat). MVP is long-only with a
fixed position budget (size = 1/max_positions); held > budget is capped by
rank-order of class strength. (spec §4.2, §4.3)
"""
from __future__ import annotations

import os
import re
from concurrent.futures import ThreadPoolExecutor
from hashlib import sha1

import pandas as pd

from compare_lab.config import MAX_POSITIONS
from compare_lab.llm_client import VLLMClient
from compare_lab.providers.base import SignalProvider
from compare_lab.snapshot import MarketSnapshotBuilder

# concurrent in-flight LLM requests; vLLM batches them. Tune via env if the
# server's KV cache (gpu-memory-utilization) starts queueing (Waiting > 0).
_MAX_WORKERS = int(os.environ.get("LLM_CONCURRENCY", "16"))

_CLASSES = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")
_HELD = {"BUY", "STRONG_BUY"}
_STRENGTH = {"STRONG_BUY": 2, "BUY": 1}
_PATTERN = re.compile(r"\[\[\[\s*(STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL)\s*\]\]\]",
                      re.IGNORECASE)

_PROMPT_HEADER = (
    "You are a disciplined equity analyst. Based only on the price and "
    "technical data below, decide a 5-class trading signal for the next week. "
    "End your reply with exactly one line: [[[STRONG_BUY|BUY|HOLD|SELL|"
    "STRONG_SELL]]].\n\n"
)


def parse_decision_status(text: str) -> tuple[str, bool]:
    """Return (decision, parsed). `parsed` is False when no [[[CLASS]]] tag was
    found — the decision then falls back to HOLD, but the caller can count it."""
    matches = _PATTERN.findall(text or "")
    if not matches:
        return "HOLD", False
    return matches[-1].upper(), True


def parse_decision(text: str) -> str:
    return parse_decision_status(text)[0]


class LLMProvider(SignalProvider):
    name = "llm_prompt_only"

    def __init__(self, client: VLLMClient, max_positions: int = MAX_POSITIONS,
                 max_no_tag_rate: float | None = 0.20, multimodal=None):
        self._client = client
        self._builder = None
        self.max_positions = max_positions
        # warn (don't crash) if more than this fraction of replies are unparseable
        self.max_no_tag_rate = max_no_tag_rate
        self._multimodal = multimodal
        self.parse_stats: dict[str, float] = {"total": 0, "no_tag": 0,
                                              "no_tag_rate": 0.0}

    def _reply(self, d, t) -> tuple[tuple, str]:
        snap = self._builder.build(t, d)
        key = sha1(snap.encode()).hexdigest()[:12]   # == snapshot_hash, one build
        return (d, t), self._client.complete(_PROMPT_HEADER + snap, key=key)

    def weights(self, ctx, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        self._builder = MarketSnapshotBuilder(ctx, multimodal=self._multimodal)
        cols = list(ctx.universe)
        w = pd.DataFrame(0.0, index=rebal_dates, columns=cols)
        size = 1.0 / self.max_positions
        # The (date, ticker) LLM calls are independent and I/O-bound; fan them out
        # so the HTTP waits overlap and vLLM batches them (Running:1 -> Running:N).
        # Cache is per-key files so concurrent writes don't collide. Assembly below
        # stays serial + ordered -> result identical regardless of finish order.
        pairs = [(d, t) for d in rebal_dates for t in cols]
        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as ex:
            replies = dict(ex.map(lambda p: self._reply(*p), pairs))
        total = no_tag = 0
        for d in rebal_dates:
            strengths: dict[str, int] = {}
            for t in cols:
                decision, parsed = parse_decision_status(replies[(d, t)])
                total += 1
                no_tag += (not parsed)
                if decision in _HELD:
                    strengths[t] = _STRENGTH[decision]
            # cap to budget by class strength (ties: stable order)
            held = sorted(strengths, key=lambda t: (-strengths[t], cols.index(t)))
            for t in held[: self.max_positions]:
                w.at[d, t] = size
        rate = no_tag / total if total else 0.0
        self.parse_stats = {"total": total, "no_tag": no_tag, "no_tag_rate": rate}
        if self.max_no_tag_rate is not None and rate > self.max_no_tag_rate:
            print(f"[compare_lab] WARNING: LLM no-tag rate {rate:.1%} "
                  f"({no_tag}/{total}) exceeds {self.max_no_tag_rate:.0%} — "
                  f"unparseable replies fell back to HOLD; tighten the prompt "
                  f"or add grammar-constrained decoding.")
        return w
