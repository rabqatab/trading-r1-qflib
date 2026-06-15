"""LLMProvider (#2) - prompt-only LLM signal mapped to target weights.

For each (rebal date, ticker): build a snapshot, ask the LLM for a 5-class
decision, map {STRONG_BUY, BUY} -> held (else flat). MVP is long-only with a
fixed position budget (size = 1/max_positions); held > budget is capped by
rank-order of class strength. (spec §4.2, §4.3)
"""
from __future__ import annotations

import re

import pandas as pd

from compare_lab.config import MAX_POSITIONS
from compare_lab.llm_client import VLLMClient
from compare_lab.providers.base import SignalProvider
from compare_lab.snapshot import MarketSnapshotBuilder

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


def parse_decision(text: str) -> str:
    matches = _PATTERN.findall(text or "")
    if not matches:
        return "HOLD"
    return matches[-1].upper()


class LLMProvider(SignalProvider):
    name = "llm_prompt_only"

    def __init__(self, client: VLLMClient, max_positions: int = MAX_POSITIONS):
        self._client = client
        self._builder = None
        self.max_positions = max_positions

    def weights(self, ctx, rebal_dates: pd.DatetimeIndex) -> pd.DataFrame:
        self._builder = MarketSnapshotBuilder(ctx)
        cols = list(ctx.universe)
        w = pd.DataFrame(0.0, index=rebal_dates, columns=cols)
        size = 1.0 / self.max_positions
        for d in rebal_dates:
            strengths: dict[str, int] = {}
            for t in cols:
                snap = self._builder.build(t, d)
                key = self._builder.snapshot_hash(t, d)
                reply = self._client.complete(_PROMPT_HEADER + snap, key=key)
                decision = parse_decision(reply)
                if decision in _HELD:
                    strengths[t] = _STRENGTH[decision]
            # cap to budget by class strength (ties: stable order)
            held = sorted(strengths, key=lambda t: (-strengths[t], cols.index(t)))
            for t in held[: self.max_positions]:
                w.at[d, t] = size
        return w
