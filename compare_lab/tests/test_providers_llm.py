import numpy as np
import pandas as pd

import compare_lab  # noqa: F401
from compare_lab.llm_client import VLLMClient
from compare_lab.providers.llm import (
    LLMProvider,
    parse_decision,
    parse_decision_status,
)


def test_parse_decision_variants():
    assert parse_decision("blah\n[[[STRONG_BUY]]]") == "STRONG_BUY"
    assert parse_decision("text [[[buy]]] more") == "BUY"
    assert parse_decision("no decision here") == "HOLD"   # default


def test_parse_decision_status_flags_unparsed():
    assert parse_decision_status("x [[[BUY]]]") == ("BUY", True)
    assert parse_decision_status("no tag at all") == ("HOLD", False)
    assert parse_decision_status("") == ("HOLD", False)


def _ctx():
    idx = pd.bdate_range("2023-01-02", periods=400)
    px = pd.DataFrame(
        {t: np.linspace(100, 150, 400) for t in ("A", "B")}, index=idx)
    vol = pd.DataFrame({t: np.full(400, 1e6) for t in ("A", "B")}, index=idx)

    class Ctx:
        adj_close = px
        open = px * 0.99
        high = px * 1.01
        low = px * 0.98
        volume = vol
        dollar_volume = px * vol
        universe = ("A", "B")
    return Ctx()


def test_llm_provider_maps_classes_to_weights(tmp_path):
    # A -> BUY (held), B -> SELL (flat)
    def fake_transport(prompt: str) -> str:
        return "[[[BUY]]]" if "Ticker: A" in prompt else "[[[SELL]]]"

    client = VLLMClient(transport=fake_transport, cache_dir=tmp_path)
    ctx = _ctx()
    rebal = ctx.adj_close.index[300::20]
    w = LLMProvider(client, max_positions=8).weights(ctx, rebal)
    last = w.iloc[-1]
    assert last["A"] > 0
    assert last["B"] == 0.0
    assert abs(last["A"] - 1 / 8) < 1e-9   # size = 1/max_positions


def test_provider_surfaces_parse_rate(tmp_path):
    # A always parses; B never emits a tag -> exactly half are NO_TAG
    def fake_transport(prompt: str) -> str:
        return "[[[BUY]]]" if "Ticker: A" in prompt else "I think hold."

    client = VLLMClient(transport=fake_transport, cache_dir=tmp_path)
    ctx = _ctx()
    rebal = ctx.adj_close.index[300::20]
    p = LLMProvider(client, max_positions=8)
    p.weights(ctx, rebal)
    assert p.parse_stats["total"] == 2 * len(rebal)
    assert p.parse_stats["no_tag"] == len(rebal)
    assert abs(p.parse_stats["no_tag_rate"] - 0.5) < 1e-9
