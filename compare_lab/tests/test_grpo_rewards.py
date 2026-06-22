"""Tests for the GRPO reward functions (paper §5.2).

The decision reward is the verifiable core — it scores the model's [[[CLASS]]]
against the deterministic volatility label via the paper's asymmetric matrix.
Structure/evidence rewards shape the XML thesis format (§8).
"""
from __future__ import annotations

from compare_lab.grpo.rewards import (
    DECISION_MATRIX,
    CLASSES,
    decision_reward,
    evidence_reward,
    structure_reward,
)


# ---- decision reward (asymmetric matrix, §5.2 Stage III) -------------------

def test_decision_matrix_shape_and_diagonal():
    assert CLASSES == ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")
    for c in CLASSES:
        assert DECISION_MATRIX[c][c] == 1.0          # correct call = +1
    # false-bullish penalised harder than false-bearish (capital preservation)
    assert DECISION_MATRIX["STRONG_BUY"]["STRONG_SELL"] == -2.25
    assert DECISION_MATRIX["STRONG_SELL"]["STRONG_BUY"] == -2.00
    assert DECISION_MATRIX["STRONG_BUY"]["STRONG_SELL"] < DECISION_MATRIX["STRONG_SELL"]["STRONG_BUY"]


def test_decision_reward_correct_and_wrong():
    assert decision_reward("analysis...\n[[[BUY]]]", "BUY") == 1.0
    assert decision_reward("x [[[HOLD]]]", "STRONG_BUY") == -1.50   # anti-HOLD
    assert decision_reward("x [[[STRONG_BUY]]]", "STRONG_SELL") == -2.25


def test_decision_reward_takes_last_and_is_case_insensitive():
    assert decision_reward("[[[sell]]] then [[[buy]]]", "BUY") == 1.0


def test_decision_reward_no_valid_decision():
    assert decision_reward("no decision here", "BUY") == -1.5
    assert decision_reward("", "HOLD") == -1.5


def test_decision_reward_lambda_scales():
    assert decision_reward("[[[BUY]]]", "BUY", lam=2.0) == 2.0


# ---- structure reward (§5.2 Stage I) ---------------------------------------

def _thesis(n_sections: int) -> str:
    secs = []
    for i in range(n_sections):
        secs.append(f"## SECTION {i}\n"
                    f"**Intro bold.**\n"
                    f"- bullet one with enough words here to count clearly\n"
                    f"- bullet two also has a reasonable number of words\n")
    secs.append("## CONCLUSION\n[[[BUY]]]")
    return "\n".join(secs)


def test_structure_reward_rewards_5_to_7_sections():
    # 6 analysis sections + conclusion -> count reward should be at its max
    r6 = structure_reward(_thesis(6))
    r2 = structure_reward(_thesis(2))      # too few
    assert 0.0 <= r2 < r6 <= 1.0


def test_structure_reward_in_unit_range():
    for n in (1, 5, 6, 7, 10):
        r = structure_reward(_thesis(n))
        assert 0.0 <= r <= 1.0


# ---- evidence reward (§5.2 Stage II) ---------------------------------------

def test_evidence_reward_grounded_bullets_score_higher():
    grounded = (
        "## NEWS\n"
        "- The chip cycle is turning up and demand looks durable across the data "
        "center segment which supports a constructive view here "
        "*Nvidia guided revenue above consensus* `Reuters`\n"
        "- Margins are expanding as supply normalises and pricing holds firm into "
        "the next quarter according to the latest filing "
        "*operating margin rose to 62%* `10-Q`\n"
        "- Analysts keep raising targets on the back of the print and order book "
        "*price target raised to 200* `Barron's`\n"
        "- Insider activity is benign with no notable selling reported recently "
        "*no Form 4 sales filed* `SEC`\n"
        "## CONCLUSION\n[[[BUY]]]"
    )
    ungrounded = (
        "## NEWS\n- buy it\n- looks good\n## CONCLUSION\n[[[BUY]]]"
    )
    assert evidence_reward(grounded) > evidence_reward(ungrounded)
    assert 0.0 <= evidence_reward(ungrounded) <= evidence_reward(grounded) <= 1.0
