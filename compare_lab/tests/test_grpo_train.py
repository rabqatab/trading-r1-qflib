"""GRPO reward-wrapper checks: text extraction from conversational completions
and per-example label alignment in the decision reward. Importing `train` only
pulls in `rewards` (light) — torch/trl are imported inside main(), not at module
load — so this runs without a GPU."""
from compare_lab.grpo import train


def _comp(text):
    return [{"role": "assistant", "content": text}]


def test_decision_wrapper_aligns_labels():
    comps = [_comp("## CONCLUSION\n[[[BUY]]]"), _comp("## CONCLUSION\n[[[SELL]]]")]
    # both calls match their label -> both positive (diagonal of the matrix)
    assert all(r > 0 for r in train.reward_decision(comps, label=["BUY", "SELL"]))
    # both wrong (swapped) -> both negative
    assert all(r < 0 for r in train.reward_decision(comps, label=["SELL", "BUY"]))


def test_structure_evidence_wrappers_return_one_score_per_completion():
    comps = [_comp("## TREND\n" + "word " * 60), _comp("no sections here")]
    assert len(train.reward_structure(comps)) == 2
    assert len(train.reward_evidence(comps)) == 2
