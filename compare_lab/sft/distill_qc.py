"""QC auditor for the Opus-distilled corpus (mirrors docs/DATA_QC_RUBRIC.md).

Programmatic, free, runs on the accumulating _cache/. The format guard in distill_opus.py
only checks the tag; this checks CONTENT quality — the real risks of reverse-reasoning
distillation (teacher told the label): hallucinated numbers, forced/contradictory
justification, teacher mode-collapse (near-duplicate theses).

HARD GATES (fail -> thesis excluded from the clean corpus):
  G-FMT     ends with exactly [[[label]]], single tag == label, <= 220 words (termination-
            focused: v2's runaway was >3000 words), no '## ' header
  G-GROUND  numeric grounding >= 0.80 : each number cited appears in the snapshot (±0.5% round)

SCORED (gate-passers, 0-100 each): grounding density, diversity (1 - max Jaccard vs others),
directional faithfulness (bull/bear lexicon vs label sign), length compliance.

    # audit only:
    uv run python -m compare_lab.sft.distill_qc --cache compare_lab/sft/data_top150_distill/_cache
    # audit + write the QC-passed clean corpus for the student SFT:
    uv run python -m compare_lab.sft.distill_qc --write compare_lab/sft/data_top150_distill_clean
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

CLASSES = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")
_SIGN = {"STRONG_SELL": -2, "SELL": -1, "HOLD": 0, "BUY": 1, "STRONG_BUY": 2}
_TAG = re.compile(r"\[\[\[\s*(STRONG_SELL|SELL|HOLD|BUY|STRONG_BUY)\s*\]\]\]")
_NUM = re.compile(r"-?\d+(?:\.\d+)?")
_BULL = ("bullish", "uptrend", "upside", "long", "breakout", "rally", "recovery",
         "momentum", "strength", "support", "reclaim", "positive")
_BEAR = ("bearish", "downtrend", "downside", "short", "breakdown", "selloff", "decline",
         "weakness", "resistance", "overbought", "negative", "trim")


def _nums(text: str) -> list[float]:
    out = []
    for m in _NUM.findall(text):
        try:
            out.append(float(m))
        except ValueError:
            pass
    return out


def _grounding(thesis: str, snapshot: str) -> float:
    """Fraction of thesis numbers that match a snapshot number (±0.5% or exact round)."""
    tnums = [n for n in _nums(thesis.split("[[[")[0]) if abs(n) >= 1e-6]
    if not tnums:
        return 1.0
    snums = _nums(snapshot)
    hit = 0
    for n in tnums:
        if any(abs(n - s) <= max(0.005 * abs(s), 0.01) or round(n, 1) == round(s, 1)
               for s in snums):
            hit += 1
    return hit / len(tnums)


def _jaccard(a: set, b: set) -> float:
    return len(a & b) / len(a | b) if (a or b) else 0.0


def _faithful(thesis: str, label: str) -> float:
    body = thesis.split("[[[")[0].lower()
    b = sum(body.count(w) for w in _BULL)
    r = sum(body.count(w) for w in _BEAR)
    tone = (b - r)
    sign = _SIGN[label]
    if sign == 0:
        return 1.0 if abs(tone) <= 2 else 0.5          # HOLD: balanced
    return 1.0 if (tone > 0) == (sign > 0) or tone == 0 else 0.0  # direction agrees


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="compare_lab/sft/data_top150_distill/_cache")
    ap.add_argument("--ground-gate", type=float, default=0.80)
    ap.add_argument("--write", default=None,
                    help="dir to write the QC-passed clean train/val corpus for student SFT")
    args = ap.parse_args()

    recs = [json.loads(p.read_text()) for p in Path(args.cache).glob("*.json")]
    if not recs:
        print("no cached theses yet"); return 1
    shingles = [set(re.findall(r"\w+", r["messages"][1]["content"].lower())) for r in recs]

    rows, fmt_fail, ground_fail = [], 0, 0
    for i, r in enumerate(recs):
        thesis = r["messages"][1]["content"]
        snap = r["messages"][0]["content"]
        label = r["label"]
        tags = _TAG.findall(thesis)
        words = len(thesis.split())
        fmt_ok = (bool(tags) and tags[-1] == label and len(tags) == 1
                  and thesis.rstrip().endswith(f"[[[{label}]]]")
                  and words <= 220 and "## " not in thesis)
        g = _grounding(thesis, snap)
        # diversity: max Jaccard vs a sample of others (cap cost)
        others = shingles[:i][-200:] + shingles[i + 1:i + 201]
        dup = max((_jaccard(shingles[i], o) for o in others), default=0.0)
        faith = _faithful(thesis, label)
        gate = fmt_ok and g >= args.ground_gate
        fmt_fail += (not fmt_ok); ground_fail += (fmt_ok and g < args.ground_gate)
        rows.append({"label": label, "words": words, "ground": g, "dup": dup,
                     "faith": faith, "n_evidence": len(_nums(thesis.split("[[[")[0])),
                     "gate": gate, "rec": r})

    A = {k: np.array([r[k] for r in rows], float) for k in
         ("words", "ground", "dup", "faith", "n_evidence")}
    gate = np.array([r["gate"] for r in rows])
    n = len(rows)
    print(f"=== distill corpus QC (n={n}) ===")
    print(f"HARD GATES: format-fail {fmt_fail}  ground-fail {ground_fail}  "
          f"-> PASS {gate.sum()}/{n} ({100*gate.mean():.1f}%)")
    print(f"grounding   : mean {A['ground'].mean():.3f}  p10 {np.percentile(A['ground'],10):.3f}  "
          f"(<0.8: {100*(A['ground']<0.8).mean():.1f}%)")
    print(f"evidence/th : mean {A['n_evidence'].mean():.1f} quoted values")
    print(f"diversity   : mean max-Jaccard {A['dup'].mean():.3f}  "
          f"(near-dup >0.6: {100*(A['dup']>0.6).mean():.1f}%)")
    print(f"faithfulness: mean {A['faith'].mean():.3f}  (direction-conflict: {100*(A['faith']==0).mean():.1f}%)")
    print(f"length      : mean {A['words'].mean():.0f} words  (>140: {100*(A['words'] > 220).mean():.1f}%)")
    print(f"per-class gate-pass: " + "  ".join(
        f"{c}:{sum(1 for r in rows if r['label']==c and r['gate'])}/{sum(1 for r in rows if r['label']==c)}"
        for c in CLASSES))

    if args.write:
        out = Path(args.write); out.mkdir(parents=True, exist_ok=True)
        passers = [r for r in rows if r["gate"]]
        train, val = [], []
        for j, r in enumerate(passers):
            rec = {"messages": r["rec"]["messages"]}       # drop the QC-only 'label' key
            (val if j % 10 == 0 else train).append(rec)    # deterministic ~10% val by index
        for name, data in (("train", train), ("val", val)):
            with (out / f"{name}.jsonl").open("w") as f:
                for rec in data:
                    f.write(json.dumps(rec) + "\n")
        print(f"[write] QC-passed clean corpus -> {out}  train={len(train)} val={len(val)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
