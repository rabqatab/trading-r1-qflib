"""Distillation v3.1 prep/filter — label-first + self-consistency filter.

Two literature-backed fixes (docs/2026-07-04-distillation-v31-design.md):
  1. LABEL-FIRST (Wadhwa EMNLP 2024): student target = "[[[LABEL]]]\\n{thesis}" instead of
     "{thesis}…[[[LABEL]]]" — kills the non-termination tax, keeps rationale as a regularizer.
  2. SELF-CONSISTENCY FILTER (STaR + SCOTT): keep only theses whose BODY (tag stripped) makes an
     independent reader (base Qwen3-4B) re-derive the true label — drops fluent noise-justifications.

Input = the QC-passed clean v3 corpus (data_top150_distill_clean); no new Opus calls.

    # 1. build the reader-eval set (CPU):
    uv run python -m compare_lab.sft.distill_v31 --stage prep
    # 2. run reader inference (sparkq) -> reader_preds.jsonl, then:
    uv run python -m compare_lab.sft.distill_v31 --stage filter
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

_TAG = re.compile(r"\[\[\[\s*(STRONG_SELL|SELL|HOLD|BUY|STRONG_BUY)\s*\]\]\]")
_SRC = Path("compare_lab/sft/data_top150_distill_clean")
_WORK = Path("compare_lab/sft/data_top150_distill_v31")
_READER_INSTR = (
    "You are given an equity analyst's note; its final recommendation has been removed. "
    "Based ONLY on this note, give the single next-week call as EXACTLY one of "
    "STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY wrapped in triple brackets (e.g. [[[BUY]]]) "
    "and nothing else.\n\nNote:\n{body}"
)


def _split(content: str):
    """(label, thesis-body-without-tag) from a thesis-first assistant message."""
    m = _TAG.findall(content)
    label = m[-1] if m else None
    body = content.split("[[[")[0].rstrip()
    return label, body


def prep():
    _WORK.mkdir(parents=True, exist_ok=True)
    reader = (_WORK / "reader_eval.jsonl").open("w")
    meta = (_WORK / "_meta.jsonl").open("w")
    n = 0
    for split in ("train", "val"):
        for line in (_SRC / f"{split}.jsonl").open():
            rec = json.loads(line)
            snap = rec["messages"][0]["content"]
            label, body = _split(rec["messages"][1]["content"])
            if not label or not body:
                continue
            key = f"{split}-{n}"
            # infer_ic.py contract: needs ticker/as_of/signal/prompt — carry key in 'ticker'
            reader.write(json.dumps({"ticker": key, "as_of": "x", "signal": 0.0,
                "prompt": [{"role": "user", "content": _READER_INSTR.format(body=body)}]}) + "\n")
            meta.write(json.dumps({"key": key, "split": split, "label": label,
                                   "snapshot": snap, "body": body}) + "\n")
            n += 1
    reader.close(); meta.close()
    print(f"[v31 prep] wrote {n} reader prompts -> {_WORK}/reader_eval.jsonl")


def filter_():
    preds = {json.loads(l)["ticker"]: json.loads(l)["pred"]
             for l in (_WORK / "reader_preds.jsonl").open()}
    kept = {"train": [], "val": []}
    n = agree = noread = 0
    for line in (_WORK / "_meta.jsonl").open():
        m = json.loads(line); n += 1
        r = preds.get(m["key"])
        if r is None:
            noread += 1; continue
        if r != m["label"]:
            continue
        agree += 1
        kept[m["split"]].append({"messages": [
            {"role": "user", "content": m["snapshot"]},
            {"role": "assistant", "content": f"[[[{m['label']}]]]\n{m['body']}"},  # label-first
        ]})
    for split, rows in kept.items():
        with (_WORK / f"{split}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"[v31 filter] {n} theses | reader self-consistent {agree} ({100*agree/n:.0f}%) | "
          f"no-tag {noread} | -> train {len(kept['train'])} val {len(kept['val'])} @ {_WORK}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["prep", "filter"], required=True)
    args = ap.parse_args()
    (prep if args.stage == "prep" else filter_)()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
