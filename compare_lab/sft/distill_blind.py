"""Investment-Thesis Distillation with REJECT SAMPLING (paper §4.3a) — Track A.

Unlike reverse-reasoning (v3, teacher SHOWN the label), here Opus predicts **blind**: it gets the
same prompt the student gets, writes a terse thesis, and commits to its OWN call. We then keep only
the theses where Opus's independent call == the true make_signal label (reject-sampling positives);
the mismatches are stored as negatives (hard cases, for a future DPO/contrastive signal). No label
leaks into the rationale, so a kept thesis is one Opus genuinely reasoned to the right answer.

Teacher = `claude -p --model opus` (subscription). Per-snapshot cache → resumable.

    # 1. blind Opus predictions (subscription; long, resumable):
    uv run python -m compare_lab.sft.distill_blind --stage predict --n 3047
    # 2. reject-sample -> label-first clean corpus + negatives:
    uv run python -m compare_lab.sft.distill_blind --stage filter
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from hashlib import sha1
from pathlib import Path

_TAG = re.compile(r"\[\[\[\s*(STRONG_SELL|SELL|HOLD|BUY|STRONG_BUY)\s*\]\]\]")
_SRC = Path("compare_lab/sft/data_top150_mm")
_OUT = Path("compare_lab/sft/data_top150_blind")
_ASK = (
    "\n\nWrite a SHORT thesis (3-4 sentences, <=110 words), each claim grounded in a specific "
    "quoted value from the data above. Do NOT hedge. The VERY LAST line must be your single call "
    "as exactly one of [[[STRONG_SELL]]] [[[SELL]]] [[[HOLD]]] [[[BUY]]] [[[STRONG_BUY]]]."
)


def _label(rec: dict) -> str | None:
    m = _TAG.findall(rec["messages"][-1]["content"])
    return m[-1] if m else None


def call_opus(prompt: str, model: str, timeout: int = 180) -> str:
    r = subprocess.run(["claude", "-p", prompt, "--model", model, "--max-turns", "1",
                        "--allowedTools", ""], capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise RuntimeError(f"claude rc={r.returncode}: {r.stderr[:160]}")
    return r.stdout.strip()


def predict_one(rec: dict, model: str, cache: Path, retries: int = 3) -> dict | None:
    snap = rec["messages"][0]["content"]
    true = _label(rec)
    if true is None:
        return None
    key = sha1(snap.encode()).hexdigest()[:16]
    cp = cache / f"{key}.json"
    if cp.exists():
        return json.loads(cp.read_text())
    prompt = snap + _ASK
    for a in range(retries):
        try:
            out = call_opus(prompt, model)
            tags = _TAG.findall(out)
            if tags and out.rstrip().endswith(f"[[[{tags[-1]}]]]"):
                body = out.split("[[[")[0].rstrip()
                rj = {"snapshot": snap, "true": true, "opus": tags[-1], "body": body}
                cp.write_text(json.dumps(rj))
                return rj
        except Exception as e:
            time.sleep((120 if "limit" in str(e).lower() else 20) * (a + 1))
    return None


def predict(n: int, model: str, conc: int):
    _OUT.mkdir(parents=True, exist_ok=True)
    cache = _OUT / "_cache"; cache.mkdir(exist_ok=True)
    recs = [json.loads(l) for l in (_SRC / "train.jsonl").open()][:n]
    print(f"[blind] {len(recs)} snapshots | teacher={model} conc={conc}", flush=True)
    ok = fail = 0
    with ThreadPoolExecutor(max_workers=conc) as ex:
        futs = [ex.submit(predict_one, r, model, cache) for r in recs]
        for i, f in enumerate(as_completed(futs), 1):
            ok += bool(f.result()); fail += not f.result()
            if i % 25 == 0 or i == len(recs):
                print(f"[blind] {i}/{len(recs)} ok={ok} fail={fail}", flush=True)


def filter_():
    cache = _OUT / "_cache"
    recs = [json.loads(p.read_text()) for p in cache.glob("*.json")]
    pos, neg = [], []
    for r in recs:
        rec = {"messages": [{"role": "user", "content": r["snapshot"]},
                            {"role": "assistant", "content": f"[[[{r['true']}]]]\n{r['body']}"}]}
        (pos if r["opus"] == r["true"] else neg).append((rec, r))
    # reject-sampled positives -> train/val (label-first, index split)
    train = [rec for j, (rec, _) in enumerate(pos) if j % 10]
    val = [rec for j, (rec, _) in enumerate(pos) if j % 10 == 0]
    for name, rows in (("train", train), ("val", val)):
        with (_OUT / f"{name}.jsonl").open("w") as f:
            for rec in rows:
                f.write(json.dumps(rec) + "\n")
    # negatives (Opus disagreed with the noisy label) for later DPO/analysis
    with (_OUT / "negatives.jsonl").open("w") as f:
        for rec, r in neg:
            f.write(json.dumps({"snapshot": r["snapshot"], "true": r["true"],
                                "opus": r["opus"], "body": r["body"]}) + "\n")
    agree = len(pos)
    print(f"[blind filter] {len(recs)} blind theses | Opus-agrees-label {agree} "
          f"({100*agree/max(len(recs),1):.0f}%) | negatives {len(neg)} | "
          f"-> train {len(train)} val {len(val)} @ {_OUT}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["predict", "filter"], required=True)
    ap.add_argument("--n", type=int, default=3047)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()
    if args.stage == "predict":
        predict(args.n, args.model, args.concurrency)
    else:
        filter_()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
