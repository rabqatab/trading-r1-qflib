"""Distillation v3 — Opus 4.8 teacher (via the Claude subscription, `claude -p`).

Reverse-reasoning, but TERSE + termination-safe (the fix for v2's failure: the long §8
style caused 9.2% NO_TAG non-termination + 2.6x drawdown). Opus is shown the ground-truth
label and writes a short evidence-grounded thesis that ends with exactly [[[LABEL]]].

Controlled comparison: we reuse the EXACT prompts of the template-SFT set
(`data_top150_mm/train.jsonl`) — same multimodal input, same label — and only swap the
templated rationale for an Opus thesis. So distill-SFT vs template-SFT isolates *teacher
reasoning vs template*, scored on the same 2025-H1 OOS eval + GBM ceiling (0.215).

Teacher = `claude -p --model opus` (subscription, no API $, no local GPU). Per-example
cache → resumable across rate-limit pauses. Low concurrency + backoff for rate limits.

    uv run python -m compare_lab.sft.distill_opus \
        --src compare_lab/sft/data_top150_mm --out compare_lab/sft/data_top150_distill \
        --n 3000 --concurrency 3
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

_TEACHER = (
    "You are a senior equity analyst. The correct volatility-adjusted call for the next "
    "week on this name is [[[{label}]]]. Write a SHORT thesis that justifies exactly this "
    "call using ONLY the data below. Rules:\n"
    "- 3 to 4 sentences, ≤110 words total. No headers, no preamble, no lists.\n"
    "- Ground each claim in a specific quoted value from the data (e.g. RSI 55.3, "
    "MACD 0.03, 50-day SMA 86.46, a headline).\n"
    "- The VERY LAST line must be exactly [[[{label}]]] and nothing else.\n\n"
    "Data:\n{snapshot}\n"
)


def _label_of(rec: dict) -> str | None:
    m = _TAG.findall(rec["messages"][-1]["content"])
    return m[-1] if m else None


def _snapshot_of(rec: dict) -> str:
    return rec["messages"][0]["content"]


def call_opus(prompt: str, model: str, timeout: int = 180) -> str:
    """One pure single-turn Opus completion via the subscription CLI (no tools)."""
    r = subprocess.run(
        ["claude", "-p", prompt, "--model", model, "--max-turns", "1",
         "--allowedTools", ""],
        capture_output=True, text=True, timeout=timeout,
    )
    if r.returncode != 0:
        raise RuntimeError(f"claude -p rc={r.returncode}: {r.stderr[:200]}")
    return r.stdout.strip()


def distill_one(rec: dict, model: str, cache: Path, retries: int = 3) -> dict | None:
    snap = _snapshot_of(rec)
    label = _label_of(rec)
    if label is None:
        return None
    key = sha1(snap.encode()).hexdigest()[:16]
    cpath = cache / f"{key}.json"
    if cpath.exists():                       # resume: already distilled
        return json.loads(cpath.read_text())
    prompt = _TEACHER.format(label=label, snapshot=snap)
    for attempt in range(retries):
        try:
            thesis = call_opus(prompt, model)
            tags = _TAG.findall(thesis)
            # termination-safe guard: must end with exactly the right tag
            if tags and tags[-1] == label and thesis.rstrip().endswith(f"[[[{label}]]]"):
                out = {"messages": [
                    {"role": "user", "content": snap},
                    {"role": "assistant", "content": thesis},
                ], "label": label}
                cpath.write_text(json.dumps(out))
                return out
        except Exception as e:
            wait = 20 * (attempt + 1)          # backoff (rate limits)
            if "rate" in str(e).lower() or "limit" in str(e).lower():
                wait = 120 * (attempt + 1)
            time.sleep(wait)
    return None                                # failed after retries → skip


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="compare_lab/sft/data_top150_mm")
    ap.add_argument("--out", default="compare_lab/sft/data_top150_distill")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--model", default="opus")
    ap.add_argument("--concurrency", type=int, default=3)
    args = ap.parse_args()

    src = Path(args.src); out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    cache = out / "_cache"; cache.mkdir(exist_ok=True)
    recs = [json.loads(l) for l in (src / "train.jsonl").open()][: args.n]
    print(f"[distill] {len(recs)} prompts | teacher={args.model} | conc={args.concurrency}",
          flush=True)

    done, fail = [], 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as ex:
        futs = {ex.submit(distill_one, r, args.model, cache): i for i, r in enumerate(recs)}
        for k, fut in enumerate(as_completed(futs), 1):
            res = fut.result()
            if res:
                done.append(res)
            else:
                fail += 1
            if k % 25 == 0 or k == len(recs):
                print(f"[distill] {k}/{len(recs)} | ok={len(done)} fail={fail}", flush=True)

    # split train/val by prompt hash (same scheme as the template set)
    train, val = [], []
    for r in done:
        h = int(sha1(r["messages"][0]["content"].encode()).hexdigest(), 16) % 10
        (val if h == 0 else train).append(r)
    for name, rows in (("train", train), ("val", val)):
        with (out / f"{name}.jsonl").open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
    print(f"[distill] DONE ok={len(done)} fail={fail} → train={len(train)} val={len(val)} @ {out}",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
