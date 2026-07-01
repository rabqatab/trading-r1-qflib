"""Compute IC (Spearman of predicted class-index vs forward signal) from preds jsonl.

Primary metric for the learning curve (#1) and the multimodal ceiling (#2). Runs in
the uv env (pandas only). Pass one or more labelled preds files:

    uv run python -m compare_lab.compute_ic \
        267=compare_lab/eval150/preds/mm_267.jsonl \
        1k=compare_lab/eval150/preds/mm_1000.jsonl ...
"""
from __future__ import annotations

import json
import sys

import numpy as np
import pandas as pd

CLASSES = ("STRONG_SELL", "SELL", "HOLD", "BUY", "STRONG_BUY")


def ic(path: str) -> dict:
    df = pd.DataFrame(json.loads(l) for l in open(path))
    n_all = len(df)
    valid = df.dropna(subset=["idx"]).copy()
    valid["idx"] = valid["idx"].astype(int)
    # Spearman = Pearson of average-ranks (handles the heavy ties in idx)
    ri = valid["idx"].rank(method="average")
    rs = valid["signal"].rank(method="average")
    r = float(np.corrcoef(ri, rs)[0, 1]) if valid["idx"].nunique() > 1 else float("nan")
    n = len(valid)
    se = 1.0 / np.sqrt(n - 3) if n > 3 else float("nan")
    dist = valid["idx"].map(dict(enumerate(CLASSES))).value_counts().to_dict()
    return {"IC": r, "n": n, "invalid": n_all - n, "SE": se, "dist": dist}


def main() -> int:
    args = sys.argv[1:]
    if not args:
        print(__doc__)
        return 1
    print(f"{'label':<14}{'IC':>8}{'±SE':>7}{'n':>6}{'inval':>6}  class-dist")
    print("-" * 72)
    for a in args:
        label, path = a.split("=", 1)
        r = ic(path)
        dist = " ".join(f"{k[:4]}:{v}" for k, v in sorted(r["dist"].items()))
        print(f"{label:<14}{r['IC']:>+8.3f}{r['SE']:>7.3f}{r['n']:>6}"
              f"{r['invalid']:>6}  {dist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
