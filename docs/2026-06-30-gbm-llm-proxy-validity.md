# Is the GBM a valid proxy for the LLM? — rigorous multi-axis test (2026-06-30)

> ⚠️ **Correction (2026-07-03, [`why-the-ceiling.md`](2026-07-03-why-the-ceiling.md)):** this doc
> treats the GBM's ~0.24 as the *input signal ceiling*. That is now known to be too low — a linear
> **momentum** model reaches **0.266** on the same proxy, so the GBM itself under-extracts. The
> "GBM = upper bound" claim below holds only vs the *LLM variants tested here*, not as the true
> information ceiling. The tradeable (raw-return) ceiling is ≈0.06.

> Code: reproduced from cached LLM decisions (`compare_lab/.cache_*`) + a HistGBM on the
> same price/technical features. No LLM inference needed.

## Why this matters

We used a gradient-boosted tree (HistGBM) as a fast stand-in for the LLM to argue about the
**IC ceiling** and a **data learning-curve**. A reviewer rightly asked: the curve ran in
seconds — was it a GBM proxy, and is that proxy honest? This note tests it rigorously.

A proxy can be valid for one claim and invalid for another. We test five axes on 6 cached
LLM models (full OOS 2024-26, 14-eq) against the GBM.

## Results

| model | LLM IC | GBM IC | gap | corr(LLM,GBM) | decision-agree % | GBM right \| LLM wrong |
|---|--:|--:|--:|--:|--:|--:|
| SFT v1 | 0.127 | 0.213 | +0.087 | 0.56 | 32 % | 24 % |
| SFT v2 | 0.134 | 0.217 | +0.083 | 0.49 | 29 % | 26 % |
| GRPO matrix | 0.189 | 0.215 | +0.026 | 0.65 | 29 % | 16 % |
| graded | 0.190 | 0.217 | +0.027 | 0.69 | 31 % | 19 % |
| v1-reg GRPO | 0.183 | 0.218 | +0.035 | 0.68 | 29 % | 17 % |
| deeper GRPO | 0.157 | 0.202 | +0.045 | 0.65 | 30 % | 18 % |

GBM-full OOS IC = **0.242**; GBM trained on the LLM-SFT's data size (N=2286) = **0.194**.
(IC standard error at n≈1500 is ≈0.026.)

## Verdict per axis

| Axis | Valid proxy? | Evidence |
|---|---|---|
| **① Input ceiling / upper bound** | ✅ **VALID** | GBM IC ≥ every LLM IC (no exception); GBM saturates at ~0.24 = the max extractable from these features. |
| ② LLM IC point estimate | ❌ invalid | gap varies 3× (0.026–0.087) across models; SFT models sit significantly (~3 SE) below GBM. |
| ③ LLM decisions (behaviour) | ❌ invalid | corr 0.5–0.7, **decision agreement only ~30 %** (chance 20 %) — they make different calls 70 % of the time. |
| ④ Independence | — (disproves "proxy") | GBM is directionally right on **16–26 %** of the rows where the LLM is wrong → GBM carries *independent* signal; it is a *different* model, not a shadow of the LLM. |
| ⑤ Matched-data / learning curve | ❌ invalid | at the same N=2286 the GBM (0.194) still beats LLM-SFT (0.127): the LLM **under-extracts ~35 %**, and the gap shrinks with RL — a model-specific dynamic the GBM cannot predict. |

## Bottom line

**The GBM is an honest proxy for the *input's signal ceiling*, and a dishonest proxy for the
*LLM itself*.**

- ✅ Legitimate: "the price/technical input caps the extractable signal at IC ≈ 0.24, so no
  model — LLM included — breaks it by modelling alone." (GBM ≥ LLM holds for all 6 models.)
- ❌ Illegitimate: using the GBM to predict the LLM's IC, its decisions, which LLM variant is
  better, or **whether more training data will help the LLM**. The GBM learning curve
  (saturates ~5k) bounds the *signal*, not the *LLM's sample efficiency* — the LLM reads text,
  under-extracts even at matched data, and must be trained at several sizes to draw its *own*
  curve.

Nuance worth keeping: **RL narrows the under-extraction gap** — SFT extracts ~60 % of the GBM
ceiling, GRPO ~88 %. So the LLM's realistic headroom from "approach the ceiling" work is the
remaining ~12 % (≈ 0.19 → 0.21), not the full 0.24, and only the LLM curve can confirm it.
