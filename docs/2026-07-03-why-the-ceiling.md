# Why the IC "ceiling" can't be broken — experiments + literature (2026-07-03)

> A rigorous, self-correcting investigation. It **overturns two of our own earlier claims**
> (that ~0.21 is the input ceiling, and that the LLM "captures 76 % of it"). Experiments:
> `compare_lab/ceiling_probe.py` (+ `gbm_ceiling.py`). Literature verified via WebSearch
> (LexiconArxiv was down); IDs confirmed on arxiv.org.

## TL;DR — there are TWO ceilings, and we had been measuring the wrong one

1. **Proxy ceiling ≈ 0.27.** Against our `make_signal` target, a **one-feature linear momentum
   model gets IC 0.266** and ridge 0.249 — *above* GBM (0.215) and the base LLM (0.205). So
   0.21 was **never the input ceiling**; it was GBM/LLM **under-extracting** a momentum signal
   that a trivial linear model captures better.
2. **Real-return ceiling ≈ 0.06.** Re-score the *same* momentum predictor against the **raw
   7-day forward return** instead of the smoothed proxy: IC collapses **0.266 → 0.064** (GBM
   0.042). That ~0.06 is the genuine, tradeable, weak-form-EMH floor.

So "why can't the LLM break the ceiling?" splits cleanly:
- On the **proxy**, it *can* be beaten — the LLM/GBM just under-fit momentum (a *model* gap).
- On **real returns**, ~0.06 is the **information limit** I(price-features; return): by the
  data-processing inequality no model — LLM, GBM, SFT, GRPO, distillation — can exceed it.
- The gap between them (0.27 vs 0.06, ~4×) is **entirely a smoothing artifact** of `make_signal`.

## The experiments (`ceiling_probe.py`, same 2024-train / 1,000-pt 2025-H1 OOS)

### E1 — model-invariance? NO: the spread is 5× → the "ceiling" was model-specific
| model | OOS IC |
|--|--:|
| momentum (`close_10_roc`, 1 feature) | **+0.266** |
| Ridge (linear, 16 feat) | +0.249 |
| RandomForest | +0.229 |
| GBM (HistGBDT) | +0.215 |
| base LLM (prompt-only) | +0.205 |
| kNN (k=100) | +0.176 |
| MLP (64,32) | +0.049 (under-fit) |

Linear/momentum **beats** the flexible tree and the LLM. Adding the other 15 features to the
linear model *lowers* it (0.266→0.249) → the label is **momentum-dominated**; extra features
are mostly noise. The LLM "over-reasons" over many weak features instead of applying the one
rule that works.

### E2 — residual audit: what GBM misses is noise, not signal
A 2nd model trained to predict GBM's residual scores **IC −0.082** on the OOS residual (no
positive structure). What the tree leaves behind is **irreducible**, not recoverable signal —
consistent with a Bayes-error floor (below), though note momentum *does* beat GBM, i.e. the
tree's *inductive bias* (not the information) is what caps it here.

### E3 — Bayes / label-noise: the label is NOT a function of the features
| quantity | value | meaning |
|--|--:|--|
| GBM **in-sample** (train, 36k rows) IC | +0.395 | even *memorizing* the training set caps here |
| GBM in-sample **R²** | **0.161** | features explain only ~16 % of `make_signal` variance |
| GBM OOS IC | +0.215 | the 0.395→0.215 gap is generalization/overfit |

If the label were a deterministic function of the features, memorization would drive train IC→1.
It plateaus at 0.40 / R² 0.16 ⇒ **~84 % of the target is noise w.r.t. these inputs.**

### E6 — smoothing inflation: the proxy is ~4× easier than real returns
| predictor → target | IC |
|--|--:|
| momentum → `make_signal` proxy | +0.266 |
| momentum → **raw 7-day return** | **+0.064** |
| GBM → raw 7-day return | +0.042 |

`make_signal` is a forward **EMA(span 3)** return, vol-adjusted, blended over **overlapping**
horizons {3,7,15d}. Smoothing + overlap **mechanically inject autocorrelation** (Lo–MacKinlay
1988), inflating apparent predictability ~4×. **Everything we've reported as IC 0.16–0.27 is the
predictability of a smoothed artifact, not tradeable skill.** Real tradeable IC here is ~0.06.

## Literature grounding

The ceiling is **I(X; Y)** — the mutual information between inputs and target. Four mechanisms:

| mechanism | key refs | grounds |
|--|--|--|
| **Data-processing inequality**: no map of X raises I(X;Y); Fano/Feder–Merhav tie MI to an irreducible Bayes-error floor | Cover & Thomas 2006 (Thm 2.8.1); Fano ([1901.00555](https://arxiv.org/pdf/1901.00555)); Feder–Merhav 1994 IEEE-IT | the cap is **model-invariant** (tree ≈ base-LLM ≈ SFT ≈ GRPO); there is a hard floor no optimizer crosses |
| **Weak-form EMH**: short-horizon returns near-unpredictable from public info; real IC 0.02–0.10 | Fama 1970 & 1991 (J. Finance); Grinold 1989 / Grinold–Kahn | why I(X;Y) is *small*; why our **raw-return 0.06** is the real floor; why **news/fundamentals/sentiment add nothing** (already public) |
| **Overlapping/smoothed returns inflate predictability** | Lo–MacKinlay 1988 (RFS); JRFM 2025 19(1):46 | why the **proxy** ceiling is 0.27 not 0.06 — grounds **E6** directly |
| **RL/LLM can't add information beyond the base** | Yue et al. [2504.13837](https://arxiv.org/abs/2504.13837) (NeurIPS 2025); Ni et al. [2510.15990](https://arxiv.org/abs/2510.15990); Tan et al. [2406.16964](https://arxiv.org/abs/2406.16964) (NeurIPS 2024, ablating the LLM doesn't hurt TS forecasting); Merrill et al. [2404.11757](https://arxiv.org/abs/2404.11757) (text adds ~no forecast lift) | why **base LLM ≈ GBM**, **GRPO ≤ base**, and **multimodal text = null** (#2) |
| **Contamination inflates apparent LLM edge** | Profit Mirage [2510.07920](https://arxiv.org/abs/2510.07920) | validates our post-cutoff OOS design; any "edge" past the cutoff is suspect |

### LexiconArxiv-verified additions & one honest tension (2026-07-03)
Once the LexiconArxiv retrieval bug was fixed we confirmed the core TS papers are in-corpus and
found two more directly on point — plus a genuine counter-case we must not hide:
- ✅ **Tan et al., "Are Language Models Actually Useful for Time Series Forecasting?"** — corpus
  lists it as **NeurIPS 2024 *spotlight*** (top hybrid hit, score 0.85). Ablating/removing the LLM
  doesn't degrade forecasting → grounds **base-LLM ≈ GBM** (the LLM's language pretraining adds ~0
  to numeric extraction).
- 🆕 **"Context parroting: a simple but tough-to-beat baseline for foundation models"** (ICLR 2026)
  — a trivial baseline matches/beats large foundation models on scientific-ML sequence tasks.
  Independent support for **E1** (our 1-feature momentum > GBM > LLM): flexible capacity is not the
  bottleneck; the right simple inductive bias is.
- 🆕 **"Context is Key: A Benchmark for Forecasting with Essential Textual Information"** (ICML 2025)
  — measures when textual context *does* help forecasting; relevant to scoping **#2** (our news/
  alt-data null) — it helps only when text carries info not in the numbers, which public headlines
  mostly don't.
- ⚖️ **TENSION — Kelly, Malamud & Zhou, "The Virtue of Complexity in Return Prediction"**
  (*J. Finance* 2023, DOI 10.1111/jofi.13298): they *prove and document* that **complex** models
  (parameters ≫ observations) reveal **more** return predictability than simple ones — the opposite
  of our E1 "linear momentum beats GBM/LLM". Reconciliation (not dismissal): their target is the
  **aggregate market-return time series** with random-feature ridge and heavy shrinkage; ours is
  **cross-sectional single-name** prediction of a *smoothed vol-adjusted proxy*, where the signal is
  dominated by one autocorrelation (momentum) and extra capacity mostly fits noise (our in-sample
  R²=0.16). So "simple wins" is **task-specific here**, and complexity-helps remains live for the
  market-timing target. This is the one place our claim is contested by top-venue evidence.

## Honest scope — proven vs our inference

**Proven / strongly empirical (literature):** DPI as a model-invariant cap; MI↔Bayes-error floor;
weak-form EMH and the 0.02–0.10 real-IC scale; smoothing/overlap inflate predictability;
RL reweights (doesn't expand) the base; LLMs don't beat simple TS baselines and text adds ~no lift.

**Our inference (literature-consistent, not directly proven):**
- That our specific **0.06 (raw) / 0.27 (proxy) equals I(X;Y)**. No paper measures I(X;Y) for
  `make_signal`. The DPI/EMH/smoothing chain explains the *shape*; the exact value being the
  information limit is inferred. **To convert to proof:** estimate the Bayes-error/MI ceiling
  directly (classifier-confusion MI, [1606.05229](https://arxiv.org/pdf/1606.05229); or MINE/kNN)
  and show the best models *saturate* it. → **E7, not yet run.**
- That **#2's multimodal null = an information ceiling**. It could instead be LLM
  *encoding/alignment* failure (text info present but not extracted). **To disambiguate:** add
  the alt-data as **numeric tabular features** to the GBM; if tabular alt-data *also* adds no OOS
  IC over price-only, the null is informational, not an encoding artifact. → **E8, not yet run.**
- Skeptic caveat on 2510.15990: it proves GRPO is **bounded by the base distribution** (a real
  theorem), *not* metaphysical "no new capability ever." Cite it for the reweighting bound only.

## Corrections to earlier docs (recorded, not hidden)
- **`gbm_ceiling.md` / learning-curve doc "GBM 0.215 = input ceiling, LLM captures 76 %":**
  wrong. Linear momentum reaches **0.266** on the proxy → GBM and the LLM both *under-extract*;
  "76 %" understated the achievable and mis-attributed the gap to the LLM alone.
- **"~0.24 input-bound ceiling" (multiple docs):** it is the ceiling *of GBM on the smoothed
  proxy*, not of the input or of tradeable return. The tradeable ceiling is **~0.06**.

## Implications
- The project's real predictive signal is **IC ≈ 0.06** (raw return) — near the EMH floor;
  distillation / better rewards **cannot** move the *real-return* ceiling (DPI), though they may
  still close the **proxy** under-extraction gap (0.21→0.27) if that is the chosen target.
- Reporting should quote **raw-return IC** alongside proxy IC, or the numbers read ~4× too good.
- Remaining levers that could raise I(X;Y) itself (not just extraction): genuinely **non-public
  / alt-data** inputs, or **longer horizons** (lower noise) at the cost of transaction drag.
