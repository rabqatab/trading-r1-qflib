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

> **External anchor (Gu, Kelly & Xiu, RFS 2020):** the best ML on ~900 predictors achieves
> monthly OOS **R² ≈ 0.4 % ⇒ ρ ≈ 0.063** per stock — statistically identical to our raw-return
> **0.064**, with the same top signals (momentum/liquidity/vol). Our 0.06 is not a pipeline
> failure; it *is* the honestly-measured state of the art. (Their R² is monthly; our 3–15d
> horizon is noisier, so 0.06 is if anything generous — reinforcing that 0.27 is smoothing.)

### E7 — MEASURED information ceiling: inference → fact
Converts our biggest inferred claim ("0.06 = I(X;Y)") into a measurement, two independent ways:
| quantity | value (nats) |
|--|--:|
| KSG kNN **I(momentum; make_signal)** (Kraskov 2004; perm-null 0.0011) | **0.0344** |
| Gaussian identity **−½ln(1−IC²)** from momentum IC 0.266 (arXiv:2603.27074) | **0.0367** |
| same identity from **raw-return** IC 0.064 | **0.0021** |
| McAllester–Stratos cap on any distribution-free MI lower bound, O(ln N), N=36 420 | **10.5** |

The nonparametric KSG estimate (0.034) and the Gaussian-identity value (0.037) **agree** → the
momentum feature extracts essentially all of I(momentum; target); the model is **saturated**, not
under-powered. And the target's MI (~10⁻³–10⁻² nats) sits **3–4 orders of magnitude below** the
statistical estimation ceiling (10.5 nats) → the limit is **informational** (I(X;Y) is tiny), *not*
sample size. This is the DPI argument **measured**, no longer merely inferred.

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

**Now measured (was inference, E7 closes it):**
- **"0.06 = I(X;Y)"** is no longer only inferred. KSG MI (0.034 nats) and the Gaussian identity
  (0.037) agree, and both sit 3–4 orders below the O(ln N) estimation ceiling → the limit is
  informational and the best model saturates it. Caveat: KSG is a marginal single-feature estimate
  and the identity assumes Gaussian+log-loss; the *joint* I(X;Y) could be marginally higher, so read
  "≈ saturated" not "exactly saturated."

**Still our inference / honestly contested:**
- **The 0.06 floor is not *pure* noise.** Chen ([2206.15365](https://arxiv.org/abs/2206.15365))
  argues cross-sectional predictability is mostly *real-but-small* (empirical-Bayes), so 0.06 is
  **inflation + a sliver of genuine smoothed signal**, not inflation alone. Don't say "0.21 is pure
  noise."
- **Complexity-helps is a live, top-venue dispute.** Kelly–Malamud–Zhou (VoC, JF 2023) claim
  complexity genuinely helps market-*timing*; but Nagel (BFI 2025, "Seemingly Virtuous Complexity")
  shows their gain **reduces to vol-timed momentum** in small samples, and Buncic (SSRN 2025) traces
  it to a zero-intercept artifact. Net: the critiques *support* our "1-feature momentum is hard to
  beat," but frame our result as evidence *within* an unsettled debate, not a closed proof.
- **#2's null: probably informational, not proven by the LLM alone.** LLMs are known-weak text
  encoders (Merrill/Tan EMNLP 2024; Tan et al. NeurIPS 2024), so an LLM null can't discriminate
  "no info" from "not extracted." Finance evidence says headline text signal is transient (1–2 d),
  small-cap, momentum-redundant, cost-fragile (Tetlock 2007; Heston–Sinha 2017; Lopez-Lira–Tang
  2023 — unprofitable at ~20 bps) → an informational null is most parsimonious. **E8 (below)
  disambiguates.**
- Skeptic caveat on 2510.15990: it proves GRPO is **bounded by the base distribution** (a real
  theorem), *not* metaphysical "no new capability ever." Cite it for the reweighting bound only.

## Recommended next experiments (now concretely specified)
- **E7 (done):** KSG MI + Gaussian identity above. Optional hardening — Ishida Bayes-error
  (ICLR 2023, `github.com/takashiishida/irreducible`) on a binary up/down framing for a
  reviewer-friendly "irreducible error = X %, best model within Y %."
- **E8 (to run) — resolve the #2 null.** Add the alt-data as **numeric tabular features** to the
  price-only GBM (analyst-revision counts, insider net-buys, news count/recency, macro levels).
  Strengthen per agent-C: (1) also add a **return-fitted "oracle" sentiment feature** (SESTM-style,
  Ke–Kelly–Xiu NBER 2019) — if even *that* adds nothing, the informational verdict is near-airtight;
  (2) **stratify by market-cap / horizon (1–2 d vs our target) / net-of-cost** — residual text signal,
  if any, should appear only in the small-cap, 1–2-day, gross-of-cost cell, which simultaneously
  confirms the null *and* explains it.

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

## Extended reference index (LexiconArxiv-verified, 2026-07-03)

**Financial-ML predictability & overfitting (grounds the 0.06 floor / momentum dominance):**
| paper | venue | role |
|--|--|--|
| Gu, Kelly & Xiu, "Empirical Asset Pricing via ML" | RFS 2020 | **anchor**: best-ML monthly OOS R²≈0.4 % ⇒ ρ≈0.063 ≈ our 0.064; top signals = momentum/liq/vol |
| Jegadeesh & Titman, "Returns to Buying Winners…" | JF 1993 | momentum ~1 %/mo — the dominant, most-replicated anomaly (grounds E1) |
| Bailey & López de Prado, "Deflated Sharpe Ratio" | JPM 2014 | multiple-testing deflation → report #variants tried; high numbers suspect |
| Bailey et al., "Pseudo-Mathematics & Financial Charlatanism" | Notices AMS 2014 | Prob. of Backtest Overfitting → OOS discipline |
| Harvey, Liu & Zhu, "…and the Cross-Section of Expected Returns" | RFS 2016 | t>3 hurdle; factor zoo; momentum survives |
| Hou, Xue & Zhang, "Replicating Anomalies" | RFS 2020 | 65–82 % anomalies fail; markets more efficient than thought |
| Chen, "Most Claimed Findings … Likely True" | [2206.15365](https://arxiv.org/abs/2206.15365) | **dissent**: predictability real-but-small → 0.06 not pure noise |
| Kelly, Malamud & Zhou, "Virtue of Complexity" | JF 2023 | **tension**: complexity helps market-timing |
| Nagel, "Seemingly Virtuous Complexity"; Buncic (SSRN) | BFI 2025 | deflate VoC → it *is* vol-timed momentum (supports us) |

**Information-ceiling estimation (E7 methodology):**
| paper | venue | role |
|--|--|--|
| "Forecastability as an Information-Theoretic Limit" | [2603.27074](https://arxiv.org/abs/2603.27074) | **identity** I = −½ln(1−R²) nats — converts our IC → MI directly |
| Kraskov, Stögbauer, Grassberger, KSG MI | Phys.Rev.E 2004 | nonparametric MI (our E7 primary, via sklearn) |
| McAllester & Stratos, "Formal Limitations…MI" | AISTATS 2020 | O(ln N) cap on MI lower bounds — our target sits 3–4 orders below |
| Song & Ermon, SMILE; Poole et al., "Variational Bounds of MI" | ICLR/2019 | neural-MI bias/variance pitfalls (why we didn't lead with MINE) |
| Ishida et al., "…Estimating the Bayes Error" | ICLR 2023 | optional E7 hardening (binary irreducible-error) |

**Text / alt-data value (grounds #2, designs E8):**
| paper | venue | role |
|--|--|--|
| Tetlock, "Giving Content to Investor Sentiment" | JF 2007 | media sentiment → transient reversal, not durable info |
| Heston & Sinha, "News vs. Sentiment" | FAJ 2017 | daily news predicts only 1–2 d |
| Ke, Kelly & Xiu, "Predicting Returns With Text" (SESTM) | NBER 2019 | **counter**: return-fitted text extraction *does* add alpha → oracle feature for E8 |
| Lopez-Lira & Tang, "Can ChatGPT Forecast…" | [2304.07619](https://arxiv.org/abs/2304.07619) | LLM headline signal real but unprofitable at ~20 bps |
| Merrill/Tan; Tan et al. | EMNLP/NeurIPS 2024 | LLMs are weak text→forecast encoders (LLM null ≠ info null) |
| Barber et al.; Lakonishok & Lee | JF 2001 / RFS 2001 | analyst/insider signal gross-only, small-cap, dies net of cost |

See the distillation-methods survey ([`2026-07-03-distillation-v3-lit.md`](2026-07-03-distillation-v3-lit.md))
for the reasoning-distillation literature informing the live experiment.
