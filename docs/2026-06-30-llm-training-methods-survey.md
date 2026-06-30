# Better LLM training methods for our task — arXiv survey (2026-06-30)

> Goal: find training methods that fit *our* diagnosed bottlenecks — LLM under-extracts
> numerics vs a GBM (IC 0.19 vs 0.24), weak/noisy verifiable reward (reward gate fails),
> small data (GRPO ~267), and a strong non-LLM baseline (GBM). LexiconArxiv was down; pulled
> via arXiv web search + paper reads.

## Honest framing first

None of these break the input ceiling (IC ~0.24 is input-bound — proven by GBM-on-31k and
the [proxy-validity test](2026-06-30-gbm-llm-proxy-validity.md)). They are **"approach the
ceiling" methods**: they target the *LLM's* ~0.05 under-extraction headroom (SFT extracts
60 % of the GBM ceiling, GRPO 88 %) and the small-data / weak-reward instability — i.e. they
could give us the **best-possible LLM (≈ GBM 0.24)**, not a break past it. A real break still
needs better *input* (article bodies, alt-data) or lower costs.

## #1 — PRPO: Permutation Relative Policy Optimization ([2510.17385](https://arxiv.org/abs/2510.17385)) ⭐ best fit

The closest paper to our exact problem: an RL post-training method that makes **Qwen3-8B match
GBDTs on tabular prediction** across 139 OpenML datasets (and beats DeepSeek-R1-685B by 53 %).
Two mechanisms, both directly useful to us:

1. **Column-permutation invariance as a structural prior.** Per example, generate `m=4` random
   permutations of the feature columns (our 12 indicators), serialize each, and reward
   consistency. Trees are natively feature-order-invariant; LLMs aren't — this *teaches* the
   invariance and is also free data augmentation (267 examples × 4 = ~1,068 views — big for
   our small set).
2. **Two-level advantage** (densifies our weak/sparse reward):
   - intra-permutation: `Â¹ = (R − μ_k)/σ_k` (z-score within a permutation's G rollouts)
   - inter-permutation: `Â² = (R − μ_global)/σ_global` (z-score across all m×G)
   - final: `Â = α·Â¹ + (1−α)·Â²`, **α=0.1** (90 % global).
   - GRPO/PPO-clip objective with these advantages.

Reward (Eq. 7, classification): 1.0 correct · 0.1 valid-but-wrong · 0.0 malformed. Settings:
Qwen3-8B, G=5 rollouts, m=4 perms, lr 1e-6, β(KL) 0.001, 30 epochs.

**Adaptation for us:** serialize the indicators as a permutable feature block, m=4 perms,
two-level advantage on top of our graded/matrix reward, our Qwen3-4B base. Expected: lift
0.19 → ~0.24 (close the GBDT gap) + stabilize the small-data GRPO that collapsed before.

## #2 — Noise-corrected GRPO ([2510.18924](https://arxiv.org/abs/2510.18924)) — for our noisy reward

Our labels (vol-adjusted forward returns) are a *noisy proxy* → the reward is noisy → the
advantage signal is weakened (cf. [2603.16140](https://arxiv.org/pdf/2603.16140), "noisy data
is destructive to RLVR"). This method debiases it:

- Estimate flip rates ρ⁺ (false-positive), ρ⁻ (false-negative) on a ~20 % balanced calibration
  split (needs ~1,500 balanced examples for ±0.1 at 95 %).
- Corrected reward: `r̂ = (r̃ − ρ⁺)/(1 − ρ⁺ − ρ⁻)`; debiased advantage `Â = r̂ − mean(r̂)`.
- Dr.GRPO variant (M=1) → provably unbiased gradient. Gains up to +6.7 pp (math) even at
  ρ⁺+ρ⁻ ≈ 0.6.

**Caveat:** assumes binary reward r*∈{0,1}; our matrix/graded reward is continuous, and our
calibration data is thin — so this needs adapting (e.g. binarize "right direction" for the
flip-rate estimate) before it applies cleanly. Principle is sound: debias the noisy-label
reward instead of trusting it raw.

## #3 — VRPO: value model for weak-signal stability ([2508.03058](https://arxiv.org/abs/2508.03058))

"GRPO suffers significant collapse without a value model on weaker models / noisy supervision."
We *saw* that collapse (all-SELL / all-StrongBuy). Adding a learned value model to absorb the
unstable signal gives more reliable advantage estimation. Candidate fix for our collapse modes
that's orthogonal to PRPO.

## #4 — Literature confirms our GBM>LLM finding + points the LLM's real job

- **GBDT vs LLM few-shot ([2411.04324](https://arxiv.org/abs/2411.04324)):** LLM wins only at
  ≤8 shots; beyond that GBDT wins on tabular. At our data size, a pure-tabular LLM *should*
  lose to the GBM — exactly what we see. ⇒ the LLM's value-add must come from **text** (news),
  which the GBM can't read; pure-numeric LLM training is the wrong battle.
- **Sample-efficiency levers for small data:** TS foundation-model pretraining needs 3–10× less
  data ([2507.07296](https://arxiv.org/abs/2507.07296)); retrieval-augmentation (FinSeer,
  [2502.05878](https://arxiv.org/abs/2502.05878)) helps small-data stock movement; "TS
  forecasting as reasoning" with reinforced LLMs ([2506.10630](https://arxiv.org/pdf/2506.10630)).

## Recommended plan (when the expanded universe lands)

1. **PRPO-style GRPO** (permutation augmentation + two-level advantage) on the bigger universe —
   the single highest-leverage method; should finally make the LLM *match* the GBM (0.24) and
   fix small-data collapse. Resolves the "does data + better RL close the gap" question.
2. Stack **noise-corrected advantage** (binarized-direction flip-rate) + optionally a **VRPO
   value model** if collapse persists.
3. Keep the LLM's *unique* job = reading **news text** (GBDT can't); pure-tabular parity with
   the GBM is the floor, not the goal.

Bottom line: these lift the LLM to the ceiling, not past it. The break is still input/data.
