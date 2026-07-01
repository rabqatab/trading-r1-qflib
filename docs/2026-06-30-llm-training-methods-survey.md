# LLM training methods for our task — literature survey & analysis (2026-06-30)

> Question: what training methods could improve *our* model — a Qwen3-4B emitting a 5-class
> trading signal — given the diagnosed bottlenecks? Sources: arXiv (2025–26) + venue-filtered
> published work via LexiconArxiv. Search done over three passes (2025, 2026, published).

## TL;DR

- **The ~0.24 IC ceiling is input-bound and now *theoretically* expected:** published/preprint
  work shows **GRPO reweights latent pretrained ability, it does not create new capability** — so
  RL cannot transcend a ceiling set by the input/pretraining. No method below claims to break it.
- **Two genuinely new directions that fit us (both = "use the LLM for what it's good at"):**
  1. **NOVER — verifier-free RL** (EMNLP 2025, code): sidesteps our broken *noisy verifier* (the
     matrix/label reward that fails the reward gate).
  2. **SCRL-LG — LLM as news *encoder*** feeding a tabular predictor (the published version of the
     hybrid our own analysis converged on; the LLM's edge is text, not numerics).
- **To merely *approach* the ceiling:** PRPO (RL that closes the LLM-vs-GBDT tabular gap) + a
  noisy-reward correction.
- **New risk surfaced & tested:** *pretraining contamination* — our LLM's pre-cutoff IC is partly
  recall, not prediction (see §4D).

## 1. Our bottlenecks (what the survey must address)

| # | Bottleneck | Evidence |
|---|---|---|
| A | LLM **under-extracts** tabular signal | LLM IC 0.13–0.19 vs GBM 0.24 on the same features |
| B | **Weak/noisy verifiable reward** | every model FAILS the reward gate (mean matrix reward < best-const); IC ≈ "barely better than random" |
| C | **Small data** | GRPO ~267 examples; collapses |
| D | **Pretraining contamination** | OOS 2024–26 overlaps Qwen3-2507's pretraining → possible recall |

## 2. Is the ceiling breakable? (theory says no — by modelling alone)

- **"Can GRPO Help LLMs Transcend Their Pretraining Origin?"** ([2510.15990](https://arxiv.org/pdf/2510.15990)).
  GRPO **reweights/sharpens latent capability; does not create new knowledge.** Where the base's
  input signal is weak, RL cannot compensate. ⇒ the *theoretical* reason our RL never beat 0.24.
- Corroborated empirically by our own [GBM-proxy test](2026-06-30-gbm-llm-proxy-validity.md): GBM IC
  ≥ every LLM IC, and saturates at ~0.24 regardless of data (so the *input*, not the model/RL/data
  quantity, is the cap). The break needs better **input** (article bodies, alt-data) or lower costs.

## 3. Methods, mapped to each bottleneck

### A. Closing the LLM-vs-GBDT tabular gap → **PRPO** ⭐ ([2510.17385](https://arxiv.org/abs/2510.17385))
RL post-training that makes **Qwen3-8B match GBDTs** on 139 OpenML tabular datasets. Two mechanisms,
both ours to reuse:
1. **Column-permutation invariance** — m=4 random feature-order permutations per example, reward
   consistency. Teaches the order-invariance trees have natively + free augmentation (267×4 views).
2. **Two-level advantage** (densifies sparse reward): intra-perm `Â¹=(R−μ_k)/σ_k` + inter-perm
   `Â²=(R−μ_global)/σ_global`, final `Â=α·Â¹+(1−α)·Â²`, **α=0.1** (90 % global), GRPO/PPO-clip.
   Reward: 1.0 correct / 0.1 valid-wrong / 0.0 malformed. Settings: Qwen3-8B, G=5, m=4, lr 1e-6,
   β 0.001, 30 epochs.
- Crossover context: **GBDT beats LLM beyond ~8 shots** ([2411.04324](https://arxiv.org/abs/2411.04324));
  at our data size pure-tabular LLM *should* lose — so PRPO's job is parity (≈0.24), not a break.

### B. Weak/noisy verifier (our reward-gate failure)
- **NOVER — verifier-free RL** ⭐ (EMNLP 2025, tier-0, [code](https://github.com/thinkwee/nover);
  `DOI 10.18653/v1/2025.emnlp-main.378`). Computes reward from the answer using only SFT data, **no
  external verifier**; beats same-size R1-671B distillation by 7.7 %. *Directly* sidesteps the noisy
  matrix/label reward we're stuck optimizing. Highest-value new direction.
- **An Imperfect Verifier is Good Enough** ([2604.07666](https://arxiv.org/pdf/2604.07666), 2026).
  Estimate per-verifier FPR/FNR, adaptively down-weight uncertain examples, iteratively refine.
  Robust at 30–50 % error / barely-better-than-random verifiers = our regime.
- **Noise-corrected GRPO** ([2510.18924](https://arxiv.org/abs/2510.18924)). Model reward as
  Bernoulli flips; correct: `r̂=(r̃−ρ⁺)/(1−ρ⁺−ρ⁻)`, `Â=r̂−mean(r̂)` → unbiased gradient. Needs
  ~1,500 balanced calibration examples; assumes binary reward (adapt: binarize "right direction").
- **VRPO** ([2508.03058](https://arxiv.org/abs/2508.03058)). Add a value model to absorb unstable
  signal → fixes GRPO collapse under weak supervision (our all-SELL/all-StrongBuy collapses).
- Reward densification (all ICLR 2026): **HERO/Hybrid-RL** (verifier + reward-model),
  **Curriculum easy→hard RLVR**, **RLBFF** (binary flexible feedback).

### C. Small data
- PRPO's permutation trick = built-in augmentation (×m).
- **TS foundation-model pretraining** ([2507.07296](https://arxiv.org/abs/2507.07296)): 3–10× less
  data for equal performance.
- **Retrieval augmentation** (FinSeer, [2502.05878](https://arxiv.org/abs/2502.05878)): retrieve
  similar historical patterns for small-data stock movement.

### D. Pretraining contamination — surfaced *and tested*
- **"Profit Mirage"** ([2510.07920](https://arxiv.org/pdf/2510.07920)): LLM financial agents leak via
  lookahead, event memorization, **and pretraining contamination**; reported returns "dramatically
  inflated." Our OOS (2024–26) overlaps Qwen3-4B-Instruct-**2507** (cutoff ~2025-07).
- **Our test (GBM as contamination-free control, CUT=2025-08-01):**
  - **graded GRPO:** LLM IC 0.203→0.150 while the GBM control held 0.201→0.210 → **excess drop
    +0.062 ⇒ contamination suspected** (recall inflated the pre-cutoff IC). Fits "GRPO amplifies
    *pretrained* memory" (§2). graded's honest IC ≈ post-cutoff **0.15**, not 0.19.
  - **SFT v1:** 0.126→0.124, no excess → clean; its 0.13 is trustworthy.
  - Caveat: post n≈430 (~1.3 SE) → suggestive, not conclusive; firm up on fresh post-cutoff data.
  - Implication: the GBM (cutoff-stable 0.21) is the cleanest predictor; report LLM IC on the
    post-cutoff slice to be contamination-safe.

## 4. The LLM's *right* role — hybrid, text not numerics

- **SCRL-LG** ([2310.05627](https://arxiv.org/abs/2310.05627)): LLM = **news-headline feature
  encoder**, embeddings aligned with stock features, a separate RL/Local-Global model predicts. The
  published version of the hybrid our analysis reached — the LLM's value-add is *news encoding*
  (which a GBM can't do), not numeric decision-making.
- **FLAG-Trader** (ACL 2025 Findings, tier-1, **same author group as Trading-R1**;
  `DOI 10.18653/v1/2025.findings-acl.716`): a partially-PEFT LLM **is** the RL policy, trained by
  policy gradient on trading rewards. Confirms our SFT→GRPO design is legitimate/publishable — but
  makes no signal-ceiling-breaking claim.

## 5. Recommended priority (expanded universe ARRIVED 2026-07-01)

> **Status:** the top-150 universe landed 2026-07-01. Before the methods below, we are
> first drawing the **real LLM data-scale learning curve** (267/1k/3k template-SFT) and
> re-testing the **multimodal ceiling** at 150 equities — see
> [`2026-07-01-top150-learning-curve-experiment.md`](2026-07-01-top150-learning-curve-experiment.md).
> That baseline decides which lever below is worth pulling: a below-ceiling plateau ⇒
> distillation / verifier-free reward (A/B); a collapse-repair-only gain ⇒ VRPO (C).


1. **Hybrid (SCRL-LG-style):** LLM news-encoder → embeddings/features → GBM predictor. Plays each
   tool to its strength; addresses bottleneck A by not fighting the LLM-on-tabular battle.
2. **Verifier-free RL (NOVER):** drop the noisy matrix/label reward that fails the gate; incentive-
   train from SFT data. Addresses bottleneck B at its root.
3. **PRPO** (permutation augmentation + two-level advantage) to push the *pure-LLM* path to GBDT
   parity (~0.24) and fix small-data collapse — the "approach the ceiling" option.
4. Stack a noisy-reward correction (Imperfect-Verifier / noise-corrected advantage) + VRPO value
   model if collapse persists.

All four lift the LLM *to* the ceiling or use it better; none break the input ceiling — that still
needs richer input (article bodies, alt-data) or lower transaction cost.

## 6. Reference index

| Paper | Venue / id | Relevance |
|---|---|---|
| Can GRPO Transcend Pretraining Origin? | [2510.15990](https://arxiv.org/pdf/2510.15990) | theory: RL can't break input/pretraining ceiling |
| PRPO (numerical reasoning, tabular) | [2510.17385](https://arxiv.org/abs/2510.17385) | close LLM↔GBDT gap; permutation aug + 2-level advantage |
| NOVER (verifier-free RL) | EMNLP 2025 · [code](https://github.com/thinkwee/nover) | sidestep our noisy verifier |
| Imperfect Verifier / Noisy Rewards | [2604.07666](https://arxiv.org/pdf/2604.07666) | robust RL at weak/noisy verifier |
| Noise-corrected GRPO | [2510.18924](https://arxiv.org/abs/2510.18924) | unbiased gradient under reward noise |
| VRPO | [2508.03058](https://arxiv.org/abs/2508.03058) | value model fixes weak-signal GRPO collapse |
| Profit Mirage (leakage) | [2510.07920](https://arxiv.org/pdf/2510.07920) | pretraining-contamination risk |
| SCRL-LG (LLM news encoder) | [2310.05627](https://arxiv.org/abs/2310.05627) | hybrid: LLM=text, model=predict |
| FLAG-Trader | ACL 2025 Findings (Trading-R1 group) | precedent for LLM-as-policy + RL |
| GBDT vs LLM few-shot | [2411.04324](https://arxiv.org/abs/2411.04324) | crossover ~8 shots; GBDT wins at scale |
| TS foundation pretraining | [2507.07296](https://arxiv.org/abs/2507.07296) | 3–10× less data |
| FinSeer (RAG) | [2502.05878](https://arxiv.org/abs/2502.05878) | retrieval for small-data stock movement |
| HERO / Curriculum-RLVR / RLBFF | ICLR 2026 | sparse→dense reward design |
