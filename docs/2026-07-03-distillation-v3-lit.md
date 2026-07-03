# Reasoning-distillation literature for distillation v3 (2026-07-03)

> Informs the live Opus-4.8 reverse-reasoning experiment. Verified via LexiconArxiv + WebSearch.
> Setup: Opus 4.8 is shown the ground-truth 5-class label and writes a ≤110-word evidence-grounded
> thesis ending `[[[LABEL]]]`; Qwen3-4B LoRA learns (snapshot → thesis) with the label hidden.

## What prior work predicts for v3

Cautiously positive but **conditional on filtering**. Terse grounded rationales *can* beat
label-only SFT (even for classification) and *can* beat the base — but every positive result
depends on **filtering** and **many-sample diversity**, which our v3 currently lacks. Against a
base that already ≈ the input ceiling with a **noisy** label, the capacity-gap and noisy-label
literature predicts unfiltered answer-conditioned SFT will **at best match the base**, and at worst
teach *fluent justification of label noise*. Expected outcome: terse Opus reverse-reasoning
**matches-to-slightly-beats base and beats template-SFT — only with filtering** — with little IC
"breakthrough" (consistent with [[why-the-ceiling]]: the ceiling is input-bound).

## Key papers → implication

| paper | venue | implication for v3 |
|--|--|--|
| **Trading-R1** (Xiao et al.) | [2509.11420](https://arxiv.org/abs/2509.11420) | our exact recipe; but SFT is a **cold-start**, the alpha comes after **GRPO** + easy→hard curriculum — don't judge distillation on SFT alone |
| **STaR** (Zelikman et al.) | NeurIPS 2022 | rationalization works *with a filter*: keep only rationales that, once written, **re-derive the answer**. v3 keeps every thesis → add this |
| **Distilling Step-by-Step** (Hsieh et al.) | ACL-F 2023 | rationales-as-auxiliary beat label-only **on classification** with less data — strongest positive; but trains rationale as a *side task*, not a mandatory prefix |
| **SCoTD** (Li et al.) | ACL 2023 | **sample k≥5 rationales/instance**; diversity beats teacher-likelihood — single highest-leverage change |
| **CoT-Augmented Distillation** (Wadhwa et al.) | EMNLP 2024 | ⭐ rationale **after** the label distills *better*; no test-time student reasoning needed; permuted CoT still helps → **label-first ordering kills pre-decision non-termination** (our v2 failure) while keeping the regularization benefit |
| **Capacity gap** | [2604.08880](https://arxiv.org/abs/2604.08880) | Opus 4.8→Qwen3-4B is a large gap; terse+grounded theses *lower* task complexity, long fluent ones the 4B can't mimic *raise* KD error |
| **SCOTT: self-consistent CoT distillation** (Wang et al.) | ACL 2023 | antidote to our core risk: answer-conditioned rationales on a noisy label are often *unfaithful*; fix via contrastive-decoding elicitation + a **counterfactual objective** (student can't ignore the thesis) |
| **CoT Correctness-Perception** | [2509.05602](https://arxiv.org/abs/2509.05602) | filter/down-weight **low-confidence labels** — coarse supervision overfits spurious reasoning |
| **Learning from Noisy Labels w/ Distillation** | ICCV 2017 | anchor on any **clean/high-confidence subset**; treat the noisy bulk as auxiliary |
| **Fin-R1** | [2503.16252](https://arxiv.org/abs/2503.16252) | winning finance recipe = **distill → filter (LLM-judge) → GRPO**; filtering not optional |
| **Skip-Thinking / short-CoT** | [2505.18642](https://arxiv.org/abs/2505.18642) | mechanistic reason for v2's regression: long rationales dilute the decision gradient → terseness is literature-endorsed, not a hack |

## Three tweaks the literature most supports (for the next distillation cut)

1. **Filter by teacher self-consistency** (STaR + SCoTD + Fin-R1 + SCOTT): sample **k≥5** Opus
   theses/snapshot; keep only those whose thesis, re-read **without** the label, re-derives the same
   label (cheap LLM-judge or a held-out reader). Down-weight low-confidence labels. Highest leverage;
   directly attacks the noisy-justification risk our QC (grounding/faithfulness) only partially covers.
2. **Reorder label-first, or dual-task the rationale** (Wadhwa + Distilling Step-by-Step): emit
   `[[[LABEL]]]` *then* the thesis (eliminates pre-decision non-termination by construction, keeps the
   soft-label regularization), or train the thesis as an auxiliary head. A/B against thesis-first.
3. **Evidence + length reward in the GRPO stage** (SCOTT counterfactual + JET/GFPO length control):
   treat SFT as cold-start; reward snapshot-grounded theses, penalize length + non-termination. Per
   Trading-R1 / Fin-R1 the real gain is in RL, not SFT.

## Caveat
Even a "working" v3 must be judged on **IC / decision accuracy**, not bull-window return — finance-RL
papers (FLAG-Trader; Fino1) rarely report IC and their headline returns are bull-window (echoing
[[graded-reward-breakthrough]]). And per [[why-the-ceiling]], no distillation moves the *real-return*
ceiling (~0.06); the honest target is the **proxy under-extraction gap** (template 0.163 → base 0.205
→ momentum 0.266) and format/parse quality.
