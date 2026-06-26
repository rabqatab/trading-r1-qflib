# Trading-R1 × qf-lib — 3-Way Signal Comparison (Memo)

**Date:** 2026-06-21 · **Author:** compare_lab · **Status:** first LLM row landed
**One line:** A prompt-only 4B LLM, given only price+technical data and no
training, produces a *defensive, differentiated* equity signal — lowest return
but lowest drawdown and the least correlation to the quant baselines.

---

## What was run

Three signal sources, each emitting the **same** object — a target-weight
matrix `[date × ticker]` — scored on **one** look-ahead-safe qf-lib event
backtest.

| Knob | Value |
|---|---|
| Universe | 12 large-caps (NVDA MSFT AAPL META AMZN TSLA BRK-B JPM LLY JNJ XOM CVX); SPY/QQQ added since → 14-ticker robustness run below |
| Out-of-sample | 2024-01-02 → 2026-04-01 (no training anywhere, so the whole span is OOS) |
| Rebalance | weekly (W-FRI), next-bar execution, IB commissions |
| Sizing | long-only, fixed 12.5% per held name (budget = 8 positions) |
| LLM | `Qwen3-4B-Instruct-2507` (served as `Qwen/Qwen3-4B`), BF16, temp 0, vLLM on DGX Spark |

**Reproducibility:** the LLM decisions are disk-cached by snapshot hash. A 12-sample
re-query against the live model reproduced **12/12** cached decisions (temp-0
greedy), confirming the numbers below are attributable to this exact model.

---

## Results (OOS)

| Strategy | Cumulative | Sharpe | Max Drawdown | Up-day rate† |
|---|---|---|---|---|
| **Equal-weight** (market proxy) | **+126.4 %** | **1.07** | 27.8 % | 0.382 |
| 12-1 Momentum (top-5) | +49.9 % | 0.66 | 19.6 % | 0.382 |
| **Prompt-only LLM** (4B) | +42.8 % | 0.71 | **14.8 %** | 0.356 |

† Up-day rate is computed over a *calendar-day* return series (weekends enter as
0), so it reads low (~0.38) and is weakly discriminative — read **CR / Sharpe /
MDD** as the real metrics. (The two baselines tying at 0.382 is a coincidence of
their 0.90 return correlation, not a bug.)

### The finding that matters: the LLM signal is *differentiated*

Pairwise daily-return correlation:

|  | EW | Mom | LLM |
|---|---|---|---|
| EW | 1.00 | 0.90 | **0.69** |
| Mom | 0.90 | 1.00 | **0.63** |
| LLM | 0.69 | 0.63 | 1.00 |

The two quant baselines are near-twins (0.90). The LLM is materially less
correlated to both (0.63–0.69) — it is **not** a noisy restatement of momentum.
That orthogonality, plus the lowest drawdown in the set, is exactly the property
you want from an alternative reasoning-based signal, even when its raw return
lags in a strong bull market.

### Why the LLM lags on return

It is structurally **long-biased but capped**. Decision distribution over all
1,404 (ticker, day) calls:

| STRONG_BUY | BUY | HOLD | SELL | STRONG_SELL | NO_TAG |
|---|---|---|---|---|---|
| 51.1 % | 6.6 % | 21.0 % | 10.3 % | 2.8 % | **8.2 %** |

58 % of calls are bullish, but the 8-position budget caps how much of that
conviction can be expressed — so in a melt-up it under-participates vs holding
all 12 names equally. The **8.2 % NO_TAG** rate (replies with no parseable
`[[[CLASS]]]`, forced to HOLD) is a real quality leak to fix.

---

## Robustness — adding SPY/QQQ (14-ticker)

Re-running on the full 14-ticker universe (SPY/QQQ now in the dataset):

| Strategy | Cumulative | Sharpe | Max DD |
|---|---|---|---|
| Equal-weight | +143 % | 1.04 | 32.3 % |
| 12-1 Momentum | +52 % | 0.70 | 19.6 % |
| Prompt-only LLM | +36 % | **0.55** | **16.4 %** |

Adding the two ETFs **lowers** the LLM's Sharpe (0.71 → 0.55) while it keeps the
lowest drawdown. The prompt-only signal is **universe-sensitive** — it goes long
the ETFs too, and the 8-position cap reshuffles holdings. Read this as: the
untrained signal's risk-adjusted edge is thin and composition-dependent — which
is the case *for* training, not against the harness.

## Caveats (be honest)

1. **Bull-market regime.** 2024-01→2026-04 was a strong up-market; equal-weight
   wins almost by definition. The LLM's edge shows in *risk* (MDD, correlation),
   not return. A bear/sideways slice would test the thesis better.
2. **The reported LLM numbers are still price+technical only.** The full
   multi-modal data (news/fundamentals/sentiment/macro) has landed and the
   PIT-safe snapshot join is implemented ([`DATA_STORE.md`](DATA_STORE.md)), but
   the prompt-only LLM comparison has not yet been re-run with `multimodal=`
   enabled. Therefore the table above should still be read as the
   price+technical baseline, not as the final paper-parity multi-modal result.
3. **8.2 % unparseable** LLM replies degrade to HOLD. Now surfaced: the provider
   reports a `parse_stats` no-tag rate and warns past 20% (`run_comparison`
   prints it). Still open: tighten the prompt / add grammar-constrained decode.

---

## So what / next

- The prompt-only LLM is a **legitimate, low-drawdown, differentiated** baseline —
  it earns its row. It does **not** beat buy-and-hold on return in this regime,
  and we should not claim it does.
- This motivates the training work precisely: can **SFT → GRPO** lift the LLM's
  return *while keeping* the low correlation and drawdown? That is the
  Sub-project 2 hypothesis.

### SFT v0 evaluation (P2.1) — a clean negative result

Served the SFT v0 LoRA (`data/sft_adapter_v0/`) with vLLM `--enable-lora` and
probed it against the base model on a stride-sample of cached prompts:

| | base (prompt-only) | SFT v0 |
|---|---|---|
| decision mix (60 probes) | StrongBuy 32 / Hold 17 / Sell 6 / Buy 3 / StrongSell 2 | **Hold 60** |

**SFT v0 collapsed to 100 % HOLD** → holds nothing → all-cash → **CR 0 %, MDD 0 %**.
It does not beat prompt-only (or anything). Root cause: the v0 *templated* rationale
teaches the boilerplate (eval token-acc ~80 % is mostly template tokens) while the
decision token degenerates to the majority class. This is exactly why the paper
uses teacher-distilled, evidence-grounded rationales + a GRPO decision reward, not
templates.

### SFT v1 (P2.1 follow-up) — the collapse is fixed ✅

v1 applies two structural fixes: **completion-only loss** (`assistant_only_loss` —
grade only the assistant turn, so the gradient lands on the decision, not the
prompt) and **class balancing** (down-sample HOLD 37 % → 24 %). Same probe:

| | base | SFT v0 | **SFT v1** |
|---|---|---|---|
| decision mix (60) | SB 32 / H 17 / S 6 / B 3 / SS 2 | **H 60** | **H 24 / SB 23 / S 11 / B 2 / SS 0** |

v1 produces a **genuine, non-degenerate distribution** (was 100 % HOLD), with
shifts in both directions (HOLD→SELL, HOLD→StrongBuy, StrongBuy→HOLD). The
single-variable change (full-sequence → completion-only loss) restored the
decision signal — confirming the v0 root-cause diagnosis. Training: eval
token-acc 80 % → **98.6 %**, train-loss 0.53 → 0.089 (`data/sft_adapter_v1/`).

**Full SFT-v1 backtest** (14-ticker, same OOS window; `compare_lab/output_sftv1/`):

| Strategy | Cumulative | Sharpe | Max DD |
|---|---|---|---|
| Equal-weight | +143 % | 1.04 | 32.3 % |
| 12-1 Momentum | +52 % | 0.70 | 19.6 % |
| Prompt-only LLM | +36 % | 0.55 | 16.4 % |
| **SFT v1** | +29 % | 0.53 | **7.9 %** |

SFT v1 is the **most defensive** strategy in the set: lowest return, but **half
the drawdown of prompt-only** (7.9 % vs 16.4 %) at a near-identical Sharpe.
Training made it risk-averse — its decisions are 39 % StrongBuy / 41 % Hold /
18 % Sell (selective long book under the 8-position cap). It does **not** beat
prompt-only on return/Sharpe, so SFT-v0→v1 fixed the *collapse* but did not (yet)
lift return. **One unambiguous training win:** the parse-rate — v1 emits a valid
`[[[CLASS]]]` on **100 %** of inputs (NO_TAG 0 %, vs the prompt-only baseline's
8.2 %); completion-only SFT solved the format leak that P1.3's guardrail only
*surfaced*. Next: teacher distillation (vary the rationale, currently templated)
+ GRPO decision reward to push return while keeping the drawdown/parse wins.
- Immediate, cheap improvements: add a parse-rate guardrail, join the multi-modal
  snapshot, and add a bear-slice to the report.

### SFT v2 (teacher distillation) — a regression, recorded ❌

Distilled 250 reverse-reasoning §8 theses from a Qwen3-30B-A3B teacher
(`sft/distill.py`: the teacher is shown the volatility label and writes a thesis
justifying it; the stored pair is label-free snapshot → thesis), trained a fresh
LoRA (`data/sft_adapter_v2/`, served `sft-v2`), and ran the same 14-ticker
backtest (`compare_lab/output_sftv2/`):

| Strategy | Cumulative | Sharpe | Max DD | NO_TAG |
|---|---|---|---|---|
| Equal-weight | +143 % | 1.04 | 32.3 % | — |
| 12-1 Momentum | +52 % | 0.70 | 19.6 % | — |
| SFT v1 | +29 % | 0.53 | **7.9 %** | 0 % |
| **SFT v2** | +34 % | 0.46 | 20.7 % | **9.2 %** |

**v2 is worse than v1 on every risk axis.** It lifts raw return slightly (+34 %
vs +29 %) but loses the defensive edge that was v1's whole story — drawdown 2.6×
higher (20.7 % vs 7.9 %), Sharpe down (0.46 vs 0.53). The distilled verbose-thesis
style also broke format reliability: **9.2 % NO_TAG** (vs v1's 0 %), the student
rambling past the decision and never emitting `[[[CLASS]]]`. This is *not* a
token-budget artifact — at 4096 max-tokens the residual 151 no-tags average
~14.8k chars (some >20k): genuine non-termination, not truncation. (The first run
at 2048 tokens *looked* better — MDD 17.6 %, NO_TAG 16.2 % — but that was a
mirage: truncated replies fell back to flat HOLD, which suppressed drawdown.
Fixing truncation made risk metrics worse, confirming v2 just bets more
aggressively.)

**Verdict:** teacher distillation as configured did not help; the long §8 style
hurts both termination and risk. Next: GRPO RL on the **v1** base (keep v1's
parse/drawdown wins, push return with the decision reward); revisit distillation
(shorter, decisive teacher theses) only if GRPO stalls.

### GRPO RL on the v1 base — best Sharpe, but broke v1's discipline ⚠️

Merged the v1 LoRA into Qwen3-4B and trained a fresh GRPO LoRA (`compare_lab/grpo/`,
TRL GRPOTrainer on DGX Spark GB10, HF generation; reward_funcs = structure /
evidence / decision kept separate; 300 balanced pre-2024 prompts, 1 epoch / 69
steps, LR 5e-6, 8 generations/prompt). Same 14-ticker backtest
(`compare_lab/output_grpo/`):

| Strategy | Cumulative | Sharpe | Max DD | NO_TAG |
|---|---|---|---|---|
| SFT v1 | +29 % | 0.53 | **7.9 %** | 0 % |
| SFT v2 | +34 % | 0.46 | 20.7 % | 9.2 % |
| **GRPO** | **+37 %** | **0.58** | 21.6 % | 10.0 % |

GRPO has the **best return and Sharpe of all trained models** — the RL did push
risk-adjusted return up. But it **lost both of v1's signature wins**: drawdown
blew out to 21.6 % (v1 7.9 %), and the parse rate regressed to **10 % NO_TAG**
(v1 0 %). The NO_TAG is *not* truncation (completions average ~1.2k chars): in
159/164 cases the model **echoes the prompt's template menu verbatim**
(`[[[STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL]]]`) instead of choosing one class — a
degenerate mode v1 never had. Training signal was weak/noisy (decision reward
flat-to-negative over the epoch, entropy ~0.026 → near-deterministic v1 policy
gives little within-group exploration for GRPO to exploit), consistent with a
policy that drifted toward more aggressive bets without learning real discipline.

**Verdict:** GRPO is a *mixed* result — pick GRPO for max risk-adjusted return
(Sharpe 0.58), but **v1 remains the keeper** for the defensive profile (½ the
drawdown, perfect parse). GRPO is promising-but-unfinished: the 10 % template-echo
is cheaply fixable (active parse guardrail / format penalty), and more epochs +
higher entropy (sampling temperature, larger LR) would give the RL real signal.

### Gap-closing cycle 1 — multimodal SFT→GRPO (2025-H1) — did not close the gap ❌

The dominant gap vs the paper is the modality gap (we trained price-only; the paper
on 5 modalities). This cycle fed the multimodal store (news/fundamentals/sentiment/
macro) into the snapshot, leak-safe **train 2024 / eval 2025-H1**, 12 equities.
Spec/plan: `docs/superpowers/{specs,plans}/2026-06-25-multimodal-sft-grpo*`.

| Row (2025-H1, 12-eq) | CR | Sharpe | HR | MDD | NO_TAG |
|---|---|---|---|---|---|
| Equal-weight | +5.3 % | 0.26 | — | 27.8 % | — |
| 12-1 Momentum | −2.1 % | −0.22 | — | 19.6 % | — |
| Prompt-only LLM, multimodal **OFF** | −6.3 % | −1.12 | 32 % | 14.2 % | 8.3 % |
| Prompt-only LLM, multimodal **ON** | −6.5 % | −0.92 | 29 % | 20.2 % | **1.3 %** |
| SFT-mm v3 | +0.4 % | −2.65 | 4 % | 0.7 % | 0 % |
| SFT-mm → GRPO | −10.0 % | −1.35 | 29 % | 19.6 % | 0 % |

**Criterion 1 (does multimodal add signal?) — no.** The OFF/ON ablation (same base
model, same window) shows **identical return** (~−6 %); Sharpe nudges −1.12→−0.92.
The one clear effect is the parse rate: **NO_TAG 8.3 %→1.3 %** — more context → the
model commits to a parseable call. So multimodal helped *format*, not *returns*.

**Criterion 2 (does training help?) — no, and the trained models are degenerate.**
SFT-mm v3 collapsed to near-all-SELL (304/312 SELL → long-only → all-cash → CR≈0,
MDD 0.7 %, the mirror of v0's all-HOLD). GRPO over-corrected to 74 % StrongBuy and,
in a *down* window, lost −10 %. The 2024-trained decision policy doesn't generalise
to the 2025-H1 regime.

**Caveats that bound the conclusion:** (1) 2025-H1 was flat/down (equal-weight only
+5.3 %, momentum −2.1 %) — a hard, short (6-mo, noisy-Sharpe) window where the
negative is as much regime as method; (2) our multimodal snapshot is compact
(~900 tokens vs the paper's 15–23k), so the modality signal is thin; (3) tiny train
sets (290 SFT / 243 GRPO). **Verdict: cycle 1 did not close the gap — multimodal
improved parse reliability but not returns, and the 2024→2025 regime shift exposed
decision-distribution collapse.** Next levers: a longer/richer multimodal context,
a regime-matched eval, and anti-collapse training (the all-SELL needs the same
completion-only/balancing care v1 needed for all-HOLD).

**Lever 1 tested — richer context is *not* the bottleneck.** Enriched
`render_sections` (3-bucket news ≤50 + 12 sentiment events; snapshot ~900→~2k tok)
and re-ran the prompt-only ablation on 2025-H1:

| Prompt-only LLM (2025-H1) | CR | Sharpe | MDD | NO_TAG |
|---|---|---|---|---|
| price-only (OFF) | −6.3 % | −1.12 | 14.2 % | 8.3 % |
| multimodal thin (~900 tok) | −6.5 % | −0.92 | 20.2 % | 1.3 % |
| multimodal **rich (~2k tok)** | **−8.4 %** | −0.89 | 22.4 % | **0.6 %** |

Doubling the context moved **parse monotonically** (NO_TAG 8.3→1.3→0.6 %) and Sharpe
marginally (−1.12→−0.89), but **CR got worse** (−6.3→−8.4 %) and drawdown rose — more
news/sentiment made the model bet *more confidently and aggressively* without
improving call *quality*, so in a down window it lost more. Conclusion: modality
*quantity* isn't the gap; per the ablation-first de-risk we **did not retrain** on the
rich context. Remaining levers: regime-matched eval + anti-collapse training.

**Lever 2 tested — regime is not the excuse; it's method + training data.** Put the
price-only trained models (sft-v1, price-only GRPO; pre-2024 training, so leak-safe)
on the **same** flat 2025-H1 / 12-equity window as the multimodal models:

| Model (2025-H1, 12-eq) | CR | Sharpe | MDD |
|---|---|---|---|
| price-only **SFT-v1** | **+0.6 %** | −0.30 | **9.3 %** |
| price-only GRPO | −5.6 % | −0.96 | 17.7 % |
| multimodal SFT-mm | +0.4 % | −2.65 | 0.7 % (all-SELL) |
| multimodal GRPO | −10.0 % | −1.35 | 19.6 % |

On the identical window, **price-only SFT-v1 survives** (+0.6 %, MDD 9.3 % — the only
positive trained model), so 2025-H1 is *not* an impossible regime. The multimodal
versions are **worse than their price-only counterparts** (SFT-mm collapsed; mm-GRPO
−10 % vs price-GRPO −5.6 %) → multimodal hurt rather than helped. Confounder to keep
honest: v1 trained on 2017–2023 (7 yr, 4.2 k examples) vs the multimodal models on
2024 only (1 yr, ~290), so "multimodal worse" also carries a much smaller/shorter
training set. **Verdict: the negatives are method + training-data, not the regime —
v1's cross-regime robustness (defensive, no collapse) is what the multimodal models
lack.** (v1's own 2024–26 number was +29 %, so it *is* regime-sensitive on return,
but it never collapses.) Next: anti-collapse training.

**Hardware aside — GB10 did *not* cap RL correctness, and barely caps RL depth.**
We'd used HF rollouts for GRPO out of caution ("vLLM+Ray broken on GB10"), but that
issue is multi-node/TP>1 only. Verified that TRL GRPO **vLLM colocate rollout**
(`--use-vllm`, TP=1, no Ray) runs fine on a single GB10: **~77 s/step vs 232 s/step
for HF (~3×)** and higher entropy (0.10 vs 0.026 → more exploration). The one blocker
was KV-cache sizing (Qwen3-4B advertises 262 k context); cap `vllm_max_model_length`.
So deeper, better-exploring RL is feasible on GB10 — a tool for the anti-collapse
cycle, not a hardware excuse for the negatives (training correctness was always fine:
SFT-mm hit 97.9 % token-acc).

**Artifacts:** `compare_lab/output{,_14,_sftv1,_sftv2,_grpo,_mm_off,_mm_on,_mm_on_rich,_mm_sft,_mm_grpo,_po_v1_h1,_po_grpo_h1}/comparison.csv`,
`compare_lab/output/oos_daily_returns.csv`, `compare_lab/output/{equity,report}.html`,
this memo.
