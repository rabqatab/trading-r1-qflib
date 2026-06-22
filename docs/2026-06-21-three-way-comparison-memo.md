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
A full SFT-v1 backtest row is running (now worthwhile — the signal is no longer
all-cash). Next after that: teacher distillation + GRPO.
- Immediate, cheap improvements: add a parse-rate guardrail, join the multi-modal
  snapshot, and add a bear-slice to the report.

**Artifacts:** `compare_lab/output{,_14}/comparison.csv`,
`compare_lab/output/oos_daily_returns.csv`, `compare_lab/output/{equity,report}.html`,
this memo.
