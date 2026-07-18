> **Historical record (2026-06-30)** of the reward-audit discussion. Conclusions absorbed into README finding #1 and [`2026-07-07-final-synthesis.md`](2026-07-07-final-synthesis.md).

# Jiwoong reward rebuttal — cause, symptoms, results, and proposed next actions

> Date: 2026-06-30  
> Project: `tradingR1_qflib`  
> Context: QF-Lib-LLM Telegram discussion following the 2026-06-29 results report and GRPO reward audit.  
> Related artifact: `docs/2026-06-29-results-report.html`  
> Related code: `compare_lab/grpo/rewards.py`, `compare_lab/eval_labels.py`, `compare_lab/labeling.py`

## 1. Executive summary

Jiwoong's reward-related rebuttal is substantively correct: if the model is being judged by the original 5×5 decision-reward matrix, then a model whose average decision reward remains materially negative — and especially below a simple constant-policy baseline — should **not** be described as having successfully optimized that reward.

The project therefore needs to separate four claims that were previously easy to conflate:

1. **Exact label accuracy** — did the predicted class exactly equal the deterministic 5-class label?
2. **Rank-IC / label-fidelity** — did the predicted class move monotonically with the underlying forward `make_signal` score?
3. **Decision-matrix reward** — did the predictions score well under the asymmetric 5×5 utility matrix used for matrix-GRPO?
4. **Portfolio backtest** — did the resulting long-only top-8 portfolio make money in a particular market regime?

The key rebuttal is that a positive backtest Sharpe or positive IC is not, by itself, proof that the **matrix reward** was learned. The mean reward itself must beat the relevant baselines.

## 2. What Jiwoong challenged

### 2.1 Direct challenge

In the 6/30 discussion, after model-level decision-reward scores were summarized, Jiwoong asked whether the mean decision reward should be **positive** to be meaningful.

The short answer is:

- **Strict reward-optimization claim:** yes, the model should ideally have positive mean reward, or at minimum beat meaningful baselines.
- **Weak signal-detection claim:** a negative mean reward can still coexist with positive IC, but then the correct claim is only that the model has weak directional signal, not that it has mastered the reward.

### 2.2 Earlier 6/29–6/30 reward context

The preceding discussion also clarified the newer graded reward:

```text
reward ≈ bet × signal
bet ∈ {STRONG_SELL, SELL, HOLD, BUY, STRONG_BUY}
signal = realized continuous forward score from 3/7/15-day smoothed price movement
```

In this framing:

- `bet` is the model's directional action.
- `signal` is the realized direction/strength of the ground-truth forward move.
- A negative reward is possible and meaningful: it means the model placed directional exposure against the realized signal, with downside magnified.

Jiwoong additionally noted that if `bet(+/-) × signal` is used, then it is essentially a long/short reward view. Nick agreed: for reward purposes, it can be read as long/short.

## 3. Root cause

### 3.1 Metric conflation

The root cause is that several metrics were being discussed under the loose phrase “reward worked”:

| Metric | What it measures | Failure mode if confused |
|---|---|---|
| Exact accuracy | Exact 5-class match | Too harsh; misses near-correct ordinal calls |
| Rank-IC | Monotonic alignment with continuous forward signal | Can be positive even when utility/reward is poor |
| Matrix reward | Asymmetric utility of predicted class vs true class | Can be negative despite positive IC if errors are costly |
| Backtest Sharpe | Portfolio performance under sizing/risk regime | Can improve from long bias or regime, not prediction skill |

Matrix-GRPO improved rank-IC but did not clearly improve the mean matrix reward. That is the central inconsistency Jiwoong's question exposed.

### 3.2 The 5×5 matrix has an intentionally harsh negative skew

The matrix reward is asymmetric and capital-preservation oriented:

- Correct class: `+1.0`
- Near-correct class: often `+0.75`
- HOLD when action is needed: negative
- False-bullish in a down move: strongly negative
- Invalid/no-tag/template echo: `-2.5`

Therefore, a model can have positive IC but still negative mean reward if it:

1. is directionally somewhat right on average,
2. but makes too many high-conviction wrong calls,
3. or emits invalid/no-tag/template-echo outputs,
4. or concentrates exposure in a way that the matrix penalizes heavily.

### 3.3 GRPO matrix optimized the wrong effective behaviour

The matrix-GRPO run moved predictions toward more aggressive bullish exposure and improved IC, but it also increased:

- `NO_TAG` / invalid outputs,
- extreme call errors,
- drawdown,
- and exposure concentration.

Thus it improved a signal-ranking view without cleanly improving matrix utility.

## 4. Observed symptoms

### 4.1 Decision distribution / collapse symptoms

The 6/30 screenshot shared by Jiwoong contrasted in-sample and out-of-sample behaviour:

| Row | sft-v1 IS | sft-v1 OOS | GRPO IS | GRPO OOS |
|---|---:|---:|---:|---:|
| IC / label hit | 0.124 | 0.127 | **0.195** | 0.189 |
| current SR | 0.85 | 0.85 | 0.90 | 1.16 |
| market EW SR | 1.02 | 1.23 | 1.02 | 1.23 |
| IS decision mix | **87% StrongBuy collapse** | — | BUY46 / HOLD25 / SELL20 normal | — |

Interpretation:

- SFT-v1 in that in-sample diagnostic showed an unhealthy StrongBuy-heavy collapse.
- GRPO repaired the distribution into a more plausible BUY/HOLD/SELL mix.
- GRPO improved IC in that diagnostic.
- However, this does not alone settle whether the reward objective was truly optimized, because the mean reward still needs to beat baselines.

### 4.2 Model-level reward audit from `eval_labels.py`

The most relevant audit computes:

- exact accuracy,
- rank-IC,
- tail-IC,
- mean 5×5 decision-matrix reward,
- best constant baseline,
- always-HOLD baseline,
- NO_TAG rate,
- and backtest metrics.

Recent outputs from `uv run python -m compare_lab.eval_labels ...` showed:

| Model | Exact acc | Rank-IC | Tail-IC | Mean matrix reward | NO_TAG | Backtest interpretation |
|---|---:|---:|---:|---:|---:|---|
| SFT v1 | 28.1% | +0.127 | +0.168 | -0.298 | 0.0% | Defensive keeper, but weak label fidelity |
| SFT v2 | 27.4% | +0.134 | +0.188 | -0.447 | 9.2% | Distillation regression |
| GRPO matrix | 22.6% | +0.189 | +0.246 | -0.513 | 10.0% | IC improved, reward worsened |
| MM SFT | 12.5% | -0.029 | -0.049 | -0.869 | 0.0% | Collapse/failure |
| MM GRPO | 21.8% | +0.174 | +0.258 | -0.335 | 0.0% | Some signal, weak backtest |
| v1-reg GRPO | 31.7% | +0.183 | +0.238 | -0.289 | 4.5% | Collapse repaired, near v1-level |
| graded GRPO | 30.7% | +0.190 | +0.257 | -0.279 | 3.7% | Best LLM trade result, still IC-capped |

Important baseline note from the same audit:

- On the full-window label set, the **best constant policy** was around `-0.055`.
- Many learned models remained below that value under the 5×5 matrix.

That means the matrix-reward claim must be weakened: the models show weak predictive structure, but do not yet beat a simple baseline under the original reward utility.

## 5. Resulting interpretation

### 5.1 What remains valid

The following claims remain defensible:

1. **The models are not random.**  
   Rank-IC values around `0.13–0.19` indicate weak-but-real monotonic alignment with the forward signal.

2. **GRPO changed behaviour.**  
   Matrix-GRPO and later variants altered the decision distribution and sometimes improved IC.

3. **Graded reward made GRPO learnable.**  
   The graded reward produced the healthiest training curve observed so far (`-1.58 → +0.33`) and the best bull-window backtest among LLM models.

4. **Backtest gains exist in bull windows.**  
   Graded GRPO reached roughly `CR +52.7%`, `Sharpe 0.93`, `MDD 11.2%` on the 2024–2026 14-equity bull window.

### 5.2 What must be walked back or qualified

The following claims should be avoided or explicitly qualified:

1. **“Matrix-GRPO learned the decision reward.”**  
   Not supported if mean matrix reward is below constant baselines.

2. **“Higher Sharpe proves better prediction.”**  
   Not supported. The 2025-H1 flat-regime check showed that graded GRPO can have higher IC while losing money.

3. **“Reward engineering broke the prediction ceiling.”**  
   Not supported. Label-fidelity IC remains near the same approximate `~0.2` ceiling.

4. **“Positive backtest = successful reward.”**  
   Not necessarily. Bull-window long bias and conviction concentration can drive backtest gains without improving true label fidelity enough.

## 6. Consequences for the research narrative

The corrected narrative should be:

> Matrix-GRPO improved rank-IC and changed the action distribution, but did not demonstrate clean optimization of the original 5×5 decision reward. Graded continuous reward made the RL curve learnable and improved bull-window returns, but the honest-lens audit shows the prediction ceiling remains around IC ≈ 0.2. Therefore the remaining problem is not primarily “more GRPO”; it is better predictive input, drawdown control, and regime robustness.

This aligns with the existing 2026-06-29 memo conclusion:

- reward density helps learning dynamics,
- but does not break the prediction ceiling,
- and does not solve the paper-level drawdown gap.

## 7. Proposed gates for future runs

Jiwoong's objection should be turned into explicit acceptance gates.

### 7.1 Reward gates

For any future model claiming to improve the decision reward:

1. `mean_matrix_reward > best_const`
2. `mean_matrix_reward > always_hold`
3. `mean_matrix_reward > base_model_mean_reward`
4. ideally `mean_matrix_reward > 0`

A model may still be interesting with negative reward, but then it should be described as **weak signal**, not **reward success**.

### 7.2 Prediction gates

Require:

1. positive pooled rank-IC,
2. positive per-ticker IC where sample size allows,
3. positive tail-IC,
4. monotone mean realized signal by predicted class,
5. confusion matrix review for extreme false-bullish errors.

### 7.3 Format gates

Require:

1. `NO_TAG` below a fixed threshold, ideally `< 2–3%`,
2. zero or near-zero template-menu echo,
3. no model accepted if invalid outputs are responsible for reward degradation.

### 7.4 Regime gates

Do not accept a model based only on the 2024–2026 bull-window result. Require at least:

1. full-window backtest,
2. 2025-H1 flat/regime check,
3. paper-window comparison,
4. drawdown comparison against v1 and paper target.

## 8. Immediate implementation suggestions

1. **Add a reward-audit table to every results report.**  
   Include exact accuracy, rank-IC, tail-IC, mean matrix reward, best-constant reward, HOLD-constant reward, NO_TAG, and backtest metrics.

2. **Make `mean_matrix_reward > best_const` a hard claim gate.**  
   If this fails, the report must say: “positive IC, but reward utility not beaten.”

3. **Separate matrix reward and graded reward claims.**  
   Graded GRPO should be evaluated on its own training reward (`bet × signal`) and also cross-checked on the 5×5 matrix only as a secondary diagnostic.

4. **Report long-bias separately.**  
   Add decision distribution and net exposure metrics, because a bull-window Sharpe can be generated by exposure concentration rather than prediction skill.

5. **Keep the research focus on predictive inputs and regime robustness.**  
   Given the observed IC ceiling near `~0.2`, more reward shaping alone is unlikely to produce the paper-level drawdown profile.

## 9. Bottom line

Jiwoong's rebuttal should be preserved as a methodological correction:

> A model with negative mean decision reward can still have weak directional signal, but it has not “won” under the reward objective. Future GRPO claims must compare mean reward against constant baselines and prior models, not only report IC or Sharpe.

This correction materially improves the rigor of the project: it prevents confusing bull-regime portfolio gains with genuine reward learning or label-fidelity improvement.
