# Post-ceiling roadmap — what related work says to do next (2026-07-06)

> We reached the paper's performance ceiling and proved it is **information-bound**
> (raw-return IC ≈ 0.06 = I(X;Y), [[why-the-ceiling]]). This synthesizes 4 parallel literature
> sweeps (LexiconArxiv + WebSearch) into what's actually worth building. The unifying finding:
> **stop optimizing point-prediction; the productive levers are (1) a more tradeable TARGET,
> (2) genuinely NEW information, (3) risk-management to convert the weak signal into performance,
> (4) LLM-as-encoder not decision-maker.** Every sweep independently landed on our own memory note:
> the real gap vs the paper is **drawdown + regime, not IC**.

## The four directions, ranked by (feasibility × honest upside) for OUR setup

### A. Better TARGET — highest-ROI, buildable now, no new data
Our `make_signal` is the *weakest* target framing in the literature (overlapping-EMA-blend, flat
5-class quantile-cut softmax). Two of our own findings are textbook artifacts:
- the 4× proxy/raw gap = **overlapping-returns inflation** (Boudoukh-Richardson-Whitelaw RFS 2008; Lo-MacKinlay).
- Opus's 29% blind class-accuracy = **quantile-cut boundary classes** (López de Prado): STRONG_BUY/SELL
  are percentile edges, not events — *un-learnable by construction*, even for Opus 4.8.

Learnability ordering (worst→best): flat 5-class softmax **(ours)** < regression < ordinal < meta-label
binary < cross-sectional ranking.

**Recommended (priority):**
1. **Triple-barrier + meta-labeling** (López de Prado, AFML Ch.3): vol-scaled barriers (EWMA-σ), predict
   a binary "act / pass" — kills overlap inflation, is risk-aware, replaces un-learnable tails with
   *event* labels. Directly fixes the 29% problem.
2. **Cross-sectional rank target** (learning-to-rank; STHAN-SR AAAI 2021; Kronos AAAI 2026 uses RankIC):
   cancels common market moves, concentrates idiosyncratic signal, matches where P&L comes from.
3. **Cheap wins if keeping 5 classes:** ordinal (cumulative-link) loss instead of softmax; collapse 5→3;
   *learn* the horizon instead of blending {3,7,15d} (Label-Horizon-Paradox, arXiv 2602.03395).
4. **Eval hygiene:** non-overlapping returns + Newey-West/Hansen-Hodrick SEs; report rank-IC + Sharpe.

⚠️ **Will not raise raw point-IC above ~0.06** (that's the info floor) — but makes the same 0.06
*profitable* rather than inflated. This is the honest reframe.

### B. Risk-management to convert IC 0.06 → performance — the paper's ACTUAL edge
"You cannot buy Sharpe with better prediction at the info bound; you buy it with portfolio mechanics."
Daniel-Moskowitz ("Momentum Crashes", JFE 2016): regime de-risking alone **~doubles** momentum Sharpe.

| # | technique | expected | honest? |
|--|--|--|--|
| 1 | **CVaR/CDaR drawdown optimization + a crash-validated 2-state regime kill-switch** (Rockafellar-Uryasev; Chekhlov-Uryasev-Zabarankin; Werge ESWA 2021) | Sharpe +0.1–0.3, MDD −30–50% | ✅ *iff* the regime model is validated on 2020/2022, not fit to 2024 |
| 2 | **Fractional-Kelly (¼–½) + vol-scaled sizing** (MacLean-Thorp-Ziemba) | drawdown −30–50%, stabilizes Sharpe | ✅ pure sizing, no beta |
| 3 | Conditional (extreme-state) vol-targeting (Moreira-Muir; but Cederburg JFE 2020 OOS-null) | Sharpe +0.05–0.1 | 🟡 continuous version = bull-window beta; use conditional only |

Breadth (Fundamental Law IR=IC·√BR): real but soft — gated by the **transfer coefficient** (Clarke-de
Silva-Thorley); long-only + correlated large-caps push TC to 0.3–0.6, and effective breadth ≪ 150 when
names are correlated. Go long/short to lift TC; don't expect the naive √BR.

### C. New INFORMATION to raise I(X;Y) — the only ceiling-*moving* lever, but narrow
DPI on a firm's own price says nothing about **other firms' info** or **forward-looking derivatives** —
those two escape the ceiling; everything else (insider, short-interest, satellite, card) is a *faster
proxy for the firm's own fundamentals* that price wins the race on in large caps.

| # | source | fit for weekly / 150-large-cap | note |
|--|--|--|--|
| 1 | **Cross-firm relational / lead-lag** (Cohen-Frazzini JF 2008, 150bps/mo) | ✅ escapes own-price DPI; **but extend links OUTSIDE the 150** to less-watched suppliers/customers | needs SEC principal-customer / supply-chain data |
| 2 | **Analyst estimate-revision momentum (Finnhub `revenue-estimate` — our paid key!)** | ✅ **large-cap-native, information event, weekly, already paid for** | the real value of the paid key — NOT insider |
| 3 | Signed options order flow (Pan-Poteshman, ~40bps/wk) | ✅ horizon, ❌ needs proprietary signed/intraday data | free unsigned put/call ≈ worthless |
| ❌ | **insider-sentiment (Finnhub)**, short-interest, IV-skew | small-cap / borrow-fee effects → ~0 in liquid large-caps | traps for our universe |
| ❌ | credit-card / satellite / web-traffic | quarterly earnings-anticipation, small-cap, decaying | wrong horizon + wrong universe |

Universe tax (all of the above): every alt-data alpha is stronger in small/low-coverage names — our
150-large-cap choice is the single biggest headwind. Haircut everything for McLean-Pontiff (58%
post-publication decay) and Harvey-Liu-Zhu (t>3, not 1.96).

### D. LLM-as-ENCODER, not decision-maker — modest, buildable, resolves our text-null
Every "text helps" result differs from our "text ≈ 0" in two ways: they use **embeddings not scalar
sentiment/decisions**, and the big-alpha one (SESTM, Ke-Kelly-Xiu) is **return-supervised**. Our
end-to-end SFT→GRPO over the raw prompt is the *weakest* config in every paper (Merrill/Tan: LLMs are
weak text→numeric encoders; TabLLM/TabPFN: keep the GBM as the head, LLM upstream as features).

**Minimal experiment (1–2 days, reuses gbm_ceiling + eval):** GBM(price) vs GBM(price + pooled
finance-embedding of the news, e.g. FinBERT) vs GBM(price + old LLM-sentiment-scalar), same time-OOS
folds. If embeddings beat scalar+price, add the SESTM move: a **return-supervised** projection of the
embedding as the feature. SCRL-LG's honest number is only +0.008 rank-IC, so expect ~0.005–0.015, not a
break. A return-supervised embedding that *still* adds ~0 OOS = a strong publishable negative (text
carries no incremental alpha over price in large caps).

## Recommended sequence (each is a self-contained experiment)
1. **Target: triple-barrier + meta-labeling** (A) — biggest honest lift, no new data, direct fix to our
   two artifacts. Build a `labeling_triplebarrier.py` alt to `make_signal`; re-run the GBM/base ceiling.
2. **Risk: CVaR/CDaR + fractional-Kelly + crash-validated regime switch** (B) — attacks the exact
   drawdown/regime gap; prediction-free; must be validated on 2020/2022.
3. **Info: analyst revision momentum from the Finnhub `revenue-estimate` endpoint** (C#2) — the one new
   signal that's large-cap-native and already paid for.
4. **Encoder hybrid ablation** (D) — cheap, and either finds ~0.01 IC or yields a clean negative.
5. (Higher effort) cross-firm links reaching outside the 150 (C#1) — the only true ceiling-mover, but
   needs supply-chain data.

**One-line honest summary:** the ceiling on *prediction* is real and ~0.06; the remaining wins are in
**target design, risk management, one new large-cap-native signal (analyst revisions), and using the
LLM as an encoder** — none "break" the ceiling, but together they turn an honest 0.06 into the paper's
risk-managed performance instead of a smoothing-inflated mirage.

## Reference index (LexiconArxiv/WebSearch-verified)
Target: Boudoukh-Richardson-Whitelaw (RFS 2008); Hodrick (RFS 1992); López de Prado AFML Ch.3
(triple-barrier/meta-labeling); Continuous Trend Labeling (Entropy 2020); Label-Horizon-Paradox
([2602.03395](https://arxiv.org/abs/2602.03395)); STHAN-SR (AAAI 2021); ordinal regression (Fathony
NeurIPS 2017). Risk: Grinold-Kahn; Clarke-de Silva-Thorley (FAJ 2002); MacLean-Thorp-Ziemba (QF 2010);
Moreira-Muir (JF 2017) + Cederburg (JFE 2020); Rockafellar-Uryasev (2000); Chekhlov-Uryasev-Zabarankin
(2005); Daniel-Moskowitz (JFE 2016); Werge (ESWA 2021). Info: Cohen-Frazzini (JF 2008); Pan-Poteshman
(RFS 2006); Lakonishok-Lee (RFS 2001); Rapach-Ringgenberg-Zhou; McLean-Pontiff (2015); Harvey-Liu-Zhu
(2016). Encoder: SCRL-LG ([2310.05627](https://arxiv.org/abs/2310.05627)); Guo-Hauptmann (EMNLP 2024);
Ke-Kelly-Xiu SESTM (NBER w26186); Sentiment Trading w/ LLMs ([2412.19245](https://arxiv.org/abs/2412.19245));
FinSeer ([2502.05878](https://arxiv.org/abs/2502.05878)); Merrill/Tan (NeurIPS 2024).
