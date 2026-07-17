# Remaining items — second literature sweep (2026-07-17)

> After roadmap B (RU-conformal CVaR control) closed, five items remained open. This sweep
> (5 parallel WebSearch agents) re-checks each against 2023–2026 literature before building.
> Predecessors: [[2026-07-06-post-ceiling-roadmap]] · [[2026-07-17-cvar-conformal-control]].
> **Headline revision: triple-barrier is demoted, estimate-revision breadth and analyst
> co-coverage momentum are promoted, and the cost/survivorship gate now has concrete numbers.**

## The five remaining items, re-ranked after the sweep

| pri | item | sweep verdict | expected effect | cost |
|--|--|--|--|--:|
| 1 | **Transaction-cost haircut + turnover reduction** (gate) | Concrete recipe exists: banding + smoothing → survivable | Sharpe 0.92 → ~0.7–0.8 (vs ~0.4–0.5 unbuffered) | 0.5 day, no new data |
| 2 | **Estimate-revision breadth (Finnhub eps/revenue-estimate)** | Best-supported new signal; survives post-2010 in institutional universes | IC ~0.02–0.04 incremental | 1 day, key already paid |
| 3 | **Analyst co-coverage momentum (C#1-lite)** | Ali-Hirshleifer subsumes all link effects; buildable from data we have | IC ~0.005–0.02 | 1 day |
| 4 | **Target redesign — now RANK target, not triple-barrier** | TBL demoted: zero replicated OOS-Sharpe evidence + one published negative | honest reframe, not IC gain | 1–2 days |
| 5 | **LLM-encoder ablation (D)** | Prior confirmed & sharpened downward: expect ~0.005–0.010 | likely publishable null | 1–2 days |
| — | **Non-survivorship universe re-test** (gate) | Needs Norgate (~$50/mo) for point-in-time top-150-by-cap | haircut 1–3 pp/yr plausible | $ decision |

## 1. Costs (gate) — the numbers we were missing

- Realistic large-cap one-way cost: **2–3 bps base, 5 bps stress** (megacap effective half-spreads
  1–2 bps + small impact; Frazzini-Israel-Moskowitz JF 2018: ~8.9 bps impact for large caps at
  AQR-scale, we'd sit lower) + 25–50 bps/yr borrow on the short leg.
- Drag ≈ 252 × daily one-way turnover × cost. Unbuffered daily quintile LS ≈ 50%/day →
  ~3.8%/yr at 3 bps ≈ **half our gross** (Sharpe 0.92, vol ~5–6% → gross 4.6–5.5%/yr).
- **Mitigation ranking (Novy-Marx-Velikov RFS 2016): buy/hold banding is the single most
  effective** (enter at quintile, hold until exiting top/bottom ~30–35% → turnover −40–60%),
  then 3–10d EMA of ranks (= Gârleanu-Pedersen JF 2013 partial adjustment), then
  Jegadeesh-Titman overlapping sub-portfolios (turnover /K).
- Keep Sharpe ≥ ~0.6 at 3 bps → need ≤ ~20–25%/day one-way turnover. **Achievable with
  banding + smoothing; expected haircut ~0.1–0.2 Sharpe + ~0.05 borrow.** Also add a
  next-open-execution variant (same-close is optimistic).

## 2. Estimate-revision breadth (C#2 refined) — the recipe

- **Signal**: diffusion/breadth = (#up − #down)/#estimates over a 3-month window, EPS and
  revenue estimates separately, **sector-neutralized**; breadth is more robust than magnitude
  in large caps and has ~83% monthly persistence (Mill Street 2003–2022: top-decile +7.6%/yr,
  t=2.9; Robeco: revisions factor stayed positive 2010–2019 when value/size died).
- Revenue revisions add *incremental* persistence over EPS (Jegadeesh-Livnat JAE 2006) — use
  as confirmation filter, not standalone.
- **Price targets**: only ΔTP/P (Brav-Lehavy JF 2003) or dispersion-conditioned TPER
  (Palley-Steffen-Zhang MS 2025: sign flips when dispersion high); **raw TPER level is a
  contrarian/short signal in large caps** (Han-Kang-Kim JFM 2022). Drop targets >90d old.
- **Combiner**: Feldman-Livnat-Zhang (JPM 2012) 3-signal agreement composite (est↑+TP↑+rec↑
  reinforce; contradictions cancel) — direct blueprint for our rec+est+TP endpoints.
- **PEAD**: Martineau (CFR 2022) — dead in non-microcaps since ~2006; 2025 "revival" papers
  don't exclude microcaps. Our +0.068 (2025-H1) is regime, consistent with our own multi-year
  finding. Don't budget durable PEAD alpha.
- Haircut everything pre-2010 by ~50% (McLean-Pontiff) + large-cap discount → **IC 0.02–0.04
  expected**, slower/more persistent than the 63d rec-change (good: lowers turnover).

## 3. Cross-firm links (C#1) — cheaper than we thought

- Plain customer momentum (Cohen-Frazzini JF 2008) is **dead post-publication** (replication
  arXiv:2301.11394: insignificant post-2008, partly small→large lead-lag artifact).
- **Shared-analyst co-coverage momentum (Ali-Hirshleifer JFE 2020) subsumes ALL link types**
  (industry/geo/customer/tech) in spanning regressions, 1.68%/mo (t=9.67) pre-haircut, and is
  buildable from Finnhub analyst coverage we already have: feature = co-coverage-weighted mean
  of neighbors' past 1–4wk returns, excluding own. ~1 day.
- Also check the Finnhub `supply-chain-relationships` endpoint (tier-dependent) and the free
  Hoberg-Phillips TNIC competitor network. GNNs: no credible OOS evidence over the linear
  neighbor-average feature — skip.
- Caveats for our setup: our 150 large caps are the *fast, well-watched* side (Hou RFS 2007:
  lead-lag runs big→small, weekly frequency) → expect **half or less of headline**, IC
  0.005–0.02; always control for the target's own 1-week reversal.

## 4. Target redesign (A) — triple-barrier DEMOTED, rank target promoted

- **TBL-as-target has zero replicated OOS-Sharpe evidence**; one published negative (AEDL,
  Appl. Sci. 2025: plain TBL avg Sharpe −0.03 across 16 assets). Naive replications also fail
  on label-concurrency (need AFML Ch.4 uniqueness weights + purged CV).
- Meta-labeling (Joubert JFDS series) **filters an existing edge, doesn't create one** — with
  decade-mean IC 0.01–0.03 the meta-model would mostly learn vol timing, which the RU-conformal
  layer already does. Keep TBL only as the {0,1} outcome generator for a *calibrated sizing*
  meta-model (JFDS 5(2):23 recipe: EWMA-σ barriers m≈1.5, 5d vertical, sigmoid sizing) — and
  only if we ever need a sizing layer beyond λ-control.
- **What the evidence actually supports**: cross-sectional **rank loss** (LambdaRankIC,
  arXiv:2605.00501 — closed-form lambda gradients in XGBoost, wins under low SNR/heavy tails =
  our regime) + **label-horizon sweep** (Label-Horizon-Paradox arXiv:2602.03395: train on
  3–10d labels, evaluate at 7d). Cheap, pandas/XGBoost-friendly.

## 5. LLM encoder (D) — prior sharpened downward

- Band confirmed: **expect +0.005–0.010 IC** for text over price-only GBM in large-cap weekly
  (upper bound: arXiv:2510.15691, +0.013 monthly IC fusion on ~1000 NA stocks; deflators:
  Lopez-Lira size-concentration, S&P 500 FinBERT-PCA null arXiv:2606.29290 (t=0.92), lookahead
  memorization arXiv:2512.23847).
- Design updates: (a) frozen FinBERT+PCA arm = **published-null replication arm**; (b) the
  wins come from *return-supervised* projections — add a supervised-reduction arm; (c) evaluate
  post-cutoff (≥ mid-2024) or with a chronological encoder (ChronoBERT arXiv:2502.21206);
  (d) gate/ensemble text vs price heads — naive concatenation can dilute; (e) encoder choice is
  second-order, supervision is first-order.
- Our end-to-end SFT/GRPO ≈ −0.01 is consistent with the literature: nothing published shows
  end-to-end small-LLM decisions beating encoder+tabular-head in large caps.

## 6. Survivorship (gate)

- Bias magnitude: +0.10 Sharpe / +5 pp/yr documented in breadth universes; large-cap version
  milder but **dollar-neutrality does not cancel it** (long leg = decade winners, short leg =
  stocks that survived shorting). Expect 1–3 pp/yr.
- Feasible path without CRSP: **Norgate US Platinum (~$50/mo)** — delisted securities +
  point-in-time membership; reconstruct *top-150-by-PIT-market-cap* (avoids needing S&P
  membership). Free cross-check: GitHub `fja05680/sp500` constituent history. Sharadar (~$49/mo)
  as alternative (no index history; proxy by PIT cap). **Requires a spend decision.**

## Recommended sequence

1. **Cost haircut + banding/smoothing** on the CVaR-controlled combo (pri 1) — decides whether
   anything else matters; 0.5 day, zero new data.
2. **Estimate-revision breadth** backfill + IC test (pri 2) — best expected value per day.
3. **Co-coverage momentum** feature (pri 3) — 1 day, reuses Finnhub data.
4. **Rank-target + horizon sweep** (pri 4) — replaces the old triple-barrier plan.
5. **Encoder ablation** (pri 5) — run last; value is mostly the clean (likely null) replication.
6. **Norgate PIT re-test** — after 1–3 show the strategy survives costs; needs user's $ call.

## Addendum (same day) — arXiv-restricted pass + LexiconArxiv corpus pass

The first sweep was WebSearch-only (LexiconArxiv was disconnected). A supplementary
arXiv-restricted agent + a reconnected-LexiconArxiv pass added, in priority order:

1. **Risk-control line (stacks on our RU-CVaR controller)** — the richest vein:
   - **arXiv:2602.03903** "Regime-Weighted Conformal Calibration for Nonstationary VaR" —
     regime-similarity + time-decay weighted conformal buffer; improves stress-regime
     calibration (equities 1990–2024). *Directly stackable*: an optional upgrade to
     `ru_conformal()` if the 2020-COVID-style burst overshoot ever matters.
   - **arXiv:2510.08748** "Conformal Risk Training" (**NeurIPS 2025**, per LexiconArxiv) —
     end-to-end differentiable conformal risk control incl. CVaR: train the *signal through*
     the controller instead of bolting it on. Research-grade, not a quick win.
   - **arXiv:2603.13252** "When Alpha Breaks" — negative result worth internalizing:
     inverse-uncertainty position sizing *degrades* rankers (uncertainty ∝ |score|); a binary
     regime-trust gate (AUROC ~0.72–0.75) beats continuous confidence sizing. Endorses our
     exposure-gate architecture over per-name confidence weighting.
   - No citations of 2606.00320 exist yet (too recent; absent from LexiconArxiv corpus too).
2. **Lead-lag/co-coverage**: **arXiv:2410.20597** (analyst-network GAT) confirmed as the
   closest recent work; **arXiv:2604.19476** (LLM-filtered 10-K semantic network, S&P 500
   2011–19) is the honest-effect-size benchmark — LS Sharpe 0.742→0.820, and *asymmetric*
   (liquid→illiquid) edges carry the underreaction signal while symmetric edges revert.
3. **Analyst signals**: **arXiv:2502.20489** — LLM embeddings of sell-side report *narratives*
   predict beyond numeric rec/EPS/TP revisions, implying numeric revisions are largely priced;
   tempers C#2 expectations and would need report text we don't have.
4. **Cost-aware**: **arXiv:2605.01176** — decision-focused (SPO) learning implicitly inflates
   predictions and churns; clipping/partial adjustment fix it (Gârleanu-Pedersen vindicated
   again). **arXiv:2604.14206** — Bayesian-student distillation halves trading activity with
   no explicit turnover penalty; curiosity, not a plan change.
5. **LexiconArxiv corpus notes**: ICLR 2026 items "Predictive CVaR Q-learning" and "STABLE"
   (shift-tolerant Black-Litterman) are adjacent but don't beat what we have; corpus is
   ML-conference-weighted, so finance-journal/preprint coverage stays with WebSearch.

**Net effect on priorities: none re-ranked.** 2602.03903 becomes the named upgrade path for
the controller; 2603.13252 reinforces gate-not-size; the rest confirm existing plans.

## Reference index (sweep-verified, incremental to the 07-06 index)
Costs: Frazzini-Israel-Moskowitz (JF 2018, SSRN 3229719); Brière-Lehalle-Nefedova-Raboun
(SSRN 3380239); Novy-Marx-Velikov (RFS 2016, NBER w20721); DeMiguel-Martín-Utrera-Nogales-Uppal
(RFS 2020); Gârleanu-Pedersen (JF 2013). Revisions: Jegadeesh-Livnat (JAE 2006); Brav-Lehavy
(JF 2003); Da-Schaumburg (JFM 2011); Han-Kang-Kim (JFM 2022); Palley-Steffen-Zhang (MS 2025);
Feldman-Livnat-Zhang (JPM 2012); Martineau (CFR 2022); Jegadeesh-Kim-Krische-Lee (JF 2004);
Asquith-Mikhail-Au (JFE 2005). Links: Ali-Hirshleifer (JFE 2020); Cohen-Frazzini (JF 2008) +
replication arXiv:2301.11394; Hou (RFS 2007); Hoberg-Phillips TNIC; LLM supply-chain extraction
arXiv:2410.13051. Labels: Joubert JFDS 4(3)/4(4)/5(2); AEDL (Appl. Sci. 15(24):13204);
Kovačević (IEEE Access 2023); LambdaRankIC arXiv:2605.00501; Label-Horizon-Paradox
arXiv:2602.03395; Kang arXiv:2504.02249. Encoder: Chen-Kelly-Xiu (RFS 2025, SSRN 4416687);
arXiv:2510.15691; arXiv:2407.18103; arXiv:2606.29290; arXiv:2512.23847; arXiv:2502.21206;
arXiv:2602.00196; Lopez-Lira-Tang arXiv:2304.07619. Survivorship: arXiv:2603.19380;
fja05680/sp500; Norgate; Sharadar.
