# Trading-R1 논문 분석 요약 (구현 관점)

> **출처**: Xiao et al., *Trading-R1: Financial Trading with LLM Reasoning via Reinforcement Learning*, arXiv:2509.11420v1 (2025-09-14). Tauric Research / UCLA·UW·Stanford.
> **이 문서의 목적**: 우리 재구현(Trading-R1 on DGX Spark) Phase 0~3에서 바로 참조할 수 있도록, 논문의 방법론·수식·하이퍼파라미터·출력 포맷을 구현 가능한 수준으로 정리한다.
> **작성일**: 2026-05-25

---

## 0. TL;DR

- **무엇을**: 금융 트레이딩 추론에 특화된 LLM. **Qwen3-4B**를 백본으로, SFT + RL(GRPO)을 **3단계 easy-to-hard 커리큘럼**으로 학습.
- **출력**: 구조화된 투자 논지(XML 섹션 + 근거 인용) + **5-class 매매 시그널**(Strong Sell / Sell / Hold / Buy / Strong Buy).
- **데이터**: 자체 구축 **Tauric-TR1-DB** (100K 샘플, 14종목, 18개월, 5개 소스). **공개 안 됨.**
- **핵심 성과**: 오프더셸 LLM·추론모델 대비 risk-adjusted return↑, drawdown↓ (예: NVDA CR 8.08%, Sharpe 1.88).
- **코드/데이터/가중치**: 전부 **미공개** ([github.com/TauricResearch/Trading-R1](https://github.com/TauricResearch/Trading-R1) = README만 있는 플레이스홀더). → **재현이 아니라 재구현이 필요.**

---

## 1. 핵심 기여 4가지

1. **Tauric-TR1-DB** — 100K 금융 추론 코퍼스. 2024-01-01 ~ 2025-05-31(18개월), 14개 주요 종목, 이종 데이터(기술적·펀더멘털·뉴스·인사이더 심리·매크로) 통합. 역추론 증류 + 변동성 인지 보상 라벨링.
2. **역추론 증류(Reverse reasoning distillation)** — 프로프라이어터리 모델(o1/o3)은 최종 답만 주고 CoT를 노출 안 함. 그래서 최종 답으로부터 **추론 과정을 역으로 합성**해 SFT 지도신호를 만든다.
3. **실행 등급 결정을 위한 RL** — 매매 추천을 RL 문제로 캐스팅. 5-class 척도(Strong Buy/Buy/Hold/Sell/Strong Sell)로 라벨링, 변동성 보정 보상.
4. **Trading-R1** — 다양한 자산·시장(불/베어)에서 학습된 금융 추론 LLM.

---

## 2. 데이터 파이프라인 (Appendix S1)

### 2.1 소스별 수집 상세

| 모달리티 | 소스 | 수집 규칙 |
|---|---|---|
| **뉴스** | Finnhub company news API + **Google News 스크래핑** | 30일 lookback. 3개 시간버킷으로 분할(아래 표). 각 버킷 랜덤 샘플, 시간 역순 정렬, 시간태그 문자열. |
| **기술적(가격)** | Yahoo Finance | 거래일마다 **15일 롤링 OHLCV** 윈도우. |
| **기술적(지표)** | `stockstats` (+Yahoo) | 200일 SMA/Ichimoku는 2년 lookback. 15일 출력 윈도우, 태그(`<macd>`, `<rsi>` 등). 전체 목록은 §2.4. |
| **펀더멘털** | SimFin API + **SEC EDGAR(10-Q/10-K)** | SimFin: 분기·연간 + TTM. EDGAR: 분기 filing을 CIK로 매핑. 타깃일 이전 published만(시간 정합성). 주요 라인아이템 추출. |
| **심리** | Finnhub insider sentiment/transactions + Yahoo `upgrades_downgrades` | insider sentiment 90일(필드: `change` K/M/B/T, `mspr`). insider transactions 30일 윈도우 역행 최대 2년, 최대 25건. analyst recs 90일 trailing. |
| **매크로** | **FRED API**(인증키) | 시리즈별 2년 히스토리, **first-of-month**만 유지(월 빈도). |

**뉴스 시간버킷 (Table S1):**

| 시간대 | 날짜 범위 (t 기준) | 최대 샘플 |
|---|---|---|
| 최근 3일 | t−3 ~ t | 10 |
| 4–10일 | t−10 ~ t−4 | 20 |
| 11–30일 | t−30 ~ t−11 | 20 |

> ⚠️ **소셜미디어(트위터 등)는 최종적으로 제외**됨 (S3.1: 관측 가능한 부분이 편향돼 정보가치 낮음).

### 2.2 입력 조립 (Input Assembly)

- 거래일×종목마다 그날 이용 가능한 모든 문서·시그널을 **단일 프롬프트**로 조립.
- **앙상블링**: 기술적·심리·펀더멘털의 부분집합을 랜덤 샘플 + 순서 셔플(예: 뉴스 먼저 / 펀더멘털 먼저). → 같은 상태의 다양한 표현을 생성, **불완전 정보 하에서의 추론** 학습.
- 거래일×종목당 **약 20개 변형** 생성 → 14종목 × ~354 거래일 × 20 ≈ **100K 샘플**.

### 2.3 토큰 절약 전략

- 숫자 약어화(1000 → 1k), 과도한 긴 글 truncate, 정규식 필터링, 마크다운 stripping.
- 정적 검증 불가 소스(뉴스/소셜)는 **LLM 기반 relevance 필터**로 고정보 콘텐츠만 유지.
- 미처리 시 입력이 쉽게 80K 토큰 초과 → 처리 후 평균 **15k~23k 토큰**(SPY 3.8k ~ JNJ 43k).

### 2.4 기술적 지표 전체 목록 (Table S2)

| 카테고리 | 지표 |
|---|---|
| Moving Averages | 50 SMA, 200 SMA, 50 EMA, 10 EMA |
| MACD Family | MACD, Signal, Histogram |
| Ichimoku | Cloud, Conversion, Base, Span B |
| Momentum | RSI, KDJ(K/D/J), CCI, ROC |
| Volatility | ATR, ATR(5), Z-score(75) |
| Volume | PVO, MFI, ADX/ADXR, VWMA |
| Bollinger | Middle, Upper, Lower |

### 2.5 데이터셋 통계 (Table S3) — 14종목 유니버스

| 섹터 | 종목 |
|---|---|
| Information Technology | NVDA, MSFT, AAPL |
| Communication Services | META |
| Consumer Discretionary | AMZN, TSLA |
| Financials | BRK.B, JPM |
| Health Care | LLY, JNJ |
| Energy | XOM, CVX |
| ETF | SPY, QQQ |

기간: 2024-01-01 ~ 2025-05-31, 약 354 거래일.

---

## 3. 변동성 기반 라벨 생성 (Appendix S2, Algorithm S1)

5-class 라벨을 **결정론적으로** 생성한다(모델 불필요). 이게 RL의 검증 가능 보상이자 SFT 타깃의 근거.

### 3.1 알고리즘 (정확 의사코드)

```python
# Require: P (가격 시계열), H={3,7,15}, w={0.3,0.5,0.2}, q={0.03,0.15,0.53,0.85}
EMA = P.ewm(span=3).mean()
for tau in H:                                  # H = [3, 7, 15]
    R[tau] = (EMA - EMA.shift(tau)) / EMA.shift(tau)   # ⚠️ 아래 주의 참조
    V[tau] = R[tau].rolling(20).std()          # 20일 롤링 변동성
    S[tau] = R[tau] / V[tau]                    # Sharpe-like 정규화 시그널
WeightedSignal = 0.3*S[3] + 0.5*S[7] + 0.2*S[15]
valid = WeightedSignal.notna()
thr = [WeightedSignal[valid].quantile(qi) for qi in [0.03, 0.15, 0.53, 0.85]]
# 라벨 부여 (x = WeightedSignal[t])
#   x >= thr[3](0.85)  -> STRONG BUY
#   x >= thr[2](0.53)  -> BUY
#   x >= thr[1](0.15)  -> HOLD
#   x >= thr[0](0.03)  -> SELL
#   else               -> STRONG SELL
```

### 3.2 결과 목표 분포 (Table 2)

| Strong Buy | Buy | Hold | Sell | Strong Sell |
|---|---|---|---|---|
| 15% | 32% | 38% | 12% | 3% |

→ blue-chip 유니버스의 장기 상승 드리프트를 반영한 **불리시 스큐**(비대칭 분위 컷). 5-class는 포트폴리오 가중치로 매핑.

### 3.3 ⚠️ 재현 주의: forward vs trailing

- **본문(p8)**: "forward returns over 3, 7, 15-day periods" (전방 수익률).
- **Algorithm S1**: `EMA.shift(tau)` → pandas에서 양수 shift는 과거 값을 가져옴 → `(EMA_t − EMA_{t−τ})` = **trailing(모멘텀)**.
- **불일치.** 라벨은 "이 시점의 최적 액션 = 이후 실현 수익률 기반 정답"이어야 하므로 **`shift(-τ)`(전방)** 이 의도에 맞다.
- **우리 결정**: `shift(-τ)`로 구현(전방 수익률). 이는 **학습 라벨에만** 미래 정보가 들어가는 것(지도학습 정답) → 백테스트 룩어헤드 편향 아님. 백테스트는 입력(t까지)만 모델에 주고 예측 액션을 사용.
- Phase 0에서 두 버전 모두 구현해 분포가 Table 2와 맞는지 검증할 것.

---

## 4. 학습 파이프라인 (3단계 커리큘럼)

### 4.1 전체 구조 (Fig 1, Table 1)

3단계 **easy-to-hard 커리큘럼**. 각 단계는 *SFT(warm-start) → RFT(GRPO) → Augmentation(reject-sampling self-distill)* 의 인터리빙.

| 단계 | SFT 목적 | RFT 목적 | Augmentation |
|---|---|---|---|
| **Stage I: STRUCTURE** | 구조적 사고·데이터 조직화 | 섹션(intro/claims/table/conclusion) 체계 분석 | 명확한 구조의 케이스 self-distill |
| **Stage II: CLAIMS** | 근거 기반 추론 토대 | 의견+인용+출처 grounding, 환각 억제 | 전문적·충실한 claim 강화 |
| **Stage III: DECISION** | 투자 추천 구조 | Equity & Volatility 보정 결정 | 방향성 정답 케이스 강화 |

→ 최종 Mixture SFT로 thesis 구조·claim 포맷 강화 → **Trading-R1**.

### 4.2 SFT 세부 (§3.6)

- **백본**: Qwen3-4B (추론 최적화 prior로 수렴 가속). warm-start 없이 바로 RL 시 표면 휴리스틱에 과적합·이전 구조 망각.
- **방법**: **LoRA**. 단계별 SFT 타깃을 다르게 설계(증류가 통제 가능하므로).
- 컨텍스트: 입력 20~30k 토큰, thesis 6~8k 토큰.

### 4.3 역추론 증류 (Fig 2)

- **(a) Investment Thesis Distillation**: 구조화 입력 → o3-mini/o4-mini → 최종 추천(front-end response). 정답(변동성 라벨)과 맞으면 채택, 틀리면 reject sampling.
- **(b) Reverse Reasoning Distillation**: (추천 + 원입력) → GPT-4.1로 reasoning perspective 분해(Factor: 경쟁사/기술적/인사이더…) → GPT-4.1-nano로 각 모달리티 기여 elaborate → 프로그램적으로 stitch → 일관된 추론 trace.

> **우리 단순화(로컬 증류)**: 변동성 라벨이 이미 정답이므로 reject sampling 불필요. 로컬 교사(예: Qwen3-32B)에게 "이 입력을 근거로 *이 라벨*을 정당화하는 XML 투자 논지를 §8 포맷으로 써라"고 직접 지시 → 비용 0, 통제력↑.

---

## 5. 강화학습 (§3.7) + 보상 설계 (Appendix S4)

### 5.1 GRPO

- **알고리즘**: Group Relative Policy Optimization (PPO 변형, value model 불필요). 입력 q마다 G개 후보 샘플, 그룹 상대 advantage:

  `Â_i = (r_i − mean(r)) / std(r)`

- 목적: 클리핑된 surrogate + **KL 페널티(β)** to reference SFT 모델 π_ref.
- 자원(논문): SFT는 8×H100(96GB), RL은 8×H200(141GB).

### 5.2 보상 = 구조 + 근거 + 결정 (정확 수식)

**Aggregation**: `R_investment(x) = λ_struct·R_structure + λ_evid·R_evidence + λ_dec·R_decision` (비음수 가중치, 도메인별 조정).

#### Stage I — Structure Reward (XML 섹션)
- `S(x)` = XML 섹션 집합 (`think` 태그 제외, `conclusion` 섹션 필수). `S = |S(x)| − 1` = conclusion 제외 분석 섹션 수.
- 섹션 수 보상 (목표 **5–7개**):
  - `R_count(S) = 1`                 if 5 ≤ S ≤ 7
  - `= max(0.3, S/5 × 0.7)`          if S < 5
  - `= max(0.3, 1 − 0.15(S−7))`      if S > 7
- 섹션별 구조 요소: `R_struct(s) = 0.3·[headers] + 0.4·[bullets] + 0.2·[bold] + 0.1·[tables]`. (단어 < 50인 섹션은 0.2)
- 종합: `R_structure(x) = 0.6·R_count(S) + 0.4·mean_i R_struct(s_i)`

#### Stage II — Evidence Reward (opinion-quote-source)
- 비-conclusion 섹션의 불릿 `B(c)` 추출. 불릿 b마다: `Q(b)`=인용(이탤릭 `*quote*`), `S(b)`=출처(백틱 `` `source` ``). opinion = 첫 인용 마커 이전 텍스트. `w_op`=opinion 단어수, 최적 [15, 90]. `C(b)=1` if 인용·출처 모두 존재.
- `R_opinion(b)`:
  - `= 1`                              if 15 ≤ w_op ≤ 90 and C(b)=1
  - `= w_op/15`                        if w_op < 15 and C(b)=1
  - `= max(0.5, 1 − 0.02(w_op−90))`    if w_op > 90 and C(b)=1
  - `= min(0.3, w_op/15 × 0.3)`        if C(b)=0
- `R_bullet(b) = 0.4·R_opinion(b) + 0.35·[|Q(b)|>0] + 0.25·[|S(b)|>0]`
- 불릿 수 보상 (최적 [4,7]): `R_count^bullet(c) = 1` if 4≤|B|≤7; `=|B|/4` if <4; `=max(0.3, 1−0.1(|B|−7))` if >7.
- 섹션 점수(조화평균): `R_evidence^section(c) = 0.3·R_count^bullet(c) + 0.7 · |B(c)| / Σ_b 1/max(R_bullet(b), 0.01)`
- 종합(섹션 조화평균): `R_evidence(x) = |S_analysis(x)| / Σ_c 1/max(R_evidence^section(c), 0.01)`

#### Stage III — Decision Reward (비대칭 행렬)
- 결정 추출: 마지막 3줄의 `[[DECISION]]` 패턴 (실제 출력은 트리플 브래킷 `[[[BUY]]]` — §8 참조).
- **비대칭 보상 행렬 M** (행=예측, 열=정답), 순서 SS, S, H, B, SB:

  |  | SS | S | H | B | SB |
  |---|---|---|---|---|---|
  | **SS** | 1.00 | 0.75 | −1.25 | −2.00 | −2.25 |
  | **S** | 0.75 | 1.00 | −0.75 | −1.50 | −2.00 |
  | **H** | −1.50 | −1.00 | 1.00 | −1.00 | −1.50 |
  | **B** | −1.75 | −1.25 | −0.75 | 1.00 | 0.75 |
  | **SB** | −2.00 | −1.50 | −1.25 | 0.75 | 1.00 |

- `R_decision(d̂, d*) = M[d̂, d*] · λ_dec`. 유효 결정 없으면 `R_decision = −1.5`.
- **설계 원리**: ① 시장은 오를 때보다 빨리 떨어짐 → 거짓 불리시(예측 SB / 정답 SS = −2.25)가 거짓 베어리시(예측 SS / 정답 SB = −2.00)보다 ~12% 더 큰 페널티. ② 자본 보존(downside 우선). ③ **anti-HOLD bias**(액션이 필요한데 HOLD 예측 시 −1.0~−1.5). ④ 0.25 단위 증분.

---

## 6. R0 실패 교훈 (Appendix S5) — 우리가 반드시 피할 것

초기 버전 **Trading-R0**는 *format 보상 + outcome 보상을 단일 합성 목적으로 병합*했다가 실패. 핵심 교훈 3가지:

1. **혼합 보상 = 불안정.** format(구조)과 outcome(시장정렬) 그래디언트가 경쟁 → 모델이 구조 준수 ↔ 노이즈 추측 사이를 진동. **→ 보상은 섞지 말고 단계적으로 분리(staged).** (이것이 R1의 3단계 커리큘럼 동기.)
2. **구조 과잉 통제 = 출력 저하.** `<think>` 블록에 좁은 보상 예산·엄격 페널티 → 모델이 보상 해킹(껍데기만 완벽, 분석은 무의미). **→ 스캐폴딩은 유연하게.**
3. **빡빡한 보상 예산 = 추론 깊이 억제.** 모델이 "minimum viable output"으로 수렴. **→ 예산 완화 + 일부 스캐폴딩 유지.**

> 결론: 서로 다른 유형의 보상은 **혼합이 아니라 단계화**. 구조 스캐폴딩은 **너무 빡빡하지 않게**.

---

## 7. 평가 (§4.3, Appendix S2.2)

### 7.1 지표 공식

- **Cumulative Return**: `CR = V_N/V_0 − 1 = Π(1+r_t) − 1`
- **Sharpe (SR)**: rf = **4% 연율**(US10Y), 초과수익 `x_t = r_t − rf`, `SR_per = x̄/s_x`, 연율화 `SR_ann = √K · SR_per` (일간 K=252).
- **Hit Rate**: `HR = (1/N) Σ 1{sign(a_t) = sign(r_t)}`
- **Max Drawdown**: `MDD = max_t (1 − V_t / max_{u≤t} V_u)`

### 7.2 백테스트 셋업

- 평가 종목: AAPL, GOOGL, AMZN, SPY (결과표는 NVDA/AAPL/MSFT/AMZN/META/SPY).
- 기간: **2024-06-01 ~ 2024-08-31** (학습 제외 홀드아웃).
- 거래는 각 거래일까지 이용 가능한 정보만 사용(룩어헤드 제거, strictly causal).

### 7.3 주요 결과 (Table 3/4)

성능 위계: **SLM < RLM < LLM < Trading-SFT ≈ Trading-RFT < Trading-R1**.

| 종목 | Trading-R1 CR(%) | SR | HR(%) | MDD(%) |
|---|---|---|---|---|
| NVDA | 8.08 | 1.88(본문) / 2.72(표) | 70.0 | 3.80 |
| AAPL | 5.82 | 1.80 | 63.6 | 3.68 |
| MSFT | 2.38 | 0.87 | 60.4 | 1.90 |
| AMZN | 5.39 | 1.72 | 63.0 | 3.20 |
| SPY | 3.34 | 1.60 | 64.0 | 1.52 |

> 오프더셸 RLM(o3-mini/o4-mini/DeepSeek)은 종종 LLM보다도 저조 — 비통제 추론이 금융 분석에서 이탈. SFT가 포맷·결정 패턴을 강제하고, RFT가 시장 정렬을 점진 강화하는 게 핵심.

---

## 8. 출력 포맷 (Appendix S6, 실제 트레이스 기반) — 증류 프롬프트의 정답지

```
<think>
[계획: 데이터 정리(fundamentals/news/market/sentiment/macro), 5-7 섹션 설계,
 각 섹션 데이터 충분성 점검, 포맷 규칙 확인]
</think>

II. INVESTMENT THESIS AND ASSESSMENT

II-A. FUNDAMENTAL ANALYSIS        (← <fundamentals> 등 XML 태그 섹션)
[굵은 인트로 문장]
○ [의견 문장]. Supporting evidence: "[정확한 인용]" SOURCES: [소스명], [날짜]
○ [의견 문장]. Supporting evidence: "[인용]" SOURCES: [소스명], [날짜]
... (섹션당 불릿 4~7개)
[표 — 예: Financial Metric | Value | Change ...]
[굵은 요약 문장]

II-B. TECHNICAL ANALYSIS
...
(섹션 총 5~7개: fundamentals, balance_sheet, technical, news, valuation,
 risk_assessment, analyst_coverage, macro 중 선택 + conclusion)

III. TRADING-R1 DECISION
[[[BUY]]]            (← 트리플 브래킷, 5-class 중 하나)
```

**포맷 규칙(보상과 직결)**: 섹션 5–7개 + 필수 conclusion / 섹션당 불릿 4–7개 / 불릿 = 의견(15–90단어) + `*인용*` + `` `출처` `` / 표·헤더·굵게 활용 / 결정은 마지막 `[[[DECISION]]]`.

---

## 9. 우리 재구현 관점 핵심 노트

1. **미공개 → 재구현.** 코드·데이터·가중치 전부 없음. 데이터 fetcher는 Tauric의 공개 **TradingAgents** 레포 재활용 검토(Finnhub/SimFin/Google News/stockstats — 확인 필요).
2. **하드웨어 매핑 (DGX Spark / GB10).** 논문 8×H100/H200 → 우리는 노드당 128GB 통합메모리 GB10 ×2. Qwen3-4B는 작아서 메모리 비병목; 병목은 **긴 컨텍스트 연산 + RL 롤아웃 속도**. GB10에서 **vLLM+Ray·FP8 깨짐** → RL은 단일노드 vLLM(`--enforce-eager`, BF16/AWQ) 롤아웃 + TRL GRPOTrainer(HTTP), Ray/verl/OpenRLHF 회피.
3. **증류는 로컬 교사.** OpenAI API 대신 로컬 강모델(Qwen3-32B급). 변동성 라벨이 정답이므로 reject sampling 생략, §8 포맷으로 직접 thesis 생성.
4. **라벨링 forward/trailing 결정** (§3.3) — Phase 0에서 분포(Table 2) 일치로 검증.
5. **보상은 단계화, 스캐폴딩은 유연하게** (§6 R0 교훈) — Phase 2 보상 구현의 제1원칙.
6. **검증 정의** — 데이터 미공개로 절대수치 재현 불가. "검증" = 베이스라인 대비 상대 개선 + 논문 추세(위계)와 방향 일치.
7. **컨텍스트 단축** — 초기 Phase는 8~12k로 truncate해 연산 절약, 연산 여유 시 확장.

---

## 부록: 빠른 참조 상수

| 항목 | 값 |
|---|---|
| 백본 | Qwen3-4B |
| 라벨 horizon H | {3, 7, 15}일 |
| 시그널 가중 w | {0.3, 0.5, 0.2} |
| 분위 임계 q | {0.03, 0.15, 0.53, 0.85} |
| EMA span | 3 |
| 롤링 변동성 윈도우 | 20 |
| 목표 분포 (SB/B/H/S/SS) | 15/32/38/12/3 % |
| 섹션 수 최적 | 5–7 (+ conclusion) |
| 섹션당 불릿 최적 | 4–7 |
| opinion 단어수 최적 | 15–90 |
| rf (Sharpe) | 4% 연율 |
| 연율화 K | 252 (일간) |
| 학습 기간 | 2024-01-01 ~ 2025-05-31 |
| 백테스트 홀드아웃 | 2024-06-01 ~ 2024-08-31 |
| 데이터 규모 | ~100K (14종목 × ~354일 × ~20변형) |
