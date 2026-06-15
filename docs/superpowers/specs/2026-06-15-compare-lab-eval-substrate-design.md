# 설계: compare_lab — 공통 평가 substrate + prompt-only LLM 비교 (서브프로젝트 1)

> **상태**: 작성됨 (2026-06-15) · 사용자 리뷰 대기
> **방식**: 접근 C — 신규 `compare_lab/` 패키지, qf-lib 백테스트 + 논문 메트릭 레이어
> **참조**:
> - [`2026-05-25-trading-r1-dgx-spark-design.md`](2026-05-25-trading-r1-dgx-spark-design.md) — #1 학습 파이프라인 (서브프로젝트 2)
> - [`docs/trading-r1-paper-summary.md`](../../trading-r1-paper-summary.md) — 논문 정밀 분석 (지표·라벨·포맷)
> - `qf-lib-harness/alpha_lab/` — #3 price-only 팩터 하니스 (FROZEN, 읽기 재사용)

---

## 0. 한 줄 요약

세 트레이딩 접근 — **#3 퀀트 팩터**(작동 중), **#2 prompt-only 오픈소스 LLM**, **#1 학습된 Trading-R1**(추후) — 을 **하나의 qf-lib 평가판** 위에서 동일 유니버스·기간·look-ahead 규칙·지표(CR/SR/HR/MDD)로 비교하는 substrate를 만든다. 서브프로젝트 1은 substrate + #2까지 구현해 **baseline vs #3 vs #2** 3자 비교를 산출한다. #1은 같은 판에 4번째 행으로 나중에 꽂는다.

---

## 1. 목표와 범위

### 1.1 목표
1. **공통 평가 substrate** — 모델/전략 종류와 무관하게 "신호 → qf-lib 백테스트 → 논문 지표"를 측정하는 단일 루프.
2. **prompt-only LLM 비교(#2)** — 학습 없이 최신 4B급 오픈소스 LLM의 제로샷 트레이딩 능력을 같은 판에서 측정.
3. **확장점 확보** — 서브프로젝트 2(DGX Spark 학습)의 모델이 `LLMProvider` 한 구현으로 substrate에 무수정 합류.

### 1.2 #1 스펙과의 관계 (중복 방지)
2026-05-25 DGX 스펙의 **Phase 0 `backtest`(모델 독립) + 베이스라인(제로샷)** 을 이 substrate가 **실제로 구현**한다. 즉:
- DGX 스펙의 `backtest` 유닛 = 본 스펙의 `compare_lab.backtest` + `compare_lab.metrics`.
- DGX 스펙 Phase 0 "베이스라인: 학습 전 제로샷 시그널→백테스트" = 본 스펙의 `LLMProvider`(#2).
- DGX 스펙의 `data/assembler`(풀 모달리티) ⊃ 본 스펙의 `MarketSnapshotBuilder`(가격+기술지표 = MVP 축소판, **동일 인터페이스의 시작점**).

→ 두 작업은 **백테스터·스냅샷 인터페이스를 공유**한다. 서브프로젝트 2는 모달리티(뉴스/펀더멘털/심리/매크로)와 학습을 추가할 뿐, 평가판을 새로 만들지 않는다.

### 1.3 명시적 비범위 (YAGNI)
- **모델 학습 없음** (SFT/GRPO는 서브프로젝트 2). 서브프로젝트 1은 추론·평가만.
- **뉴스/펀더멘털/심리/매크로 모달리티 없음** — MVP 스냅샷은 가격+기술지표만. (서브프로젝트 2 또는 후속 확장)
- **숏/레버리지 없음** — 롱온리(보유 또는 flat).
- **등급별 차등가중 보류** — MVP는 5-class를 보유/flat 이진 매핑. 차등가중(SB=2×B)은 후속.
- **한국 시장 없음** — 미국만 (서브프로젝트 2 Phase 3 영역).

---

## 2. 제약 조건

| 제약 | 함의 |
|---|---|
| `alpha_lab` 코어는 FROZEN(인간 전용) | 수정 금지. `compare_lab` 신규 패키지로, `alpha_lab.core` 로더는 **읽기**만 재사용. qf-lib 브리지는 패턴 **복제**(import 대신)로 디커플링. |
| 데이터: S&P500 503종목, 2015-01-02~2026-06-12 (이미 보유) | 논문 14종목+SPY/QQQ 전부 포함. 추가 다운로드 불필요. |
| GB10: vLLM+Ray·FP8 깨짐 | LLM 서빙은 **단일노드 vLLM `--enforce-eager` BF16** (OpenAI-호환). Ray/멀티노드 회피. (`dgx-spark-gpu`/요약문서 §9.2) |
| 모든 GPU 잡은 sparkq 경유 | vLLM 서버·추론 잡을 sparkq로 제출. 직접 GPU 점유 금지. |
| LLM은 비결정적 | (snapshot hash → 응답) 디스크 캐시로 재현성·재시작·비용 통제. |

---

## 3. 아키텍처

```
prices.parquet
   │ (alpha_lab.core 로더 — 읽기 재사용)
   ▼
[각 SignalProvider] ──────────────► 목표 가중치 행렬 (date × ticker, ≥0, 합=1)
   ├ EqualWeightProvider   (순진한 시장 baseline)
   ├ MomentumProvider      (#3: 12-1, top-N 동일가중)
   └ LLMProvider           (#2: snapshot → vLLM → 5-class → weight)
                                   │
                                   ▼
                          backtest.py  (qf-lib 이벤트 백테스트, 수수료 반영)
                                   │  일간 simple returns
                                   ▼
                          metrics.py   (CR / SR rf=4% / HR / MDD)
                                   │
                                   ▼
                          report.py    (비교 표 + plotly HTML 에쿼티 커브)
```

**공통 표현 = 목표 가중치 행렬 `DataFrame[date × ticker]`.** per-ticker(각 종목 long/flat 독립)와 cross-sectional(top-N 랭킹)이 모두 이 행렬의 특수 케이스:
- cross-sectional top-N 동일가중 → top-N에 1/N, 나머지 0.
- per-ticker 5-class → 보유 종목 집합에 동일가중(또는 후속: 등급별 차등).

### 3.1 패키지 레이아웃

```
qf-lib-harness/compare_lab/
├── config.py          # 유니버스·기간·rebal·rf·경로 상수
├── snapshot.py        # MarketSnapshotBuilder (per-ticker, as_of, 가격+기술지표 → 텍스트)
├── providers/
│   ├── base.py        # SignalProvider ABC
│   ├── equal_weight.py
│   ├── momentum.py
│   └── llm.py
├── llm_client.py      # vLLM OpenAI-호환 클라이언트 + 응답 캐시
├── backtest.py        # 가중치 행렬 → qf-lib 백테스트 → 일간 수익률
├── metrics.py         # CR / SR / HR / MDD
├── report.py          # 비교 표 + HTML
├── run_comparison.py  # CLI 오케스트레이터
└── tests/
```

### 3.2 컴포넌트 경계

| 유닛 | 책임 | 인터페이스(I/O) | 의존 |
|---|---|---|---|
| `snapshot` | per-ticker 스냅샷 직렬화 | `(ticker, as_of_date) → prompt str` | pandas, stockstats, alpha_lab.core 로더 |
| `providers/base` | 신호 계약 | `weights(ctx, rebal_dates) → DataFrame[date×ticker]` | — |
| `providers/equal_weight` | 시장 baseline | ↑ 구현 | — |
| `providers/momentum` | 퀀트 baseline(#3) | ↑ 구현 | pandas |
| `providers/llm` | LLM 신호(#2) | ↑ 구현 | snapshot, llm_client |
| `llm_client` | vLLM 호출 + 캐시 | `prompt → completion str` | requests/openai, 디스크 캐시 |
| `backtest` | 신호→백테스트 | `weights → 일간 returns Series` | qf-lib |
| `metrics` | 지표 계산(순수함수) | `returns + 결정 → {CR,SR,HR,MDD}` | numpy (모델 독립) |
| `report` | 비교 산출물 | `{provider→metrics, returns} → 표+HTML` | plotly |

> `metrics`·`backtest`·`snapshot`은 **모델 독립 순수 로직** → 단독 테스트 가능.

---

## 4. 컴포넌트 상세

### 4.1 MarketSnapshotBuilder (`snapshot.py`)
- 입력: `ticker`, `as_of_date`. 출력: LLM 프롬프트용 텍스트 1개.
- 내용(MVP): 15일 출력 윈도우의 OHLCV + 기술지표(논문 Table S2: 50/200 SMA, 50/10 EMA, MACD(+signal/hist), Ichimoku, RSI, KDJ, CCI, ROC, ATR/ATR(5)/Zscore(75), PVO, MFI, ADX/ADXR, VWMA, Bollinger). `stockstats`로 계산. 200일 SMA 등은 2년 lookback 입력 허용.
- **엄격 인과성**: `as_of_date` 초과 바를 물리적 슬라이스 제거 후 지표 계산.
- **snapshot hash**: `sha1(ticker || as_of || 직렬화내용)` → 캐시 키 + 재현성.
- 토큰 절약(요약문서 §2.3): 숫자 약어화(1000→1k), 과도한 소수 truncate.

### 4.2 SignalProvider (`providers/`)
- ABC: `weights(ctx, rebal_dates) -> pd.DataFrame`  (index=rebal_dates, columns=universe, ≥0, 행 합=1; 전부 0 행 = 전액 현금).
- `EqualWeightProvider`: 매 rebal일 유니버스 동일가중.
- `MomentumProvider`: 12-1 모멘텀(`pct_change(252).shift(21)`) 랭킹 → top-N 동일가중. (#3을 이 substrate 표현으로 옮긴 것.)
- `LLMProvider`: 각 (rebal일, 종목) → snapshot → `llm_client` → 5-class 파싱 → 매핑. MVP 매핑: `{STRONG_BUY, BUY}`→보유집합, `{HOLD, SELL, STRONG_SELL}`→제외 → 보유집합 동일가중.

### 4.3 llm_client (`llm_client.py`)
- DGX Spark 단일노드 vLLM, OpenAI-호환 `/v1/chat/completions`.
- 모델: 구현 시점 최신 4B급 instruct/reasoning(검증 폴백 = Qwen3-4B). 4B급 유지로 #1 백본과 apples-to-apples.
- 프롬프트가 마지막 줄에 `[[[STRONG_BUY|BUY|HOLD|SELL|STRONG_SELL]]]` 트리플 브래킷 강제(논문 §8). 짧은 rationale 동반.
- **(snapshot hash → 응답) 디스크 캐시**: 동일 입력 동일 결과 보장, 재시작 견딤, 비용 절감. 배치 호출.
- 서버 잡·추론 잡 모두 sparkq로 제출.

### 4.4 backtest (`backtest.py`)
- 가중치 행렬을 qf-lib 이벤트 백테스트로. `alpha_lab/pipeline.py`의 브리지 패턴(`PrecomputedSignal...AlphaModel` + `FixedPortfolioPercentagePositionSizer` + `IBCommissionModel` + `_weasyprint_stub` 선import)을 **compare_lab에 얇게 복제**(FROZEN import 회피·디커플링).
- 멤버십(보유=LONG, 외=OUT) + 포지션 사이즈 = 1/보유수. 일간 simple returns 반환.

### 4.5 metrics (`metrics.py`) — 논문 §7.1
- `CR = Π(1+r_t) − 1`
- `SR = √252 · mean(r−rf_daily)/std(r−rf_daily)`, rf=4% 연율.
- `HR = mean(1{sign(decision_t)=sign(realized_t)})` — per-ticker·포트폴리오.
- `MDD = max_t(1 − V_t/max_{u≤t}V_u)`.
- 표본 < 30 → NaN (alpha_lab.sharpe 규약과 일치).

### 4.6 report / run_comparison
- `run_comparison.py`: provider 목록 → 각 weights → backtest → metrics → 비교 표(논문 위계 스타일) + plotly HTML 에쿼티 커브. OOS 전구간 + 논문 3개월 슬라이스 둘 다 출력.

---

## 5. 평가 설정 (`config.py`)

| 항목 | 값 | 근거 |
|---|---|---|
| 유니버스 | NVDA, MSFT, AAPL, META, AMZN, TSLA, BRK-B, JPM, LLY, JNJ, XOM, CVX, SPY, QQQ | 논문 Table S3 (14) + ETF. 전부 보유 데이터에 존재 |
| OOS 기간 | 2024-01-02 ~ 2026-04-01 | 학습 없으니 전구간 OOS. 논문 3개월(2024-06~08) 슬라이스도 부가 |
| rebal | 주간 (W-FRI) | LLM 호출 비용 ↔ 해상도 |
| 방향 | 롱온리 | MVP 매핑과 일치 |
| rf | 4% 연율 | 논문 §7.1 |
| 수수료 | qf-lib IBCommissionModel | 기존 브리지와 동일 |
| top-N (momentum) | 5 (14종목 유니버스 기준) | 소형 유니버스에 맞춤 |

---

## 6. Look-ahead 통제 (제1원칙)

1. **스냅샷**: `MarketSnapshotBuilder`가 `as_of_date` 초과 바 제거 → 단위테스트로 누설 0 검증.
2. **타이밍**: rebal일 t 결정은 `as_of=t`(종가) 스냅샷 → 포지션은 **다음 거래일** 진입(qf-lib 강제).
3. **라벨 없음**: 서브프로젝트 1엔 학습 라벨 없음 → 미래 수익률은 *지표 계산*에만, 신호엔 절대 미사용.
4. snapshot hash가 (입력, as_of) 고정 → 재현 보장.

---

## 7. 에러 처리

| 상황 | 처리 |
|---|---|
| LLM 5-class 파싱 실패 | 기본 HOLD(flat) + 경고·카운트 (조용한 실패 금지) |
| 종목 데이터 결측(특정일) | 그 rebal일 해당 종목 제외, 나머지 정규화 |
| vLLM 서버 다운 | 즉시 실패(fail fast), 캐시 부분 보존 |
| 지표 표본 부족 | NaN 반환 (≥30 obs 규약) |
| 캐시 손상 | 해당 키만 무효화·재계산 |

---

## 8. 테스트

- **단위**: ① 스냅샷 as_of 누설 0 ② 5-class→weight 매핑 ③ 메트릭 공식(손계산 일치) ④ provider 결정성(캐시 hit 동일).
- **통합 스모크**: 2종목 × 짧은 윈도우 × **모의 LLM**(고정 응답) → backtest→report 크래시 없이 end-to-end.
- **재현성**: 같은 캐시 2회 실행 → 동일 메트릭.

---

## 9. 검증 정의

데이터 미공개 전제(요약문서 §7과 동일 철학):
1. **상대 비교** — baseline/#3/#2의 CR/SR/HR/MDD를 같은 판에서 산출, risk-adjusted(Sharpe) 위주 해석.
2. **방향 일치** — 추후 #1 합류 시 논문 위계(LLM < SFT ≤ RFT < R1) 방향 재현 여부 확인.
3. **인프라 검증 우선** — "돈 번다"가 아니라 "엄밀히·재현 가능하게 측정된다"가 산출물 (로드맵 원칙).

---

## 10. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| 가격+기술지표만 본 LLM이 약함(뉴스 근거 없음) | MVP는 *비교 백본 검증*이 목표. 모달리티는 서브프로젝트 2에서 추가. 약하면 그것도 유효한 결과. |
| qf-lib 브리지 복제가 alpha_lab과 어긋남 | baseline을 alpha_lab 결과와 교차검증(동일 모멘텀→유사 수치)으로 sanity check. |
| 최신 4B 모델의 GB10 서빙 호환성 | Qwen3-4B 폴백 확정. DGX 스펙 Phase 0 호환성 스파이크와 공유. |
| LLM 호출 비용/시간 | 주간 rebal + 응답 캐시. 14종목×~120주 = 일회성 ~1.7k 호출. |
| rebal 주기·top-N 등 임의 상수 | config 단일지점. 민감도는 후속. |

---

## 11. 산출물 & 다음 단계

**산출물**: `run_comparison.py` 1회 실행 → baseline(동일가중) vs #3(모멘텀) vs #2(prompt-only LLM)의 CR/SR/HR/MDD 비교 표 + HTML 에쿼티 커브 + 응답 캐시.

**다음**: 이 스펙 승인 후 **구현 계획(writing-plans)** 작성. 구현 순서 제안: config/snapshot/metrics(순수로직, 테스트 먼저) → backtest(브리지) → equal/momentum provider → llm_client+llm provider → report/CLI. 서브프로젝트 2(#1 학습)는 별도 spec→plan 사이클로, 완성 모델을 `LLMProvider`로 본 substrate에 합류.
