# Data QC Rubric — trading-r1-qflib

> **목적:** "데이터를 다 받았다"고 가정한 뒤, 그 데이터가 백테스트/학습 인풋으로
> 쓸 만한지 판정하는 현실적 검수 루브릭. 근거: [`DATA_REQUIREMENTS.md`](DATA_REQUIREMENTS.md),
> [`trading-r1-paper-summary.md`](trading-r1-paper-summary.md) §2, `compare_lab/snapshot.py`.
>
> **통합 노트 (2026-06-22):** 데이터는 `data/qflib_data_store/`에 있고, `validate_data.py`는
> 누출 수정본 `*_pit.parquet`(`compare_lab/{macro,insider,fundamentals}_pit.py`)을 우선 검증한다.
> 현재 전 모달리티 **all-hard-gates PASS, 가중종합 98.6** (G3의 SPY/QQQ는 이제 포함됨;
> macro `release_date` 누출은 새 `G2_macro_release_lag` 게이트가 잡는다 — [`DATA_STORE.md`](DATA_STORE.md)).

## 0. 설계 원칙

일반 데이터품질 6축(DAMA: 완전성·정확성·일관성·적시성·유효성·유일성)만으로는 부족하다.
이 프로젝트는 **세 개의 도메인 특화 축**이 일반 6축보다 우선한다.

- **(A) Point-in-Time 무결성** — look-ahead가 한 건이라도 새면 백테스트 전체가 무효. *점수*가 아니라 **하드 게이트**.
- **(B) Quotability(인용가능성)** — Stage II evidence reward는 "원문 인용+출처명+날짜"가 있어야만 보상. 사전 스코어링된 숫자(sentiment=0.7)는 학습에 못 씀.
- **(C) 입력형태 계약(Input-Shape Contract)** — 파이프라인(`snapshot.py`/프롬프트 빌더)이 실제로 읽는 **스키마·구조**가 안 맞으면 내용이 아무리 깨끗해도 인풋으로 못 들어감. ← **이번에 추가된, 가장 중요한 축.**

그래서 루브릭은 **2단 구조**: ① 통과 못 하면 즉시 리젝하는 Hard Gate ② 통과분에 매기는 Scored Quality.

---

## 1. HARD GATES (하나라도 fail → 데이터 반려, 점수 매기지 않음)

### G1. PIT 타임스탬프 존재·유효성 (모든 비가격 항목)
- 모든 record에 `published_at` 필수 — news=기사 발행시각, fundamentals=SEC filing date(≠fiscal period end), insider/analyst=disclosure date.
- 검사: `published_at` null 비율 = **0%**. null이면 PIT 필터 불가 → 사용 불가.

### G2. 미래누수(look-ahead) 제로
- 임의 trading day t에 대해 "as-of t 스냅샷"을 뽑았을 때 `published_at > t`인 레코드가 섞이면 fail.
- 특히 fundamentals: filing date 대신 period-end로 라벨링하면 평균 30~45일 누수 → 표본검사 필수.
- 검사: 각 모달리티 랜덤 200건의 `published_at`이 fiscal/event 날짜보다 **이후**인지 확인.

### G3. 가격=라벨 무결성 (라벨의 ground truth)
- 5-class 라벨과 RL decision reward가 **price EMA 수익률(Algorithm S1)에서 결정론적으로 파생** → 가격이 더러우면 정답이 더러워짐.
- 필수: split/dividend **조정완료(adjusted)** OHLCV, 14종목 전 기간(2024-01~최신) 결측 0, **SPY/QQQ 포함**(현재 누락).
- 검사: (a) `low ≤ open,close ≤ high` 위반 0건, (b) 인접일 |log-return| > 0.5 점프 = split 미조정 의심 전수 플래그, (c) 거래일 캘린더 대비 결측 갭 0.

### G4. 텍스트 원문 보존 (news/fundamentals/sentiment)
- 단일 점수가 아니라 **실제 snippet 텍스트 + 출처명 + 날짜** 3종 세트. 헤드라인만 있고 출처/날짜 없으면 evidence reward 학습 불가 → fail.

### G5. 입력형태 계약 준수 (Input-Shape Contract) ★ 최우선
> 파이프라인이 실제로 소비하는 형태. 근거: `DATA_REQUIREMENTS.md` §Preferred delivery format + `snapshot.py`.

- **(a) Delivery 스키마:** 레코드 1행 = **(ticker, date, modality, item)** 단위, 각 행 `published_at` 보유. 포맷 = **Parquet 또는 JSONL**(raw API dump 허용 시 timestamp 보존 필수). 소스별 license/redistribution 노트 동봉.
- **(b) 조립 스냅샷 = 1 prompt / (ticker, 1 trading day):** 5모달리티(가격·기술지표·뉴스·펀더·센티·매크로)를 1개 컨텍스트로 합침.
- **(c) 가격 스냅샷 strict-causal & 재현:** `date > as_of` 행은 **물리적으로 제거 후** 지표 계산, trailing 15봉 OHLCV + 최신 indicator, **`snapshot_hash` 재현 가능**.
- **(d) 기술지표 키 고정:** stockstats 스펙(`close_50_sma`,`close_200_sma`,`close_10_ema`,`macd/macds/macdh`,`rsi_14`,`kdjk`,`cci`,`close_10_roc`,`atr`,`boll/boll_ub/boll_lb`,`dx`,`mfi`). `roc` 무효키 → `close_10_roc` 사용.
- **(e) 뉴스 시간버킷 구조:** 각 trading day마다 30일 lookback을 **3버킷**(`t-3..t / t-10..t-4 / t-30..t-11`)으로 정렬. 버킷 누락/미정렬 = 형태 위반.
- **(f) 라벨 형태:** **5-class**(Strong Sell…Strong Buy), price EMA 수익률(Algo S1) volatility-adjusted로 결정론 파생 → 별도 제공이 아니라 **가격에서 재현 가능**해야 함.
- 하나라도 깨지면 파이프라인이 인풋을 못 읽음 → **fail**.

---

## 2. SCORED QUALITY (게이트 통과분, 모달리티별 0~100)

| 축 | 가격 | 뉴스 | 펀더멘털 | 센티먼트 | 매크로 |
|---|---|---|---|---|---|
| **완전성** | 종목×거래일 셀 채움률 ≥99.5% | 거래일 중 뉴스 0건인 날 비율, 14종목 균일성 | 분기마다 IS/BS/CF 3종 다 있나 | insider/analyst 커버리지 | FRED 시리즈 결측 |
| **정확성** | 2nd 소스(stooq) 크로스체크 오차 | 헤드라인-본문 출처 일치 | SEC EDGAR 원본 line-item 대조 | mspr/change 범위검증 | 발표값 vs 개정값 |
| **일관성** | 티커 심볼/조정방식 통일 | 중복기사 dedup, 인코딩 | 단위(천/백만) 통일, TTM 합산 검증 | 필드 스키마 일관 | first-of-month 정렬 |
| **적시성** | 최신일 = 직전거래일 | 30일 lookback 3버킷 채워짐 | 최신 분기 filing 반영 지연 | 90일 trailing 신선도 | 2년 history |
| **유일성** | (ticker,date) PK 중복 0 | URL/제목 해시 중복률 | (ticker,filing) 중복 0 | event 중복 | series×date 중복 |
| **형태적합성** ★ | OHLCV 컬럼/dtype, adj_close 존재 | 3버킷 태그·snippet+source+date 필드 | line-item 단위·TTM·fiscal_period | 이벤트 스키마/disclosure date | (series,date) 정렬·dtype |

**모달리티 가중치(라벨 영향도 기준):** 가격 0.35 / 뉴스 0.20 / 펀더 0.20 / 센티 0.15 / 매크로 0.10.

---

## 3. 합격 기준

- **Hard Gate 5개(G1~G5) 전부 PASS**가 절대조건. 하나라도 fail이면 종합점수 N/A·반려.
- 통과 시 가중종합 **≥85 = 학습/백테스트 투입 가능**, 70~85 = 조건부(결측구간 명시 후 제한사용), <70 = 재수급.
- **킬스위치:** 백테스트 윈도우 내 PIT 위반(G2)이 **1건이라도** 발견되면 점수 무관 전량 리젝. "label quality is the whole game"이라 여기만큼은 타협 없음.

---

## 4. 자동화 — `validate_data.py`

코드로 자동검증 가능:
1. **스키마/PK/null 체크** — G1, G5(a) PK=(ticker,date,modality,item)+`published_at`, Parquet/JSONL 포맷.
2. **거래일 캘린더 대비 결측맵** — G3(c), 완전성.
3. **OHLC 논리·split 점프 플래그** — G3(a)(b).
4. **`as_of(t)` 샘플링 미래누수 스캔** — G2.
5. **모달리티별 커버리지 히트맵** — 완전성/적시성.
6. **형태 계약 체크 (G5)** — ① 뉴스 3버킷 존재·정렬 ② stockstats 지표 키 일치 ③ 스냅샷 strict-causal 재실행 후 `snapshot_hash` 재현 일치 ④ 5-class 라벨을 price EMA(Algo S1)로 재계산해 제공 라벨과 대조.

정확성(2nd 소스 대조)·일관성 일부만 수동/표본.
