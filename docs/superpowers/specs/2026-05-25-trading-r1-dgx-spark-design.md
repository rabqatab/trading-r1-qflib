# 설계: Trading-R1 on DGX Spark (US→KR)

> **상태**: 승인됨 (2026-05-25) · **방식**: MVP→점진 심화 (Approach B)
> **참조**: [`docs/trading-r1-paper-summary.md`](../../trading-r1-paper-summary.md) — 논문 정밀 분석
> **하드웨어**: DGX Spark 2노드 (GB10, 노드당 128GB 통합메모리)

---

## 1. 목표와 범위

### 1.1 목표 (3중)
1. **학습/업스킬링** — 금융 RL LLM 파이프라인(데이터→증류→SFT→RFT→백테스트)을 직접 구현하며 체득.
2. **재현/검증** — Trading-R1 방법론과 결과 추세를 우리 환경에서 재현·검증.
3. **실전 배포** — 최종적으로 회사 리서치/실거래에 쓸 시그널 모델로 발전.

→ 셋을 동시 충족하기 위해 **단계마다 작동하는 산출물 + 측정 가능한 지표**를 남기는 점진적 접근(B)을 채택.

### 1.2 시장 범위
**미국 우선 → 이후 한국.** Phase 0~2는 미국(논문 재현 용이), Phase 3에서 한국(KRX/DART) 이식.

### 1.3 명시적 비범위 (YAGNI)
- **nanochat 생략** (사용자 결정). 논문은 Qwen3-4B 파인튜닝이며 nanochat(from-scratch)과 스택이 겹치지 않음.
- 고빈도/실시간 트레이딩 제외(논문도 medium-term ~1주 보유).
- 절대 성능수치의 정확 재현 목표 아님(데이터 미공개) — §7 검증 정의 참조.

---

## 2. 제약 조건 (설계를 강제하는 사실)

| 제약 | 함의 |
|---|---|
| **코드·데이터·가중치 전부 미공개** | 다운로드 백테스트 불가 → 파이프라인 전체 재구현 |
| **GB10: vLLM+Ray 깨짐(멀티노드), FP8 커널 없음** | RL 롤아웃은 단일노드 vLLM(`--enforce-eager`, BF16/AWQ) + TRL(HTTP). Ray/verl/OpenRLHF 멀티노드 회피 |
| **GB10: 통합메모리 128GB, 모델은 4B(작음)** | 메모리 비병목. 병목 = 긴 컨텍스트 연산 + 롤아웃 속도 |
| **2노드 = TP 무의미(모델 작음)** | Node2 = 교사/롤아웃 서버 또는 병렬 실험(CONFIG_SLICE) |
| **로컬 증류(API 비용 0 결정)** | 교사 = 로컬 강모델(Qwen3-32B급), reject sampling 생략 |
| **모든 GPU 작업은 sparkq 큐 경유** | 학습/서빙/증류 잡을 sparkq로 제출·모니터 |

**노드**: Node1 = `alphabridge@192.168.200.12` (spark-0), Node2 = `nvidia@192.168.200.13` (gx10-3d56, SSH 키).

---

## 3. 아키텍처

```
                 ┌─────────────────────────── 공통 인프라 ───────────────────────────┐
                 │  중앙 config · W&B 트래킹(ml-convention) · uv env · sparkq 잡 제출   │
                 └──────────────────────────────────────────────────────────────────┘

[데이터층]  fetchers (Finnhub/SimFin/EDGAR/FRED/stockstats/news)
   │          └─ 모달리티별 수집 → 거래일×종목 스냅샷 조립 → 토큰절약 → 변형 생성
   ▼
[라벨링]   변동성 기반 5-class (Algorithm S1, 결정론적)  ──┐
   │                                                       │ (정답 라벨)
   ▼                                                       ▼
[증류]     교사 Qwen3-32B (Node2 vLLM) : 입력+정답라벨 → XML 투자논지 trace
   │
   ▼
[학습]     Qwen3.5-4B-Base + LoRA   (GB10 호환성 검증 통과 시; 미통과 시 Qwen3-4B 폴백)
   │   Phase1: SFT(단일)      Phase2: 3단계 커리큘럼(SFT→GRPO→augment)
   ▼
[평가]     qf-lib 이벤트 백테스터 : 5-class→포트폴리오 가중 → CR/Sharpe/HR/MDD 리포트
```

### 3.1 컴포넌트 경계 (각 유닛: 무엇을/어떻게 쓰나/의존)

| 유닛 | 책임 | 인터페이스(I/O) | 의존 |
|---|---|---|---|
| `data/fetchers` | 소스별 원시 데이터 수집 | `(ticker, date) → modality dict` | Finnhub/SimFin/EDGAR/FRED/yfinance/stockstats |
| `data/assembler` | 스냅샷 조립·토큰절약·변형 | `modality dicts → prompt str (변형 N개)` | fetchers |
| `labeling` | 5-class 라벨 생성 | `price series → label per (ticker,date)` | pandas (모델 무관) |
| `distill` | 교사로 thesis trace 생성 | `(prompt, label) → XML thesis str` | 교사 vLLM 엔드포인트 |
| `train/sft` | LoRA SFT | `(prompt→thesis) 데이터셋 → adapter` | TRL/PEFT |
| `train/rft` | GRPO RFT | `정책+ref+롤아웃+보상 → adapter` | TRL GRPOTrainer + vLLM 서버 |
| `reward` | 구조/근거/결정 보상 | `completion str → scalar` | 라벨, 정규식/파서 |
| `backtest` | 시그널→백테스트→리포트 | `signal series → metrics` | qf-lib |
| `infer` | 모델→시그널 | `prompt → 5-class + thesis` | vLLM |

> 라벨링·백테스트는 **모델 독립**(순수 함수) → 단독 테스트 가능. 보상 함수도 completion만 받는 순수 함수 → 단위 테스트 용이.

---

## 4. Phase 분해

### Phase 0 — 토대 + 평가 하네스 (먼저 리스크 제거)
**목적**: 끝에서 끝까지 도는 측정 루프 확보. 모델을 바꿀 때마다 즉시 metric으로 평가 가능하게.

- repo 스캐폴딩, `uv` env, 중앙 config, W&B 연동
- **백본 호환성 스파이크 (GB10)** — `Qwen3.5-4B-Base`를 GB10에서 ① transformers(BF16) forward/generate ② vLLM(nightly) 단일노드 서빙(`--enforce-eager`) ③ 소형 LoRA 1-step backward 검증. **관건 = Gated DeltaNet(선형어텐션) 커스텀 커널의 SM121 지원**. 미통과 시 `Qwen3-4B` 폴백 확정.
- `data/fetchers` + `assembler` — **TradingAgents 레포 재활용 검증**, 소규모 유니버스(AAPL/NVDA/SPY, 수개월)
- `labeling` — Algorithm S1 구현, **forward/trailing 두 버전** → 분포가 Table 2(15/32/38/12/3)와 맞는지 유닛테스트
- `backtest` — qf-lib 이벤트 전략(5-class→가중), Buy&Hold·랜덤으로 sanity check
- **베이스라인**: 학습 전 Qwen3-4B 제로샷 시그널 → 백테스트 → "출발점 수치"

**산출물**: 데이터→라벨→(제로샷)시그널→백테스트 루프 작동, 베이스라인 metric W&B 기록.
**완료 기준**: 라벨 분포 검증 통과 + 베이스라인 CR/Sharpe/HR/MDD 리포트 생성.

### Phase 1 — MVP SFT (단일 단계)
**목적**: 데이터→증류→SFT→백테스트 전체를 소규모로 완주, 학습 모델이 베이스라인을 넘는지 확인.

- 교사 서빙: Qwen3-32B-AWQ, 단일노드 vLLM(`--enforce-eager`) on Node2 (sparkq)
- `distill` — 입력+정답라벨 → §8 포맷 XML thesis trace 생성
- `train/sft` — **`Qwen3.5-4B-Base`** + LoRA(TRL/PEFT), 커리큘럼 없이 1단계. Qwen3.5는 262K 네이티브 컨텍스트 + 선형어텐션이라 초기부터 풀 컨텍스트 가능(폴백 Qwen3-4B 시에만 8~12k truncate)
- 백테스트: SFT 모델 vs 베이스라인

**산출물**: 학습된 LoRA adapter + 비교 리포트.
**완료 기준**: 루프 완주 + (SFT가 베이스라인 상회 또는 원인 분석 문서화).

### Phase 2 — 3단계 커리큘럼 + GRPO RL (재현 등급)
**목적**: 논문 방법론 충실 구현, 홀드아웃에서 논문 추세와 비교.

- `reward` — 구조(XML)·근거(인용 grounding)·결정(비대칭 행렬) 정확 구현 (요약문서 §5). **보상 단계화, 스캐폴딩 유연**(R0 교훈)
- `train/rft` — TRL GRPOTrainer + vLLM 롤아웃 서버(Node2, HTTP), KL→ref SFT. **Ray 회피**
- 3단계 커리큘럼: STRUCTURE→CLAIMS→DECISION (각 SFT warm-start→RFT→augment)
- 데이터 확장: 14종목·18개월·풀 컨텍스트(연산 허용 시)
- 검증: 논문 홀드아웃(AAPL/GOOGL/AMZN/SPY, 2024-06~08)

**산출물**: 재현 등급 모델 + 논문 대비 비교표.
**완료 기준**: GRPO 학습 안정 수렴 + 홀드아웃에서 위계(SLM<LLM<SFT≤RFT<R1) 방향 재현.

### Phase 3 — 한국 시장 + 배포 경화
**목적**: 데이터층 교체로 한국 적용, 배포 경로 확립.

- `data/fetchers` 한국 어댑터(KRX 가격·DART 공시·한국 뉴스). qf-lib KR 데이터 공백 시 backtrader/vectorbt/커스텀 검토
- 한국어 thesis 재증류 → 재라벨 → 재학습
- `infer` 서비스(vLLM 단일노드) + signal API + 일배치 백테스트 + 모니터링

**산출물**: KR 시장 모델 + 배포 가능한 추론 경로.
**완료 기준**: KR 종목 백테스트 리포트 + 추론 API 동작.

---

## 5. 핵심 설계 결정

1. **로컬 증류 + reject sampling 생략** — 변동성 라벨이 정답이므로 교사에게 "이 라벨을 정당화하는 thesis" 직접 생성. 비용 0, 통제↑.
2. **라벨 = forward returns(`shift(-τ)`)** — 지도학습 정답으로서 미래수익 사용. 백테스트는 입력만 사용 → 룩어헤드 아님. Phase 0에서 분포 검증.
3. **Ray-free RL** — TRL GRPOTrainer + 원격 vLLM(HTTP). GB10 호환성 때문.
4. **보상 단계화 + 유연 스캐폴딩** — R0 실패(혼합 보상 불안정, 과잉 통제) 회피.
5. **컨텍스트** — Qwen3.5-4B 채택 시 초기부터 풀 컨텍스트(262K 네이티브) 가능; Qwen3-4B 폴백 시에만 8~12k→점진 확장.
6. **데이터층 추상화 선반영** — Phase 0부터 `fetchers` 인터페이스를 시장 독립으로 설계해 Phase 3 한국 이식 비용↓.
7. **2노드 역할 분리** — Node2=교사/롤아웃 서버, Node1=학습. 또는 병렬 실험(CONFIG_SLICE).
8. **백본 = `Qwen3.5-4B-Base` (게이트)** — 논문 Qwen3-4B 대비 업그레이드. 이유: ① 262K 네이티브 컨텍스트(우리 20~43k 입력의 최대 난제 해소) + Gated DeltaNet 선형어텐션 O(n) 효율, ② `<think>` thinking 모드가 논문 출력 포맷과 정합, ③ Apache 2.0(배포 목표 유리), ④ dense 4B(메모리 동급). **단, 신규 아키텍처라 GB10/SM121 커널·nightly vLLM 호환성을 Phase 0 스파이크에서 검증**, 미통과 시 Qwen3-4B 폴백. **텍스트 전용 사용**(비전 인코더 미사용).

---

## 6. 기술 스택 매핑

| 영역 | 선택 | 비고 |
|---|---|---|
| 환경/패키징 | `uv` | 사용자 지정 |
| **백본** | `Qwen/Qwen3.5-4B-Base` (폴백 `Qwen3-4B`) | Apache 2.0, dense, 262K ctx, thinking. **GB10 게이트** |
| 데이터 fetcher | TradingAgents 재활용 | Phase 0 검증 |
| 라벨링/백테스트 | pandas + qf-lib | 모델 독립 |
| 교사 서빙 | vLLM 단일노드(`--enforce-eager`) | AWQ-INT4, GB10 |
| SFT | TRL + PEFT(LoRA) | Qwen3.5는 transformers/vLLM **main·nightly** 필요→버전 고정·검증 필수. unsloth GB10 지원 불확실→보류 |
| RFT | TRL GRPOTrainer + vLLM(HTTP) | Ray/verl/OpenRLHF 회피 |
| 트래킹 | W&B + 중앙 config | ml-convention 준수 |
| 잡 오케스트레이션 | sparkq | 모든 GPU 잡 |

---

## 7. 검증 정의 (데이터 미공개 전제)

절대수치 재현 불가. "검증"은 다음으로 정의:
1. **상대 개선** — 각 단계 모델이 직전 베이스라인을 risk-adjusted(주로 Sharpe)로 상회.
2. **방향 일치** — 논문 위계(SLM < LLM < SFT ≤ RFT < R1)와 같은 추세 재현.
3. **라벨 검증** — 라벨 분포가 Table 2와 일치.
4. **포맷/보상 검증** — 보상 함수가 §8 포맷 출력에 대해 의도대로 점수 부여(단위 테스트).

---

## 8. 리스크 & 완화

| 리스크 | 완화 |
|---|---|
| **Qwen3.5 신규 아키텍처(Gated DeltaNet)가 GB10/SM121 미지원** | **Phase 0 호환성 스파이크로 조기 판별, Qwen3-4B 폴백** |
| 긴 컨텍스트 연산 비용 | Qwen3.5 채택 시 선형어텐션으로 완화; 폴백 시 초기 컨텍스트 축소·점진 확장 |
| GRPO GB10 불안정 | Ray-free 경로, 단일노드 롤아웃, KL 페널티 |
| 교사 품질 상한(32B<o3) | 학습·재현엔 수용, 배포 시 재검토(더 큰 교사/2노드 llama.cpp) |
| qf-lib 한국 데이터 공백 | Phase 3 대체 백테스터 검토 |
| TradingAgents fetcher 부적합 | Phase 0 조기 검증, 미스매치 시 자체 fetcher |
| 데이터 미공개로 수치 불일치 | §7 상대·방향 검증으로 목표 재정의 |

---

## 9. 미해결/추후 결정

- 교사 모델 최종 선정(Qwen3-32B vs Qwen3.5-27B/35B-A3B vs 2노드 llama.cpp 70B급) — Phase 1 진입 시 품질·속도 벤치 후 결정. **주의**: 교사도 Qwen3.5면 동일한 GB10 신규아키텍처 서빙 리스크 → 서빙 안정성 우선 시 Qwen3-32B(구 아키텍처) 또는 GGUF/llama.cpp 경로가 더 안전.
- 한국 백테스터 — Phase 3 진입 시 qf-lib KR 어댑터 가능성 확인 후 결정.
- GRPO G(그룹 크기)·KL β·LoRA rank 등 하이퍼파라미터 — Phase 2 실험으로 튜닝.

---

## 10. 다음 단계

이 설계 승인 후 **Phase 0의 구현 계획(writing-plans)**부터 작성. 전체 프로젝트는 한 plan으로 묶기엔 크므로 **Phase별로 spec→plan→구현 사이클**을 반복한다.
