# Human-in-the-Loop Offline RL for VLA

## 아이디어 개요

학습된 VLA 정책으로 로봇을 동작시키면서, 인간의 자연어 피드백과 텔레오프 개입을 통해
Offline RL 데이터셋을 반복적으로 구축하여 정책을 개선하는 방법.

---

## 동작 흐름

```
1. VLA 정책으로 로봇 동작
        ↓
2. 이상한 동작 감지 → 일시정지
        ↓
3. 인간이 자연어로 설명 ("팔이 엉뚱한 방향으로 움직였다")
        ↓
4. LLM + 미리 정의한 포맷 → reward 값으로 변환
        ↓
5. 해당 시간 구간의 프레임에 reward 부여
        ↓
6. Offline RL 데이터셋 축적
        ↓
7. 반복 → 정책 개선
```

## 텔레오프 개입

- 로봇이 너무 엉뚱하게 움직이면 인간이 텔레오프로 가로챔
- 텔레오프로 모인 데이터 → **최고 보상값 부여**
- DAgger (Dataset Aggregation) 방식과 동일한 원리
- 가장 고품질의 수정 데이터를 자연스럽게 수집

---

## 관련 연구

| 연구 | 유사한 부분 |
|------|------------|
| DAgger | 전문가 개입 + 데이터 수집 |
| RLHF | 인간 피드백 → 보상 |
| TAMER | 실시간 인간 평가로 RL |

---

## 장점

- 자동 reward function 설계 불필요 → 인간이 자연어로 제공
- 텔레오프 개입이 자연스러운 고품질 correction 데이터
- Offline RL이라 실시간 gradient 계산 불필요
- 반복할수록 데이터 품질 향상

---

## 기술적 도전 과제

### 1. Temporal Credit Assignment
- "몇 초 동안 이상했다"에서 어느 프레임이 얼마나 나쁜지 판단
- 구간 시작과 끝의 reward를 어떻게 차등 부여할지

### 2. Diffusion Policy + RL 통합
- 표준 RL (policy gradient, Q-learning)이 diffusion model에 바로 적용되지 않음
- 관련 연구: DDPO, DPPO (아직 성숙하지 않음)
- **현실적 대안**: action head를 Gaussian policy로 교체 → IQL, CQL 바로 적용 가능
  (단, diffusion 대비 성능 저하 가능)

### 3. LLM Reward 일관성
- 같은 상황에 대해 LLM이 매번 다른 reward 값을 줄 수 있음
- 포맷 표준화 및 few-shot 예시로 완화 가능

---

## 구현 방향

### Offline RL 알고리즘 후보
- **IQL** (Implicit Q-Learning): offline RL에 안정적
- **CQL** (Conservative Q-Learning): out-of-distribution action 억제
- **TD3+BC**: 간단하고 실용적
- **Decision Transformer**: sequence 모델링 기반

### VLM 처리
- VLM backbone은 **frozen 유지**
- RL은 action head (projector + diffusion/gaussian)만 업데이트
- 이유: RL 신호의 노이즈로 VLM 표현이 망가질 수 있음

### LLM Reward 포맷 예시
```json
{
  "description": "오른팔이 딸기를 지나쳐서 반대 방향으로 움직임",
  "severity": "high",
  "reward": -0.8,
  "affected_joints": ["right_arm"],
  "suggestion": "딸기 위치로 더 정확하게 접근 필요"
}
```

---

## BC → RL 전환 전략

### 모방학습이 먼저여야 하는 이유
- RL은 처음에 아무것도 모르는 상태에서 탐험하면 너무 비효율적
- 로봇이 랜덤하게 움직이다가 딸기를 우연히 집을 확률은 거의 0
- BC로 "어느 정도 되는 정책"을 먼저 만들어야 RL이 의미있는 탐험 가능 → **warm start**

```
텔레오프 데이터 수집
    → BC/모방학습 (Flow Matching loss)
    → 어느 정도 동작하는 정책 완성
    → RL로 개선 (분포 밖 상황, 세밀한 조정)
```

### 기존 BC 모델을 RL에 그대로 사용 가능
- 모델 구조 변경 불필요
- **학습 목적함수만 교체**

```
BC 학습:
  loss = flow_matching_loss(predicted_velocity, target_velocity)
  → reward 없음, demonstration 따라하기

RL 파인튜닝:
  BC로 학습된 가중치 그대로 가져옴
  + critic(가치함수) 추가
  → reward 기반으로 업데이트
```

### Flow Matching + RL 기술 허들
표준 RL은 정책의 `log_prob`이 필요한데 Flow Matching은 이를 직접 제공하지 않음.

| 방법 | 설명 | 난이도 |
|------|------|--------|
| **Offline RL (IQL)** | log_prob 없이도 동작, critic만 추가 | 낮음 (권장) |
| Actor-Critic | critic 따로 학습, BC loss + RL loss 합산 | 중간 |
| DDPO 스타일 | ODE 풀이 과정을 multi-step decision으로 처리 | 높음 |

### 검증 순서 (권장)
```
본인 Flow Matching policy에서 먼저 검증
    → RL 아이디어 + IQL 붙이기
    → 성공하면 GR00T action head에 적용
    → 궁극적으로 GR00T 전체 파이프라인 통합
```
- GR00T는 3B 파라미터라 실험 루프가 느림
- 본인 모델에서 먼저 검증하면 실패 원인 파악이 명확해짐
- Flow Matching은 ODE 기반 직선 trajectory라 diffusion보다 RL 연동이 유리할 수 있음

---

## 메모

- 논문으로 낼 수 있는 수준의 아이디어
- GR00T의 diffusion head를 어떻게 RL과 연결할지가 핵심 기술 허들
- 텔레오프 개입 + max reward 설계가 가장 실용적인 부분
- **IQL이 현재 구조에 가장 자연스럽게 붙음** - 기존 모델 건드릴 필요 없이 critic만 추가
