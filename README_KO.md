# BI-SO101 양팔 로봇 — 전체 파이프라인 가이드

> 🇺🇸 [English version](README.md)

**BI-SO101** 양팔 로봇의 텔레오퍼레이션, 데이터셋 수집, 학습, 추론까지 전체 과정을 다룬 가이드입니다. [LeRobot](https://github.com/huggingface/lerobot) 기반으로 동작합니다.

설정 파일 및 캘리브레이션 데이터는 [`bi_so101_configs/`](https://github.com/XHAKA3456/lerobot/tree/dual-arm-xlerobot/bi_so101_configs)에 있습니다.

---

## 목차

1. [설치](#설치)
2. [하드웨어 설정](#하드웨어-설정)
3. [텔레오퍼레이션](#텔레오퍼레이션)
4. [데이터셋 수집](#데이터셋-수집)
5. [학습](#학습)
6. [추론](#추론)
7. [좋은 데이터 수집 팁](#좋은-데이터-수집-팁)
8. [문제 해결](#문제-해결)

---

## 설치

```bash
# dual-arm-xlerobot 브랜치 직접 클론
git clone -b dual-arm-xlerobot https://github.com/XHAKA3456/lerobot.git
cd lerobot

# conda 환경 생성
conda create -n lerobot python=3.10 -y
conda activate lerobot

# feetech 모터 지원 포함 설치
pip install -e ".[feetech]"
```

> **Hugging Face 로그인** (데이터셋 업로드 및 모델 다운로드에 필요):
> ```bash
> huggingface-cli login
> ```

---

## 하드웨어 설정

### 로봇 구성

| 역할 | 장치 | 포트 |
|------|------|------|
| **Follower** (실제 동작하는 로봇) | BI-SO101 왼팔 | `/dev/ttyACM0` |
| **Follower** | BI-SO101 오른팔 | `/dev/ttyACM1` |
| **Leader** (사람이 조작하는 팔) | BI-SO101 왼팔 | `/dev/ttyACM2` |
| **Leader** | BI-SO101 오른팔 | `/dev/ttyACM3` |

### 카메라 구성

| 카메라 | 장치 | 연결 위치 |
|--------|------|----------|
| 왼쪽 | `/dev/video2` | 첫 번째 USB 허브 |
| 오른쪽 | `/dev/video4` | 두 번째 USB 허브 |
| 정면 | `/dev/video6` | 첫 번째 USB 허브 (마지막에 연결) |

### 연결 순서 — 예시

> **⚠️ 카메라 인덱스와 시리얼 포트는 PC마다 다릅니다.** `ls /dev/video*` 와 `ls /dev/ttyACM*` 으로 본인 시스템에 실제 할당된 장치를 먼저 확인하고, 설정 파일을 그에 맞게 수정하세요.

> **이것은 예시 구성입니다.** 리더 팔 2개, 팔로워 팔 2개, 카메라 3개로 구성한 경우를 보여줍니다. 본인 환경에 맞게 포트를 조정하세요.

OS는 연결 순서대로 `/dev/ttyACM*` 및 `/dev/video*`를 순차 할당합니다. 매번 동일한 순서로 연결해야 포트가 일정하게 유지됩니다.

```
예시 구성:

첫 번째 USB 허브:
  1. Follower 왼팔   → /dev/ttyACM0
  2. Follower 오른팔 → /dev/ttyACM1
  3. 왼쪽 카메라     → /dev/video2

두 번째 USB 허브:
  4. Leader 왼팔     → /dev/ttyACM2
  5. Leader 오른팔   → /dev/ttyACM3
  6. 오른쪽 카메라   → /dev/video4

첫 번째 USB 허브 (마지막):
  7. 정면 카메라     → /dev/video6
```

설정 파일의 `port` 및 `index_or_path` 값을 실제 할당된 장치에 맞게 수정하세요.

### USB 권한 설정

모든 장치를 연결한 후, 먼저 실제로 인식된 장치를 확인합니다:

```bash
ls /dev/ttyACM*
ls /dev/video*
```

확인된 장치에만 권한을 부여합니다:

```bash
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1 /dev/ttyACM2 /dev/ttyACM3
sudo chmod 666 /dev/video2 /dev/video4 /dev/video6
```

케이블을 다시 뺐다 꽂을 필요 없이, 인식된 장치 노드에 맞게 chmod만 실행하면 됩니다.

### 캘리브레이션

캘리브레이션 파일이 이미 저장소에 포함되어 있습니다:

```
bi_so101_configs/calibration/
├── bi_so101_follower/
│   ├── black_left.json
│   └── black_right.json
└── bi_so101_leader/
    ├── black_left.json
    └── black_right.json
```

모터를 교체하지 않는 한 재캘리브레이션은 필요하지 않습니다.

---

## 텔레오퍼레이션

데이터 수집 전 하드웨어 연결 확인 및 카메라 각도 조절에 사용합니다.

```bash
cd lerobot
./bi_so101_configs/run_teleoperate.sh
```

Follower 팔이 Leader 팔을 실시간으로 따라갑니다. 카메라 피드가 화면에 표시되므로, 세 카메라 앵글을 보면서 위치를 조절하세요.

**설정 파일:** `bi_so101_configs/scripts/bi_so101_teleoperate.yaml`

---

## 데이터셋 수집

### 1. 설정 파일 수정

**`bi_so101_configs/scripts/bi_so101_record.yaml`**

최소한 `repo_id`와 `root`를 변경해야 합니다:

```yaml
dataset:
  repo_id: your_hf_username/your_dataset_name   # ← 변경
  root: /path/to/your/datasets/your_dataset_name # ← 변경
  single_task: "태스크 설명을 입력하세요"
  episode_time_s: 20    # 에피소드당 녹화 시간 (초)
  reset_time_s: 5       # 에피소드 간 리셋 대기 시간 (초)
  num_episodes: 30      # 총 에피소드 수

resume: false           # 기존 데이터셋에 이어서 수집하려면 true
```

| 파라미터 | 설명 |
|----------|------|
| `repo_id` | Hugging Face 데이터셋 ID (`사용자명/이름`) |
| `root` | 로컬 저장 경로 |
| `episode_time_s` | 에피소드당 녹화 시간 (초) |
| `reset_time_s` | 에피소드 종료 후 다음 시작까지 대기 시간 (초) |
| `num_episodes` | 수집할 총 에피소드 수 |
| `resume` | `true`: 기존 데이터셋에 추가 / `false`: 새로 시작 |

### 2. 실행

```bash
cd lerobot
./bi_so101_configs/run_record.sh
```

설정한 에피소드 수만큼 수집이 완료되면 자동으로 Hugging Face Hub에 업로드됩니다 (약 10분 소요).

### Ctrl+C로 중단했을 때

걱정하지 마세요 — 업로드가 안 됐더라도 **데이터는 로컬에 저장**되어 있습니다. 스크립트를 다시 실행해서 나머지 에피소드를 채우면, 이전에 저장된 데이터가 함께 업로드됩니다.

### 수동 업로드

```bash
python3 bi_so101_configs/push_dataset.py \
    --repo_id "your_hf_username/your_dataset_name" \
    --root "/path/to/your/datasets/your_dataset_name"
```

---

## 학습

> **⚠️ Rubik Pi 등 엣지 디바이스에서는 학습이 불가능합니다.** 데이터셋을 Hugging Face Hub에 업로드한 뒤, CUDA GPU가 있는 서버에서 학습을 진행해야 합니다.

학습은 LeRobot 공식 파이프라인을 사용합니다:

👉 **[LeRobot 학습 가이드](https://github.com/huggingface/lerobot?tab=readme-ov-file#train-your-own-policy)**

예시 (ACT 정책, GPU 서버에서):

```bash
lerobot-train \
  --policy.type=act \
  --dataset.repo_id=your_hf_username/your_dataset_name \
  --output_dir=outputs/train/your_run_name
```

학습된 모델은 `output_dir`에 저장되며, `--hub_id` 옵션으로 Hugging Face Hub에 업로드할 수 있습니다.

---

## 추론

### 1. 학습된 모델 다운로드

```bash
huggingface-cli download your_hf_username/your_model_name \
    --local-dir bi_so101_configs/models/your_model_name
```

### 2. 설정 파일 수정

**`bi_so101_configs/scripts/bi_so101_infer.yaml`**

```yaml
policy:
  type: act
  pretrained_path: /path/to/bi_so101_configs/models/your_model_name  # ← 변경
  device: cuda   # 또는 "cpu"

dataset_repo_id: your_hf_username/your_dataset_name  # ← 변경
```

### 3. 실행

```bash
cd lerobot
./bi_so101_configs/run_infer.sh
```

Follower 로봇이 학습된 정책으로 자율 동작합니다.

> **Rubik Pi의 경우:** 먼저 `source ~/miniconda3/bin/activate lerobot` 으로 환경을 활성화하세요.

### 추론 에피소드 녹화 (선택)

로봇의 자율 동작을 데이터셋으로 기록하려면:

```bash
./bi_so101_configs/run_record_policy.sh
```

또는 특정 정책을 지정해서 실행:

```bash
./bi_so101_configs/run_record_policy.sh \
    --policy.path=your_hf_username/your_model_name \
    --policy.n_action_steps=50
```

---

## 좋은 데이터 수집 팁

### 깔끔한 배경 설정

카메라 중 하나 이상이 **아무것도 없는 벽**을 향하도록 배치하세요. 에피소드 사이에 위치가 바뀔 수 있는 물체는 모두 프레임 밖으로 치우세요. 배경의 변화는 모델이 처리하기 어려운 노이즈가 됩니다.

### 동작을 단계별로 나눠서 진행

태스크를 한 번에 빠르게 끝내려 하지 말고, 동작을 명확한 단계로 분리하세요:

1. 그리퍼를 목표 물체 **위쪽**으로 이동
2. 잡기 좋은 각도로 회전
3. 그리퍼 완전히 벌리기
4. 내려서 물체 잡기
5. 이동 후 내려놓기

단계별로 명확하게 움직일수록 액션 궤적이 깔끔해지고 정책 학습에 유리합니다.

### 조명 환경 일정하게 유지

비전 기반 정책은 조명에 민감합니다. 에피소드 사이에 그림자 방향이 바뀌거나, 아침/오후 햇빛이 달라지는 것만으로도 성능이 크게 떨어질 수 있습니다. **일정한 조명 환경**에서 수집하고, 가능하면 직사광선이 드는 창가는 피하세요.

### 매 에피소드마다 물체 위치 바꾸기

항상 같은 자리에 물체를 두면 그 위치에서만 동작하는 정책이 됩니다. 매 에피소드마다 물체의 위치와 방향을 바꿔서 일반화 성능을 높이세요.

---

## 문제 해결

### 포트 권한 오류
```bash
ls /dev/ttyACM*   # 실제 인식된 장치 확인
ls /dev/video*
sudo chmod 666 /dev/ttyACM0 /dev/ttyACM1 ...   # 확인된 장치만 입력
```

### 장치가 잘못된 포트에 할당됨
케이블을 모두 분리하고 [하드웨어 설정](#하드웨어-설정)의 순서대로 다시 연결하세요.

### 데이터셋 디렉토리가 이미 존재함
로컬 데이터셋 디렉토리를 삭제하거나, 설정 파일에서 `resume: true`로 변경하세요.

### 추론 시 모델을 찾을 수 없음
`bi_so101_infer.yaml`의 `pretrained_path`가 모델이 다운로드된 경로와 일치하는지 확인하세요.

### 업로드가 너무 느리거나 실패함
대용량 데이터셋은 업로드에 10~30분이 걸릴 수 있습니다. 실패하면 `push_dataset.py`로 수동 재시도하세요.
