# RunPod A100 빠른 시작 가이드 (비용 절약 버전)

**핵심: GPU 비용 절약 = 모델 다운로드는 CPU로, 학습은 GPU로**

---

## 📋 2단계 워크플로우

```
CPU Instance (비용 0)          A100-80GB GPU (비용 발생)
─────────────────             ────────────────────
1. setup_runpod.sh    ──→      2. train_runpod.sh
   (20-30분)                     (11시간)
   - 의존성 설치                 - 학습만 실행
   - 모델 다운로드              - 비용: $5.30
   - 환경 설정
```

---

## 🚀 Step-by-Step

### **Step 1: 저가 CPU 인스턴스 생성 (1-2분)**

RunPod 웹사이트:
1. GPU Pods → CPU (가장 저렴함)
2. "Start Pod" 클릭
3. SSH 접속

**비용: ~$0.002/시간**

### **Step 2: 세팅 실행 (20-30분, 비용 0)**

```bash
cd /workspace

# 저장소 복사
git clone https://github.com/YOUR_ORG/visualprm.git
cd visualprm

# 세팅 실행 (모델 다운로드 포함)
bash setup_runpod.sh

# 검증 (선택사항)
bash verify_setup.sh
```

**실행 내용:**
- ✅ Python 의존성 설치
- ✅ Qwen3-VL-30B 모델 다운로드 (~10-20분)
- ✅ 환경 설정
- ✅ 워크스페이스 준비

### **Step 3: GPU 인스턴스로 변경 (1분)**

RunPod 웹사이트:
1. 현재 Pod "Pause" (중지하되 데이터 유지)
2. A100-80GB 새 Pod 생성
3. SSH 접속

**비용: $0.48/시간부터 시작**

### **Step 4: 학습 실행 (11시간, 비용 $5.30)**

```bash
cd /workspace/visualprm

# 학습 시작 (Qwen 서버 + 학습 자동)
bash train_runpod.sh standard

# 또는 다른 데이터셋
bash train_runpod.sh mvp      # 2.5시간, $1.20
bash train_runpod.sh large    # 27시간, $13
```

**자동 실행:**
- Qwen 서버 시작 (포트 8000)
- 학습 시작
- 로그 저장
- 완료 후 정리

### **Step 5: 결과 다운로드**

```bash
# 로컬 머신에서
scp -P YOUR_PORT -r root@YOUR_IP:/workspace/models ./

# 또는 RunPod 웹사이트에서 저장
```

---

## 💰 비용 비교

```
표준 방식 (세팅 중 GPU 켜놓음):
┌──────────────────────────┐
│ CPU: ~2시간 × $0.002 = $0.004
│ GPU: 12시간 × $0.48 = $5.76  ← 낭비!
│ 모델 다운: API $3,250
├──────────────────────────┤
│ 총합: $9,000+ (매우 비쌈)
└──────────────────────────┘

최적화 방식 (이 가이드):
┌──────────────────────────┐
│ CPU: 2시간 × $0.002 = $0.004
│ GPU: 11시간 × $0.48 = $5.30
│ 모델 다운: 로컬 (API 0)
├──────────────────────────┤
│ 총합: $5.30 (1,700배 저렴!)
└──────────────────────────┘
```

---

## 📊 예상 시간표

```
CPU 세팅 단계 (20-30분)
├─ 의존성 설치: 5분
├─ 모델 다운로드: 15-20분 (인터넷 속도에 따라)
└─ 환경 설정: 2분

GPU 학습 단계 (11시간, Standard)
├─ 서버 시작: 1분
├─ Epoch 1: 4시간
├─ Epoch 2: 4시간
├─ Epoch 3: 3시간
└─ 정리: 1분

────────────────────────
총소요: ~12시간
비용: ~$5.30
```

---

## ⚠️ 문제 해결

### 모델 다운로드가 느림
```bash
# 더 빠른 캐시 위치로 변경
export HF_HOME=/workspace/.cache
bash setup_runpod.sh
```

### 저가 CPU 찾기
```
RunPod 웹사이트:
GPU Pods → 정렬: "Price (Low → High)"
"CPU Only" 선택 (~$0.001-0.01/시간)
```

### GPU 스위칭
```bash
# 이전 Pod 유지 (데이터 보존)
1. Pause (Stop이 아닌 Pause!)
2. New Pod 생성
3. 동일 workspace 사용
```

---

## 🎯 학습 후

### 모니터링
```bash
# 실시간 로그
tail -f /workspace/logs/training.log

# GPU 사용량
nvidia-smi dmon 1
```

### 결과 확인
```bash
# 모델 구조
ls -lh /workspace/models/final/

# 체크포인트
ls -lh /workspace/models/checkpoint-*/
```

### 평가
```bash
# 학습된 모델로 평가 실행
python test_mc_pipeline.py --model_path /workspace/models/final
```

---

## 📌 핵심 정리

| 단계 | 인스턴스 | 시간 | 비용 | 작업 |
|------|---------|------|------|------|
| 1 | CPU | 20-30분 | $0.004 | 세팅 + 모델 다운 |
| 2 | A100-80GB | 11시간 | $5.30 | 학습 실행 |
| **총합** | - | **~12시간** | **$5.30** | ✅ 완료 |

---

## 🚨 꼭 기억할 것

1. **먼저 CPU로 세팅하기**
   ```bash
   bash setup_runpod.sh  # GPU 없이 실행
   ```

2. **GPU 변경 시 "Pause" 사용**
   - Stop이 아니라 **Pause** (데이터 유지)

3. **Qwen 서버는 train_runpod.sh에 포함**
   - 별도로 실행할 필요 없음

4. **완료 후 Pod 삭제**
   - RunPod 웹사이트에서 "Delete" 선택
   - 데이터는 다운받아서 보관

---

**준비되면 가시겠어요?** 🚀

```bash
# 바로 실행
bash setup_runpod.sh    # CPU: 비용 최소
# ... 완료 후 GPU로 변경 ...
bash train_runpod.sh standard  # GPU: 비용만 발생
```
