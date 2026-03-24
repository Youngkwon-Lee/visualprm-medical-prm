# Phase 1: CPU Instance Setup (비용 절약 모드)

**목표**: 모델 다운로드 + 환경 설정 (GPU 없이)
**시간**: 20-30분
**비용**: ~$0.004 (무시할 수준)
**인스턴스**: RunPod CPU 저가 옵션

---

## 🚀 Step-by-Step 실행

### Step 1: RunPod 접속

https://www.runpod.io/console/pods

### Step 2: CPU Instance 생성

1. **"GPU Pods"** 클릭
2. **"CPU"** 선택 (또는 Filter: Price Low→High, GPU는 안 선택)
3. **가장 저렴한 옵션** 클릭
   - 예시: CPU-only, 1-2 vCPU, 4-8GB RAM
   - 가격: ~$0.001-0.01/hr
4. **"Start Pod"** 클릭

### Step 3: Pod 접속

```bash
# RunPod 웹사이트에서 Pod 선택 → "Connect" → SSH 명령 복사
# 예시:
ssh -p YOUR_PORT root@YOUR_IP
```

### Step 4: 저장소 복사 및 세팅 실행

```bash
# Pod 내에서 실행
cd /workspace

# 저장소 복사
git clone https://github.com/YOUR_ORG/visualprm.git
cd visualprm

# 세팅 실행 (자동으로 모든 것 처리)
bash setup_runpod.sh
```

**자동으로 실행되는 것들:**
```
✓ 의존성 설치 (torch, transformers, peft 등)
✓ Qwen3-VL-30B 모델 다운로드 (10-20분, 65GB)
✓ 워크스페이스 구성
✓ 환경 설정
```

### Step 5: 검증 (선택사항)

```bash
# 모든 준비가 완료되었는지 확인
bash verify_setup.sh

# Expected output:
# Passed: 21
# Failed: 0
# All tests passed! Ready for GPU training.
```

---

## ⏱️ 예상 시간

```
Step 1-2: RunPod 계정 로그인 및 Pod 생성   → 2-3분
Step 3: SSH 접속                          → 1분
Step 4: 저장소 복사 + setup_runpod.sh    → 20-30분
  - 의존성 설치: 5분
  - 모델 다운로드: 15-20분 (인터넷 속도에 따라)
  - 환경 설정: 2분
Step 5: 검증 (선택)                      → 2분

────────────────────────────────────────────
총소요: 약 25-35분
비용: ~$0.004 ✅
```

---

## 💰 비용 예상

```
CPU 인스턴스 35분 × $0.002/hr = $0.001-0.004

매우 저렴! (A100-80GB 1시간 = $0.48)
```

---

## 📝 운영 팁

### 실시간 모니터링

```bash
# 로그 확인 (실시간)
tail -f /workspace/logs/server.log
```

### 문제 발생 시

```bash
# 1. Pod 상태 확인
nvidia-smi  # (GPU 없으면 "command not found" 정상)

# 2. 로그 확인
cat /workspace/logs/server.log

# 3. 수동으로 다시 실행
cd /workspace/visualprm
bash setup_runpod.sh
```

### 네트워크 느림 (모델 다운 느림)

```bash
# HuggingFace 캐시 위치 변경
export HF_HOME=/workspace/.hf_cache
bash setup_runpod.sh
```

---

## ✅ Phase 1 완료 확인

**다음과 같으면 Phase 1 완료:**

```
✅ setup_runpod.sh 실행 완료
✅ verify_setup.sh 통과 (Passed: 21, Failed: 0)
✅ 모델 다운로드 완료
  - /workspace/.cache/huggingface/hub/models--Qwen--Qwen3-VL-30B-Instruct/ 확인
  - 크기: ~65GB
```

---

## 🚀 Phase 2로 진행

**Phase 1 완료 후:**

1. **현재 Pod 중지 (데이터 유지)**
   - RunPod 웹사이트 → Pod 선택 → "Pause" (Stop이 아님!)

2. **A100-40GB 새 Pod 생성**
   - GPU: A100-40GB
   - Storage: 500GB+
   - "Start Pod"

3. **새 Pod에서 SSH 접속**

4. **학습 실행**
   ```bash
   cd /workspace/visualprm
   bash train_runpod.sh standard
   ```

---

## 🎯 기억할 것

| 항목 | 설명 |
|------|------|
| **Phase 1 목표** | 모델 다운로드 + 세팅 (GPU 불필요) |
| **인스턴스** | CPU (저가) |
| **시간** | 25-35분 |
| **비용** | ~$0.004 |
| **다음 단계** | Pause → A100-40GB 생성 → 학습 |
| **학습 비용** | ~$3.52 (11시간) |

---

## 📞 문제 발생 시

1. **모델 다운로드 실패**
   ```bash
   # 수동으로 다시 실행
   huggingface-cli download Qwen/Qwen3-VL-30B-Instruct --cache-dir /workspace/.cache/huggingface
   ```

2. **의존성 설치 실패**
   ```bash
   pip install -r requirements.txt
   ```

3. **SSH 연결 실패**
   - RunPod 웹사이트에서 Pod 상태 확인
   - "Running" 상태인지 확인
   - 몇 초 후 재시도

---

**준비됐으면 지금 바로 RunPod에서 CPU Pod 생성하세요!** 🚀

```
1. https://www.runpod.io/console/pods 열기
2. CPU 저가 옵션 선택
3. "Start Pod" 클릭
4. SSH 접속
5. 이 문서의 Step 4 실행
```

---

**완료하면 알려주세요!** ✅
