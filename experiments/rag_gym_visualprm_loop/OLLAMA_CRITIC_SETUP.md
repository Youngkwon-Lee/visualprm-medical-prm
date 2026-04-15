# Ollama(Gemma)를 Critic으로 붙이는 방법

## 목표
- Actor: 기존 Gemini 백엔드 유지 (`/generate-steps`)
- Critic: Ollama Gemma 백엔드 사용 (`/verify-steps`)

핵심 아이디어: `api_backend.py`를 **두 개 포트로 분리 실행**
- 5050 = actor(gemini)
- 5051 = critic(open_model + ollama)

---

## 1) Ollama 준비 (맥북)
예시:
```bash
ollama pull gemma3:4b
ollama serve
```

OpenAI 호환 엔드포인트(기본):
- `http://<MAC_IP>:11434/v1`

---

## 2) Critic 전용 env 파일 만들기
`/mnt/d/visualprm/.env.critic_ollama`:

```env
MODEL_PROVIDER=open_model
OPEN_MODEL_BASE_URL=http://<MAC_IP>:11434/v1
OPEN_MODEL_API_KEY=EMPTY
OPEN_MODEL_GENERATE_MODEL=gemma3:4b
OPEN_MODEL_VERIFY_MODEL=gemma3:4b
```

> 참고: 이 인스턴스는 critic용이므로 사실상 `OPEN_MODEL_VERIFY_MODEL`이 핵심.

---

## 3) 백엔드 2개 실행
### (A) Actor 서버 (기존 gemini)
```bash
cd /mnt/d/visualprm
# 기존 .env 사용
python api_backend.py --port 5050
```

### (B) Critic 서버 (ollama)
```bash
cd /mnt/d/visualprm
set -a; source .env.critic_ollama; set +a
python api_backend.py --port 5051
```

---

## 4) 루프 설정
`config.yaml`:
```yaml
backend:
  base_url: http://127.0.0.1:5050   # actor

critic:
  mode: verify_steps
  base_url: http://127.0.0.1:5051   # critic 분리
```

---

## 5) 실행
```bash
cd /mnt/d/visualprm/experiments/rag_gym_visualprm_loop
python run_inference_loop.py --config config.yaml
```

---

## 6) 체크포인트
- `curl http://127.0.0.1:5050/health`
- `curl http://127.0.0.1:5051/health`
- `outputs/summary.json` 정확도/샘플수 확인

문제 시:
- JSON 파싱 실패 → temperature 낮추기(0.8→0.4)
- 속도 느림 → `num_actions` 5→3
- critic 불안정 → verify 전용 프롬프트 강화
