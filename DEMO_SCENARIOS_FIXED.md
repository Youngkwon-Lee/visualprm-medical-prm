# VisualPRM 고정 데모 시나리오 (v1)

목적: 발표/회의/검증에서 항상 같은 흐름으로 재현 가능한 데모 세트.

## 공통 실행 순서
1. `start_clean_stack.ps1` 실행
2. `check_stack_e2e.ps1`로 사전 점검 (HTTP 200 확인)
3. 아래 5개 시나리오를 순서대로 시연

---

## Scenario 1 — Baseline Pathology Yes/No
- Dataset: PathVQA
- Case type: Pathology
- Question type: Yes/No
- 질문 예시: `Is the nucleus enlarged?`
- 데모 포인트:
  - `router.specialist = pathology_reasoner`
  - `retrieval_mode = qdrant`
  - `document_hits >= 1`

## Scenario 2 — Pathology Multi-choice (해석형)
- Dataset: PathVQA
- Case type: Pathology
- Question type: Multi-choice
- 질문 예시: `Which pattern best matches the lesion?`
- 데모 포인트:
  - rerank 적용 후 상위 retrieval hit 표시
  - step별 reasoning trace + 최종 답변 일관성

## Scenario 3 — Radiology Routing 확인
- Dataset: VQA-RAD
- Case type: Radiology
- Question type: Yes/No 또는 Multi-choice
- 질문 예시: `Is there pleural effusion?`
- 데모 포인트:
  - `router.specialist`가 radiology 계열로 분기되는지
  - case-memory + document support 동시 동작

## Scenario 4 — Document Support 강조 케이스
- Dataset: PathVQA 또는 VQA-RAD
- 문서 근거가 잘 붙는 질문 선택
- 데모 포인트:
  - `document_hits`와 doc title/text 일부를 근거로 제시
  - "모델 단독 추론"이 아니라 "근거 기반"임을 설명

## Scenario 5 — 안정성/운영 시나리오 (재기동 후 재현)
- 동작 중 서버 재시작 후 동일 질문 재실행
- 데모 포인트:
  - `start_clean_stack` → 재질문 → 정상 응답
  - 운영 안정화(재현 가능성) 강조

---

## 발표용 한 줄 메시지
- "현재 스택은 dataset 로드부터 routing, retrieval/rerank, document-supported answering까지 end-to-end로 재현 가능하며, clean start + e2e check로 운영 안정성을 검증할 수 있습니다."

## 실패 시 빠른 체크
1. `http://localhost:8764/health` 확인
2. `http://localhost:8765/app.html` 200 확인
3. `backend_clean_runtime.err.log`에서 traceback 확인
4. 필요 시 `start_clean_stack.ps1` 재실행
