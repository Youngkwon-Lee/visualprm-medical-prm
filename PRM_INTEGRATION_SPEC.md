# PRM Integration Spec (v1)

## Goal
Surrogate PRM 점수 경로를 팀원 실제 PRM 모델 점수로 교체.

## Request Schema
```json
{
  "question": "string",
  "options": ["string"],
  "retrieval_context": "string",
  "candidate": {
    "final_answer_index": 0,
    "final_answer_letter": "A",
    "steps": [{"title":"...","text":"..."}]
  },
  "meta": {
    "dataset": "pathvqa|vqarad|...",
    "case_type": "Medical",
    "modality": "string"
  }
}
```

## Response Schema
```json
{
  "score": 0.0,
  "confidence": 0.0,
  "reason": "optional"
}
```

## Runtime knobs
- `PRM_PROVIDER=surrogate|http`
- `PRM_HTTP_ENDPOINT=<url>`
- `PRM_HTTP_TIMEOUT_SEC=15`
- `PRM_FAIL_OPEN=1` (실패 시 surrogate fallback)

## Selection
- candidate별 score 계산 후 max score 선택
- 응답 로깅: `prm_provider`, `prm_score`, `prm_confidence`, `prm_error`
