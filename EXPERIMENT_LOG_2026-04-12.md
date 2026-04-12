# EXPERIMENT LOG — 2026-04-12 (KST)

## Scope
VisualPRM thread에서 수행한 Gemini / RAG / PRM-selection 관련 실험 및 코드 변경을 누락 없이 기록.

## Code Changes (commits)
1. `f60da14`
- 파일: `api_backend.py`, `ablation_online_compare.py`
- 변경:
  - `/generate-steps`에서 provider 429(APIError) 시 500 대신 fallback 200 응답
  - retrieval gating 추가 (`retrieval_min_score`, `retrieval_gated`, `retrieval_effective_hits`)
  - ablation summary에 `baseline_pred_dist`, `rag_pred_dist` 추가
  - 결과 row에 `decision_source`, `retrieval_gated`, `retrieval_top_hit_score` 기록

2. `a6b2ffb`
- 파일: `api_backend.py`
- 변경:
  - `/agent-answer`에 `bon_n`, `selection_mode(model|prm)` 추가
  - 후보 N개 생성(temperature sweep)
  - surrogate PRM score 계산 후 `selection_mode=prm` 선택
  - 응답에 `candidate_count`, `candidate_scores`, `selection_mode`, `bon_n` 추가

3. `bc579be`
- 파일: `api_backend.py`
- 변경:
  - PRM scoring에 diversity penalty 추가(후보 한 옵션 과집중 시 감점)
  - yes/no calibration 추가(강한 retrieval 근거 + 근소 점수차 시 retrieval 클래스 후보 선택)

## Runs & Results

### A) online rag vs baseline (n_each=10, total 20) — pre-fix
- 파일: `ablation_online_rag_vs_baseline.json`
- 결과(이전 상태): baseline 0/20 완료(전부 실패), rag 20/20 완료, rag_acc 0.5
- 원인: `/generate-steps` 500 (Gemini spend-cap 429 전파)

### B) online rag vs baseline smoke (n_each=1, total 2) — after fallback fix
- 파일: `ablation_online_rag_vs_baseline.json`
- 결과: baseline_acc 1.0, rag_acc 1.0, errors 0/0
- 의미: baseline hard-fail 복구 확인

### C) online rag vs baseline (n_each=10, total 20) — after fix
- 파일: `ablation_online_rag_vs_baseline.json`
- 결과:
  - baseline_acc 0.5
  - rag_acc 0.5
  - baseline_avg_latency 14.981s
  - rag_avg_latency 15.221s
  - baseline_errors/rag_errors = 0/0
  - baseline_pred_dist = 0:20, 1:0
  - rag_pred_dist = 0:20, 1:0
- 해석: 안정성 개선, 정확도 개선 없음, 0-class collapse 지속

### D) online rag vs baseline (n_each=5, total 10)
- 파일: `ablation_online_rag_vs_baseline.json`
- 결과:
  - baseline_acc 0.5
  - rag_acc 0.5
  - baseline_avg_latency 15.076s
  - rag_avg_latency 15.395s
  - pred_dist 모두 0:10, 1:0

### E) pathvqa-only 5 샘플 커스텀 러닝
- 파일: `ablation_online_pathvqa_5.json`
- 결과: `connection refused`로 실패(`acc=null`)
- 상태: 유효 성능지표로 사용하지 않음

### F) PRM mode compare (total 10, baseline vs rag_model vs rag_prm)
- 스크립트: `prm_mode_compare.py`
- 파일: `ablation_online_prm_mode_compare_10.json`

#### F-1) BoN+PRM 도입 직후
- baseline: acc 0.5, pred 0:10
- rag_model: acc 0.5, pred 0:10
- rag_prm: acc 0.5, pred 0:10
- prm_decision_source_dist: prm_select 10/10
- 해석: PRM selection path는 작동, 성능/편향 개선 없음

#### F-2) diversity penalty + yes/no calibration 반영 후
- 파일 mtime 기준 최신 결과 동일 유지
- baseline: acc 0.5, pred 0:10
- rag_model: acc 0.5, pred 0:10
- rag_prm: acc 0.5, pred 0:10
- 해석: heuristic surrogate PRM으로는 collapse 해소 실패

## Infra / Runtime Notes
- backend 로그에서 Gemini 429 spend-cap 반복 확인
- `/generate-steps`는 fallback 처리 후 200 유지
- WSL 직접 localhost 호출은 간헐 `connection refused`; PowerShell 런 경로가 상대적으로 안정

## Current Conclusion (as of this log)
1. 파이프라인 안정성(에러율)은 개선됨
2. retrieval 자체 점수는 높은 편(이전 분석 기준 top_hit avg ~0.889)
3. 병목은 retrieval보다 final selection
4. surrogate PRM + calibration heuristic만으로는 0-class collapse 해소 불충분
5. 다음 단계는 팀원 PRM(실제 학습 모델) 연결이 필요

## Next Actions (locked)
1. 팀원 PRM API/함수 스펙 확정
   - 입력: question/options/retrieval_context/candidate_steps
   - 출력: candidate score
2. `/agent-answer`의 surrogate score 경로를 real PRM score로 교체
3. 동일 10/20 샘플 A/B 재실행
   - rag_model vs rag_prm_real
4. 비교 지표
   - acc / latency / error / pred_dist / decision_source
