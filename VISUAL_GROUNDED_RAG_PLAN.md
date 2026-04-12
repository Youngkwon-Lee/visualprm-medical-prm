# Visual-grounded RAG Plan (Phase 1)

## Phase 1 (text-grounded visual bridge)
1. 이미지 -> VLM 요약 생성 (`image_summary`)
2. 이미지 -> OCR 텍스트 추출 (`ocr_text`)
3. query = question + image_summary + ocr_text
4. 기존 텍스트 벡터DB 검색 수행

## Phase 2 (multimodal index)
1. 이미지 임베딩 추가 저장
2. text+dense hybrid + image similarity fusion
3. rerank with PRM-aware scoring

## Immediate TODO
- `image_summary`/`ocr_text` 필드 API payload 추가
- retrieval query composer 함수 분리
- 실험: text-only vs visual-bridged text-RAG
