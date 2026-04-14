# RAG-Gym × VisualPRM Inference Loop Setup

## 목적
VisualPRM critic + actor 후보생성 + (선택적)RAG를 결합한 추론 루프를 빠르게 실험.

## 파일
- `config.yaml` : 루프/샘플링/검색 기본 설정
- `run_inference_loop.py` : 최소 실행 가능한 루프 템플릿
- `outputs/` : 추론 결과 저장 폴더

## 빠른 실행
```bash
cd /mnt/d/visualprm/experiments/rag_gym_visualprm_loop
python run_inference_loop.py --config config.yaml
```

## 다음 단계
1. actor/critic 실제 호출 함수 연결
2. 데이터셋 로더 연결 (PathVQA/Kvasir)
3. rag on/off ablation 실행
