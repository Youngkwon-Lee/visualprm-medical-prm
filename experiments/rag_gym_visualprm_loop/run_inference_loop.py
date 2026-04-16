import os
import re
import json
import argparse
import random
from pathlib import Path
from urllib import request, error

import yaml


def load_json(path):
    with open(path, 'r', encoding='utf-8-sig') as f:
        return json.load(f)


def http_post_json(url, payload, timeout=45):
    data = json.dumps(payload).encode('utf-8')
    req = request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method='POST',
    )
    with request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode('utf-8')
        return json.loads(body)


def normalize_gold(sample):
    if isinstance(sample.get('gold'), int):
        return int(sample['gold'])
    if isinstance(sample.get('answer_index'), int):
        return int(sample['answer_index'])
    return 0


def build_image_url(image_url, prefix=""):
    image_url = (image_url or '').strip()
    prefix = (prefix or '').strip()
    if not image_url:
        return ''
    if image_url.startswith('http://') or image_url.startswith('https://'):
        return image_url
    if prefix:
        return prefix.rstrip('/') + '/' + image_url.lstrip('/')
    # 상대경로 이미지는 원격 백엔드(ollama)에서 접근 불가할 수 있어 기본적으로 비활성화
    return ''


def actor_generate_candidates(sample, cfg, prefix_steps):
    base = cfg['backend']['base_url'].rstrip('/')
    timeout = float(cfg['backend'].get('timeout_sec', 45))
    n = int(cfg['loop']['num_actions'])

    dataset = sample.get('dataset') or 'pathvqa'
    case_type = sample.get('case_type') or 'Medical'
    modality = sample.get('modality') or ''
    question = sample.get('question', '')
    options = sample.get('options', [])
    gold = normalize_gold(sample)
    image_url = build_image_url(sample.get('image_url', ''), cfg['data'].get('image_url_prefix', ''))

    candidates = []
    for _ in range(n):
        payload = {
            'question': question,
            'options': options,
            'gold': gold,
            'dataset': dataset,
            'case_type': case_type,
            'modality': modality,
            'image_url': image_url,
            'temperature': float(cfg['loop'].get('temperature', 0.8)),
            'top_p': float(cfg['loop'].get('top_p', 1.0)),
            'prefix_steps': prefix_steps,
        }
        try:
            out = http_post_json(f"{base}/generate-steps", payload, timeout=timeout)
        except Exception as exc:
            out = {'error': f'generate_error:{exc}'}

        candidates.append({
            'steps': out.get('steps', []),
            'final_answer_index': out.get('final_answer_index'),
            'final_answer_letter': out.get('final_answer_letter'),
            'raw': out,
        })
    return candidates


def extract_query_from_steps(steps, fallback_question=""):
    """Heuristic query extractor from generated steps."""
    query_patterns = [
        r"(?:generated_query|query)\s*[:=]\s*(.+)",
        r"검색\s*질의\s*[:=]\s*(.+)",
        r"추가\s*검색\s*[:=]\s*(.+)",
    ]
    for step in steps or []:
        text = str(step.get('text', ''))
        for pat in query_patterns:
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                q = m.group(1).strip().strip('"\'')
                if q and q.lower() not in {'none', 'n/a', '없음'}:
                    return q
    return fallback_question.strip()


def retrieve_docs_via_backend(sample, cfg, query):
    """Use backend /agent-answer retrieval side-effects to get top-k hits for the query."""
    base = cfg['backend']['base_url'].rstrip('/')
    timeout = float(cfg['backend'].get('timeout_sec', 90))
    top_k = int(cfg.get('rag', {}).get('top_k', 3))

    payload = {
        'question': query,
        'options': sample.get('options', []),
        'gold': normalize_gold(sample),
        'dataset': sample.get('dataset') or 'pathvqa',
        'case_type': sample.get('case_type') or 'Medical',
        'modality': sample.get('modality') or '',
        'image_url': build_image_url(sample.get('image_url', ''), cfg['data'].get('image_url_prefix', '')),
        'top_k': top_k,
        'bon_n': 1,
        'selection_mode': 'model',
    }
    try:
        out = http_post_json(f"{base}/agent-answer", payload, timeout=timeout)
        hits = out.get('retrieval_hits', [])[:top_k]
        docs = []
        for h in hits:
            docs.append({
                'id': h.get('id'),
                'score': h.get('score'),
                'question': h.get('question', ''),
                'case_type': h.get('case_type', ''),
                'modality': h.get('modality', ''),
            })
        return docs, None
    except Exception as exc:
        return [], str(exc)


def critic_score_candidate(sample, candidate, cfg):
    mode = cfg.get('critic', {}).get('mode', 'verify_steps')
    if mode == 'random':
        return random.random(), {'mode': 'random'}

    steps = candidate.get('steps', []) or []
    if not steps:
        return -1.0, {'mode': 'verify_steps', 'error': 'skip_no_steps'}

    base = (cfg.get('critic', {}).get('base_url') or cfg['backend']['base_url']).rstrip('/')
    timeout = float(cfg['backend'].get('timeout_sec', 90))
    payload = {
        'question': sample.get('question', ''),
        'options': sample.get('options', []),
        'gold': normalize_gold(sample),
        'case_type': sample.get('case_type') or 'Medical',
        'modality': sample.get('modality') or '',
        'steps': steps,
    }
    try:
        out = http_post_json(f"{base}/verify-steps", payload, timeout=timeout)
        results = out.get('results', [])
        avg = (sum(float(x.get('score', 0.0)) for x in results) / len(results)) if results else 0.0
        return float(avg), {'mode': 'verify_steps', 'results': results}
    except Exception as exc:
        return -1.0, {'mode': 'verify_steps', 'error': str(exc)}


def run_one(sample, cfg):
    trace = []
    prefix_steps = []
    prev_answer = None
    same_answer_rounds = 0

    max_iter = int(cfg['loop'].get('max_iter', 3))
    stop_if_answer = bool(cfg['loop'].get('stop_if_answer', True))
    early_same = int(cfg['loop'].get('early_stop_same_answer_rounds', 2))

    final_answer_index = None
    final_answer_letter = None
    rag_enabled = bool(cfg.get('rag', {}).get('enabled', False))

    for t in range(max_iter):
        candidates = actor_generate_candidates(sample, cfg, prefix_steps)
        scored = []
        for cand in candidates:
            score, info = critic_score_candidate(sample, cand, cfg)
            scored.append({'candidate': cand, 'score': score, 'critic': info})

        scored.sort(key=lambda x: x['score'], reverse=True)
        best = scored[0]
        best_cand = best['candidate']
        final_answer_index = best_cand.get('final_answer_index')
        final_answer_letter = best_cand.get('final_answer_letter')

        generated_query = extract_query_from_steps(best_cand.get('steps', []), sample.get('question', '')) if rag_enabled else ''
        retrieved_docs = []
        retrieval_error = None

        # RAG branch: query -> retrieval -> history(prefix) update
        if rag_enabled and generated_query:
            retrieved_docs, retrieval_error = retrieve_docs_via_backend(sample, cfg, generated_query)
            if retrieved_docs:
                doc_lines = [
                    f"- id={d.get('id')} score={d.get('score')} case_type={d.get('case_type')} modality={d.get('modality')} q={d.get('question')}"
                    for d in retrieved_docs
                ]
                retrieval_step = {
                    'title': f"Retrieved Evidence (step {t+1})",
                    'text': "query: " + generated_query + "\n" + "\n".join(doc_lines),
                }
                prefix_steps = (best_cand.get('steps', []) or []) + [retrieval_step]
            else:
                prefix_steps = best_cand.get('steps', [])
        else:
            prefix_steps = best_cand.get('steps', [])

        trace.append({
            'step': t,
            'best_score': best['score'],
            'best_answer_index': final_answer_index,
            'best_answer_letter': final_answer_letter,
            'generated_query': generated_query,
            'retrieved_docs_count': len(retrieved_docs),
            'retrieval_error': retrieval_error,
            'best_steps': best_cand.get('steps', []),
            'all_scores': [item['score'] for item in scored],
            'all_candidates': scored if cfg['output'].get('save_candidates', True) else None,
        })

        if stop_if_answer and final_answer_index is not None and not (rag_enabled and generated_query and t < max_iter - 1):
            if prev_answer == final_answer_index:
                same_answer_rounds += 1
            else:
                same_answer_rounds = 1
            prev_answer = final_answer_index
            if same_answer_rounds >= early_same:
                break

    # Fallback: 모델이 끝까지 답 인덱스를 못 내면 0번 옵션으로 안전 대체
    fallback_used = False
    if final_answer_index is None:
        final_answer_index = 0
        final_answer_letter = 'A'
        fallback_used = True

    return {
        'final_answer_index': final_answer_index,
        'final_answer_letter': final_answer_letter,
        'gold': normalize_gold(sample),
        'correct': (final_answer_index == normalize_gold(sample)),
        'fallback_used': fallback_used,
        'trace': trace,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = yaml.safe_load(f)

    random.seed(int(cfg.get('experiment', {}).get('seed', 42)))

    out_dir = Path(cfg['output']['dir'])
    out_dir.mkdir(parents=True, exist_ok=True)

    data = load_json(cfg['data']['input_path'])
    if isinstance(data, dict) and 'data' in data:
        data = data['data']
    data = data[: int(cfg['data'].get('max_samples', len(data)))]

    results = []
    for i, sample in enumerate(data):
        r = run_one(sample, cfg)
        results.append({
            'idx': i,
            'id': sample.get('id', f'sample_{i}'),
            'question': sample.get('question', ''),
            **r,
        })
        print(f"[{i+1}/{len(data)}] id={sample.get('id','')} pred={r['final_answer_index']} gold={r['gold']} correct={r['correct']}")

    acc = sum(1 for r in results if r.get('correct')) / len(results) if results else 0.0
    summary = {
        'experiment': cfg.get('experiment', {}).get('name', 'unnamed'),
        'samples': len(results),
        'accuracy': round(acc, 4),
    }

    out_path = out_dir / 'inference_results.json'
    sum_path = out_dir / 'summary.json'
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    with open(sum_path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"Saved: {out_path}")
    print(f"Saved: {sum_path}")
    print(f"Accuracy: {summary['accuracy']}")


if __name__ == '__main__':
    main()
