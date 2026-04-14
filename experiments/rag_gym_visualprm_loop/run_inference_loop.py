import os
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
    return image_url


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


def critic_score_candidate(sample, candidate, cfg):
    mode = cfg.get('critic', {}).get('mode', 'verify_steps')
    if mode == 'random':
        return random.random(), {'mode': 'random'}

    base = cfg['backend']['base_url'].rstrip('/')
    timeout = float(cfg['backend'].get('timeout_sec', 45))
    payload = {
        'question': sample.get('question', ''),
        'options': sample.get('options', []),
        'gold': normalize_gold(sample),
        'case_type': sample.get('case_type') or 'Medical',
        'modality': sample.get('modality') or '',
        'steps': candidate.get('steps', []),
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

        trace.append({
            'step': t,
            'best_score': best['score'],
            'best_answer_index': final_answer_index,
            'best_answer_letter': final_answer_letter,
            'best_steps': best_cand.get('steps', []),
            'all_scores': [item['score'] for item in scored],
            'all_candidates': scored if cfg['output'].get('save_candidates', True) else None,
        })

        prefix_steps = best_cand.get('steps', [])

        if stop_if_answer and final_answer_index is not None:
            if prev_answer == final_answer_index:
                same_answer_rounds += 1
            else:
                same_answer_rounds = 1
            prev_answer = final_answer_index
            if same_answer_rounds >= early_same:
                break

    return {
        'final_answer_index': final_answer_index,
        'final_answer_letter': final_answer_letter,
        'gold': normalize_gold(sample),
        'correct': (final_answer_index == normalize_gold(sample)) if final_answer_index is not None else False,
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
