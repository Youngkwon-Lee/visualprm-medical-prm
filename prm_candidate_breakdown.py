import json
from pathlib import Path
import requests

ROOT = Path(__file__).resolve().parent
BASE = "http://127.0.0.1:8764"

res = json.loads((ROOT / "ablation_online_rag_vs_baseline.json").read_text(encoding="utf-8"))
ids = [r["id"] for r in res.get("results", [])]

rows = []
for fn in ["vqarad_for_app.json", "pathvqa_for_app.json"]:
    rows.extend(json.loads((ROOT / fn).read_text(encoding="utf-8")))
idx = {r.get("id"): r for r in rows}

out = []
for rid in ids:
    r = idx.get(rid, {})
    payload = {
        "id": rid,
        "question": r.get("question", ""),
        "options": r.get("options", []),
        "gold": r.get("gold", 0),
        "dataset": "vqarad" if str(rid).startswith("vqarad") else "pathvqa",
        "case_type": r.get("case_type", "Medical"),
        "modality": r.get("modality", ""),
        "image_url": r.get("image_url", ""),
        "selection_mode": "prm",
        "bon_n": 3,
    }
    try:
        resp = requests.post(BASE + "/agent-answer", json=payload, timeout=120)
        body = resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {"raw": resp.text[:300]}
        out.append({
            "id": rid,
            "gold": r.get("gold"),
            "status": resp.status_code,
            "pred": body.get("final_answer_index") if isinstance(body, dict) else None,
            "decision_source": body.get("decision_source") if isinstance(body, dict) else None,
            "candidate_scores": body.get("candidate_scores") if isinstance(body, dict) else None,
        })
    except Exception as e:
        out.append({"id": rid, "gold": r.get("gold"), "status": "ERR", "error": str(e)})

p = ROOT / "PRM_CANDIDATE_BREAKDOWN_2026-04-14.json"
p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"saved {p}")
print("ok", sum(1 for x in out if x.get("status") == 200), "err", sum(1 for x in out if x.get("status") != 200))
