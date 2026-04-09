import base64
import json
import time
from pathlib import Path

import requests

ROOT = Path("D:/visualprm")
API = "http://127.0.0.1:8764"

rows = json.loads((ROOT / "pathvqa_for_app.json").read_text(encoding="utf-8"))
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--n", type=int, default=2, help="number of samples")
parser.add_argument("--offset", type=int, default=0, help="start index in filtered cases")
args = parser.parse_args()

filtered = [
    r for r in rows
    if isinstance(r.get("image_url"), str)
    and r.get("image_url")
    and isinstance(r.get("question"), str)
    and isinstance(r.get("options"), list)
    and len(r["options"]) >= 2
    and isinstance(r.get("gold"), int)
]
start = max(0, int(args.offset))
end = start + max(1, int(args.n))
selected = filtered[start:end]

results = []
for case in selected:
    img_path = ROOT / case["image_url"]
    b64 = base64.b64encode(img_path.read_bytes()).decode("ascii")
    data_url = f"data:image/jpeg;base64,{b64}"
    payload = {
        "id": case.get("id", ""),
        "question": case.get("question", ""),
        "options": [str(x) for x in case.get("options", [])],
        "gold": int(case.get("gold", 0)),
        "dataset": case.get("dataset") or "pathvqa",
        "case_type": case.get("case_type", "Medical"),
        "modality": case.get("modality", ""),
        "image_url": data_url,
    }

    row = {"id": payload["id"], "gold": payload["gold"]}
    for ep in ("/generate-steps", "/agent-answer"):
        t0 = time.time()
        r = requests.post(API + ep, json=payload, timeout=180)
        dt = time.time() - t0
        try:
            j = r.json()
        except Exception:
            j = {"error": r.text[:300]}

        row[ep] = {
            "status": r.status_code,
            "latency": round(dt, 2),
            "pred": j.get("final_answer_index"),
            "ok": (j.get("final_answer_index") == payload["gold"]) if isinstance(j.get("final_answer_index"), int) else False,
            "error": j.get("error"),
            "retrieval_mode": j.get("retrieval_mode"),
            "document_hits": len(j.get("document_hits") or []) if isinstance(j.get("document_hits"), list) else None,
        }
    results.append(row)

summary = {
    "n": len(results),
    "generate_ok": sum(1 for r in results if r["/generate-steps"]["ok"]),
    "agent_ok": sum(1 for r in results if r["/agent-answer"]["ok"]),
    "generate_avg_latency": round(sum(r["/generate-steps"]["latency"] for r in results) / max(1, len(results)), 2),
    "agent_avg_latency": round(sum(r["/agent-answer"]["latency"] for r in results) / max(1, len(results)), 2),
}

summary["offset"] = start
summary["end"] = end
out = {"summary": summary, "results": results}
out_path = ROOT / f"vlm_eval_gemini_{len(results)}samples_offset{start}.json"
out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=2))
print(f"saved: {out_path}")
