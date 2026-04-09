import json
import time
from pathlib import Path
from urllib import request, error

BASE = "http://127.0.0.1:8764"
ROOT = Path("/mnt/d/visualprm")


def load_rows(path: Path, dataset: str, n: int):
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = []
    for r in rows:
        q = str(r.get("question", "")).strip()
        opts = r.get("options") or []
        if not q or len(opts) < 2 or not isinstance(r.get("gold"), int):
            continue
        out.append(
            {
                "id": r.get("id", ""),
                "dataset": dataset,
                "question": q,
                "options": [str(x) for x in opts],
                "gold": int(r["gold"]),
                "case_type": str(r.get("case_type", "Medical")),
                "modality": str(r.get("modality", "")),
                "image_url": str(r.get("image_url", "")),
            }
        )
        if len(out) >= n:
            break
    return out


def post_json(path: str, payload: dict, timeout: int = 90):
    data = json.dumps(payload).encode("utf-8")
    req = request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"})
    with request.urlopen(req, timeout=timeout) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))


def run_case(item: dict):
    payload = {
        "id": item["id"],
        "question": item["question"],
        "options": item["options"],
        "gold": item["gold"],
        "dataset": item["dataset"],
        "case_type": item["case_type"],
        "modality": item["modality"],
        "image_url": item["image_url"],
    }

    out = {"id": item["id"], "dataset": item["dataset"]}

    # A: baseline (/generate-steps)
    t0 = time.time()
    try:
        sc, res = post_json("/generate-steps", payload)
        out["baseline_latency"] = round(time.time() - t0, 3)
        if sc == 200 and isinstance(res.get("final_answer_index"), int):
            out["baseline_ok"] = int(res["final_answer_index"] == item["gold"])
        else:
            out["baseline_ok"] = None
            out["baseline_error"] = res.get("error", f"status={sc}")
    except error.HTTPError as e:
        out["baseline_ok"] = None
        out["baseline_error"] = f"HTTP {e.code}"
        out["baseline_latency"] = round(time.time() - t0, 3)
    except Exception as e:
        out["baseline_ok"] = None
        out["baseline_error"] = str(e)
        out["baseline_latency"] = round(time.time() - t0, 3)

    # B: rag+agent (/agent-answer)
    t0 = time.time()
    try:
        sc, res = post_json("/agent-answer", payload)
        out["rag_latency"] = round(time.time() - t0, 3)
        if sc == 200 and isinstance(res.get("final_answer_index"), int):
            out["rag_ok"] = int(res["final_answer_index"] == item["gold"])
            out["retrieval_mode"] = res.get("retrieval_mode", "")
            out["specialist"] = (res.get("router") or {}).get("specialist", "")
        else:
            out["rag_ok"] = None
            out["rag_error"] = res.get("error", f"status={sc}")
    except error.HTTPError as e:
        out["rag_ok"] = None
        out["rag_error"] = f"HTTP {e.code}"
        out["rag_latency"] = round(time.time() - t0, 3)
    except Exception as e:
        out["rag_ok"] = None
        out["rag_error"] = str(e)
        out["rag_latency"] = round(time.time() - t0, 3)

    return out


def mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return round(sum(xs) / len(xs), 3) if xs else None


def main():
    testset = load_rows(ROOT / "vqarad_for_app.json", "vqarad", 10) + load_rows(ROOT / "pathvqa_for_app.json", "pathvqa", 10)
    results = [run_case(item) for item in testset]

    b_ok = [r.get("baseline_ok") for r in results if r.get("baseline_ok") is not None]
    r_ok = [r.get("rag_ok") for r in results if r.get("rag_ok") is not None]

    summary = {
        "n_total": len(testset),
        "n_baseline_done": len(b_ok),
        "n_rag_done": len(r_ok),
        "baseline_acc": round(sum(b_ok) / len(b_ok), 3) if b_ok else None,
        "rag_acc": round(sum(r_ok) / len(r_ok), 3) if r_ok else None,
        "baseline_avg_latency": mean([r.get("baseline_latency") for r in results]),
        "rag_avg_latency": mean([r.get("rag_latency") for r in results]),
        "baseline_errors": sum(1 for r in results if r.get("baseline_ok") is None),
        "rag_errors": sum(1 for r in results if r.get("rag_ok") is None),
    }

    out = {"summary": summary, "results": results}
    out_path = ROOT / "ablation_online_rag_vs_baseline.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
