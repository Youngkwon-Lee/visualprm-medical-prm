import json
import time
from pathlib import Path
import requests

BASE = "http://127.0.0.1:8764"
sample = json.loads(Path(r'D:/visualprm/pathvqa_for_app.json').read_text(encoding='utf-8'))[0]
payload = {
    "question": sample["question"],
    "options": [str(x) for x in sample["options"]],
    "gold": int(sample["gold"]),
    "dataset": sample.get("dataset", "pathvqa"),
    "case_type": sample.get("case_type", "Medical"),
    "modality": sample.get("modality", ""),
    "image_url": sample.get("image_url", ""),
    "selection_mode": "model",
    "bon_n": 1,
    "top_k": 2,
}
print('payload built', {k:v for k,v in payload.items() if k!='question'})
start=time.time()
try:
    r=requests.post(f"{BASE}/agent-answer", json=payload, timeout=(5,180))
    dt=time.time()-start
    print('status',r.status_code,'time',round(dt,2))
    print(r.text[:400])
except Exception as e:
    print('ERR',type(e),str(e))
    print('elapsed',round(time.time()-start,2))
