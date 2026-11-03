"""Backend and transport overheads around the model call, measured without touching production data.

 - FastAPI route (/knowline/scan) in-process through TestClient, with TRACKING_CSV pointed at a temporary file: compared with
   compute_scan on the same texts, the difference is route overhead (validation, _loaded_model GET, CSV append, JSON response).
 - Pieces timed alone: pydantic validation, GET /v1/models, CSV append.
 - Transport: GET /knowline/health on the production API (local) and through the public Cloudflare link (30 samples each).
Usage: measure_overheads.py ISOLATED_LLAMA_URL [PUBLIC_HOST]
"""
import csv
import json
import statistics
import subprocess
import sys
import tempfile
import time
import types
import urllib.request
from pathlib import Path

import profile_knowline as P
import knowline_bridge as kb

url = sys.argv[1].rstrip("/")
public = sys.argv[2] if len(sys.argv) > 2 else None
kb.KNOWLINE_URL = url
tmp = Path(tempfile.mkdtemp()) / "knowline_tracking_test.csv"
kb.TRACKING_CSV = tmp  # never the production CSV
out = {}
cases = P.texts if False else None
import experiments_knowline as E  # noqa: E402

items = E.texts(20, seed=51)
qs_of = lambda intents: [{"name": d["name"], "description": d["description"], "do": d["do"], "dont": d["dont"]} for d in intents]

# 1. route vs compute_scan
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402
app = FastAPI()
app.include_router(kb.router)
client = TestClient(app)
scorer = kb._Scorer(read_twice=False, cache=False)
client.post("/knowline/scan", json={"raw_prompt": items[0][0], "questions": qs_of(items[0][1])})  # warm-up
route, core = [], []
for text, intents in items:
    t = time.perf_counter()
    r = client.post("/knowline/scan", json={"raw_prompt": text, "questions": qs_of(intents)})
    route.append((time.perf_counter() - t) * 1000)
    assert r.status_code == 200, r.text[:200]
    t = time.perf_counter()
    qs = [types.SimpleNamespace(**q) for q in qs_of(intents)]
    kb.compute_scan(scorer, text, qs, 0.4, 0.7, read_twice=False)
    core.append((time.perf_counter() - t) * 1000)
out["route_ms_mean"] = round(statistics.mean(route), 1)
out["compute_scan_ms_mean"] = round(statistics.mean(core), 1)
out["route_overhead_ms_mean"] = round(statistics.mean(r - c for r, c in zip(route, core)), 1)

# 2. pieces alone
t = time.perf_counter()
for _ in range(1000):
    kb.ScanRequest(raw_prompt=items[0][0], questions=qs_of(items[0][1]))
out["pydantic_validation_ms"] = round((time.perf_counter() - t) * 1000 / 1000, 3)
t = time.perf_counter()
for _ in range(50):
    urllib.request.urlopen(url + "/v1/models", timeout=5).read()
out["get_v1_models_ms"] = round((time.perf_counter() - t) * 1000 / 50, 2)
req = types.SimpleNamespace(raw_prompt=items[0][0], questions=[types.SimpleNamespace(model_dump=lambda q=q: q) for q in qs_of(items[0][1])], low=0.4, high=0.7)
res = {"model": "m", "selected_intent": None, "status": "no_match", "intent_scores": {"a": 0.1}, "lines": [{"line": items[0][0]}], "questions_asked": 3, "latency_ms": 1.0}
t = time.perf_counter()
for _ in range(50):
    kb.append_tracking(req, res)
out["csv_append_ms"] = round((time.perf_counter() - t) * 1000 / 50, 2)

# 3. transport
def curl_ms(target, n=30):
    vals = []
    for _ in range(n):
        o = subprocess.run(["curl", "-s", "-o", "/dev/null", "-w", "%{time_total}", "--noproxy", "*", "-m", "20"] + target, capture_output=True, text=True).stdout
        vals.append(float(o) * 1000)
    return {"mean": round(statistics.mean(vals), 1), "p95": round(P.pct(vals, .95), 1)}
out["local_api_health_ms"] = curl_ms(["http://127.0.0.1:8000/knowline/health"])
if public:
    ip = subprocess.run(f"dig +short @1.1.1.1 {public} | head -1", shell=True, capture_output=True, text=True).stdout.strip()
    out["public_link_health_ms"] = curl_ms(["--resolve", f"{public}:443:{ip}", f"https://{public}/knowline/health"])
print(json.dumps(out, indent=1))
(P.OUT / "overheads.json").write_text(json.dumps(out, indent=1))
