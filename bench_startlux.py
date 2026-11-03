"""Run eval_set_500.json against a StartLux /v1/systemone server (startlux-decide serve) and save results in the same
format as benchmark_knowline.py. The question is a noul with the intent's description and Do/Don't, as for KnowLine.

Usage: bench_startlux.py LABEL [PORT]
"""
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
label, port = sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 8095
import os
DATASET = Path(os.environ.get("EVAL_SET", ROOT / "eval_set_500.json"))
cases = json.loads(DATASET.read_text())


def ask(case):
    q = case["q"]
    body = {"state": case["text"], "questions": {"q": {"type": "noul", "instructions": q["description"],
                                                         "criteria": {"true": q["do"], "false": q["dont"]}}}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t = time.perf_counter()
    answers = json.load(urllib.request.urlopen(req, timeout=300))["answers"]
    return float(answers["q"]["noul"]), (time.perf_counter() - t) * 1000


ask(cases[0])  # warm-up
rows = []
for i, case in enumerate(cases):
    score, ms = ask(case)
    rows.append({"intent": case["q"]["name"], "text": case["text"], "expected": case["expected"], "score": score, "ms": round(ms, 1)})
    if i % 100 == 0:
        print(label, i, flush=True)
out = ROOT / "eval_results"
out.mkdir(exist_ok=True)
(out / f"bench-{DATASET.stem}-{label}.json").write_text(json.dumps({"config": label, "rows": rows}, indent=1))
s, y = [r["score"] for r in rows], [r["expected"] for r in rows]
ms = sorted(r["ms"] for r in rows)
ok = sum((a >= 0.7) == b for a, b in zip(s, y)); ok5 = sum((a >= 0.5) == b for a, b in zip(s, y))
fp = sum(a >= 0.7 and not b for a, b in zip(s, y)); fn = sum(a < 0.7 and b for a, b in zip(s, y))
pos, neg = [a for a, b in zip(s, y) if b], [a for a, b in zip(s, y) if not b]
auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
by = {}
for r in rows:
    by.setdefault(r["intent"], [0, 0]); by[r["intent"]][0] += (r["score"] >= 0.7) == r["expected"]; by[r["intent"]][1] += 1
print(f"RESULT {label}: acc@0.7 {ok}/{len(rows)} ({100*ok/len(rows):.1f}%) acc@0.5 {100*ok5/len(rows):.1f}% fp {fp} fn {fn} AUC {auc:.4f} | ms mean {sum(ms)/len(rows):.0f} p50 {ms[len(rows)//2]:.0f} p95 {ms[int(.95*(len(rows)-1))]:.0f} | " + " ".join(f"{k[:6]} {v[0]}/{v[1]}" for k, v in by.items()))
