"""Client for eval_suite100.json: 100 prompts, each judged against 3 intents. Time is per prompt (all 3 intents together).

  bench_suite100.py LABEL knowline PORT     KnowLine prompt layout, llama-server on PORT, 3 passes per prompt, read once
  bench_suite100.py LABEL startlux PORT     StartLux /v1/systemone on PORT, one request with 3 questions
Writes eval_results/suite100-LABEL.json and prints a summary.
"""
import json
import os
import sys
import time
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
label, mode, port = sys.argv[1], sys.argv[2], int(sys.argv[3])
os.environ["KNOWLINE_URL"] = f"http://127.0.0.1:{port}"
sys.path.insert(0, str(ROOT))
cases = json.loads((ROOT / os.environ.get("SUITE", "eval_suite100.json")).read_text())


def judge_knowline(case):
    import knowline_bridge as kb
    qs = [types.SimpleNamespace(**{k: d[k] for k in ("name", "description", "do", "dont")}) for d in case["intents"]]
    _, best = kb.compute_scan(judge_knowline.scorer, case["text"], qs, 0.4, 0.7, read_twice=False)
    return [best[q.name] for q in qs]


def judge_startlux(case):
    body = {"state": case["text"], "questions": {f"q{i}": {"type": "noul", "instructions": d["description"],
                                                               "criteria": {"true": d["do"], "false": d["dont"]}}
                                                   for i, d in enumerate(case["intents"])}}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(), {"Content-Type": "application/json"})
    answers = json.load(urllib.request.urlopen(req, timeout=300))["answers"]
    return [float(answers[f"q{i}"]["noul"]) for i in range(3)]


if mode == "knowline":
    import knowline_bridge as kb
    judge_knowline.scorer = kb._Scorer(read_twice=False, cache=False)
    judge = judge_knowline
else:
    judge = judge_startlux
judge(cases[0])  # warm-up
rows = []
for i, c in enumerate(cases):
    t = time.perf_counter()
    scores = judge(c)
    rows.append({"words": len(c["text"].split()), "expected": c["expected"], "scores": scores, "ms": round((time.perf_counter() - t) * 1000, 1)})
    if i % 25 == 0:
        print(label, i, flush=True)
(ROOT / "eval_results").mkdir(exist_ok=True)
(ROOT / "eval_results" / f"suite100-{label}.json").write_text(json.dumps({"label": label, "rows": rows}, indent=1))


def summary(label, rows):
    dec = [(s, e) for r in rows for s, e in zip(r["scores"], r["expected"])]
    ok = sum((s >= 0.7) == e for s, e in dec); ok5 = sum((s >= 0.5) == e for s, e in dec)
    fp = sum(s >= 0.7 and not e for s, e in dec); fn = sum(s < 0.7 and e for s, e in dec)
    pos = [s for s, e in dec if e]; neg = [s for s, e in dec if not e]
    auc = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))
    all3 = sum(all((s >= 0.7) == e for s, e in zip(r["scores"], r["expected"])) for r in rows)
    ms = sorted(r["ms"] for r in rows)
    under = sum(m < 1000 for m in ms)
    big = sorted(r["ms"] for r in rows if r["words"] >= 200)
    print(f"RESULT {label}: decisions {ok}/{len(dec)} ({100*ok/len(dec):.1f}%) acc@0.5 {100*ok5/len(dec):.1f}% fp {fp} fn {fn} AUC {auc:.4f} all-3-right {all3}/{len(rows)} | "
          f"ms/prompt mean {sum(ms)/len(ms):.0f} p50 {ms[len(ms)//2]:.0f} p95 {ms[int(.95*(len(ms)-1))]:.0f} max {ms[-1]:.0f} | under 1 s: {under}/{len(ms)} | prompts >=200 words: mean {sum(big)/max(len(big),1):.0f} ms (n={len(big)})")


summary(label, rows)
