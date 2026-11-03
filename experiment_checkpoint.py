"""Does llama-server reuse the shared message across the 3 intents when context checkpoints are allowed at small spacing?
Control (cache_prompt=false) and test (cache_prompt=true) on the same server and texts; also checks the scores are identical.
Usage: experiment_checkpoint.py URL LABEL
"""
import json
import statistics
import sys

import experiments_knowline as E
import profile_knowline as P

url, label = sys.argv[1], sys.argv[2]
probe = P.Probe(url)
items = E.texts(25, seed=61)
probe.request(items[0][0], items[0][1])  # warm-up
rows = []
for text, intents in items:
    ctrl = probe.request(text, intents, cache=False)
    test = probe.request(text, intents, cache=True)
    rows.append({"control_ms": ctrl["e2e_ms"], "cache_ms": test["e2e_ms"], "prompt_n": [p["prompt_n"] for p in test["passes"]],
                 "cache_n": [p["cache_n"] for p in test["passes"]], "score_diff": max(abs(a - b) for a, b in zip(ctrl["scores"], test["scores"]))})
s = {"label": label, "control_ms_mean": round(statistics.mean(r["control_ms"] for r in rows), 1), "cache_ms_mean": round(statistics.mean(r["cache_ms"] for r in rows), 1),
     "cache_ms_p95": round(P.pct([r["cache_ms"] for r in rows], .95), 1),
     "tokens_processed_per_pass": [round(statistics.mean(r["prompt_n"][i] for r in rows), 1) for i in range(3)],
     "tokens_reused_per_pass": [round(statistics.mean(r["cache_n"][i] for r in rows), 1) for i in range(3)],
     "max_score_diff_vs_control": round(max(r["score_diff"] for r in rows), 6)}
print(json.dumps(s))
(P.OUT / f"checkpoint_{label}.json").write_text(json.dumps({"rows": rows, "summary": s}, indent=1))
