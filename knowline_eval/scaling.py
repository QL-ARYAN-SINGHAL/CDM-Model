"""Latency vs #intents (1,2,3) x prompt length (short <=60w, long >=150w) + server RSS, for one running model. Schema A.
usage: scaling.py --kind knowline4b|knowline2b|aurelis --pid PID"""
import argparse, json, subprocess, time, statistics as st, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from run_bench import q_noul
import run_sysone
ap = argparse.ArgumentParser(); ap.add_argument("--kind", required=True); ap.add_argument("--pid", required=True); a = ap.parse_args()
suite = json.load(open(Path(__file__).parent / "suite100.json"))["prompts"]
short = [p for p in suite if p["lang"] == "en" and p["words"] <= 60][:8]
long_ = [p for p in suite if p["lang"] == "en" and p["words"] >= 150][:8]
if a.kind.startswith("knowline"):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent)); import knowline_bridge as kb
    key = "4b" if a.kind.endswith("4b") else "2b"; sc = kb._Scorer(base=kb.MODELS[key]["url"], cache=True)
    def run(p, k):
        t = time.perf_counter()
        for it in p["intents"][:k]: sc.p_true(p["text"], q_noul("A", it))
        return (time.perf_counter() - t) * 1000
else:
    url = "http://127.0.0.1:8011"
    def run(p, k):
        qs = {i["name"]: q_noul("A", i) for i in p["intents"][:k]}
        t = time.perf_counter(); run_sysone.post(url, {"model": "m", "state": p["text"], "questions": qs}); return (time.perf_counter() - t) * 1000
run(short[0], 1)  # warm-up
out = {}
for label, group in (("short", short), ("long", long_)):
    for k in (1, 2, 3):
        v = [run(p, k) for p in group]
        out[f"{label}_{k}intent"] = {"mean_ms": round(st.mean(v)), "p50_ms": round(st.median(v)), "avg_words": round(st.mean(p["words"] for p in group))}
rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", a.pid]).split()[0]) // 1024
out["server_rss_mb"] = rss
print(json.dumps(out))
