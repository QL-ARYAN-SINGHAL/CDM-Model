"""Benchmark KnowLine-4B accuracy and latency under different serving options.

  benchmark_knowline.py single CONFIG [DATASET]    single prompts (default eval_set_500.json), one intent each
  benchmark_knowline.py big CONFIG STRATEGY        eval_big.json: all five intents on one big prompt
  benchmark_knowline.py summary [PATTERN]          print accuracy / latency for saved runs

CONFIG names (see CONFIGS): which GGUF, whether the prompt is read twice, compact state, prompt cache.
STRATEGY: "line" = the tester's current way (every line x intent); "whole" = the whole prompt once per intent.
Each run starts its own llama-server on port 8033, sends a warm-up request, then times every request end to end
(prompt rendering + tokenizing + llama-server), one at a time.
"""
import json
import os
import subprocess
import sys
import time
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PORT = 8033
os.environ["KNOWLINE_URL"] = f"http://127.0.0.1:{PORT}"
sys.path.insert(0, str(ROOT))
import knowline_bridge as kb  # noqa: E402

Q5 = ROOT / "models" / "knowline" / "KnowLine-4B-Gen5.5.Q5_K_M.gguf"
Q4 = ROOT / "models" / "candidates" / "KnowLine-4B-Gen5.5.Q4_K_M.gguf"
K2B = ROOT / "models" / "candidates" / "knowline-2b" / "KnowLine-2B-Gen1.Q5_K_M.gguf"
K08B = ROOT / "models" / "candidates" / "knowline-08b" / "KnowLine-0.8B-Gen1.Q5_K_M.gguf"
CONFIGS = {  # name: (gguf, read_twice, compact_state, cache)
    "k2b-once": (K2B, False, True, False),
    "k08b-once": (K08B, False, True, False),
    "q5-twice": (Q5, True, False, False),          # what the tester ran so far
    "q5-once": (Q5, False, False, False),
    "q5-twice-compact": (Q5, True, True, False),
    "q5-once-compact": (Q5, False, True, False),
    "q4-once-compact": (Q4, False, True, False),
    "q4-twice-compact": (Q4, True, True, False),
    "q5-once-compact-cache": (Q5, False, True, True),
    "q5-twice-compact-cache": (Q5, True, True, True),
    "q4-once-compact-cache": (Q4, False, True, True),
    "q4-twice-compact-cache": (Q4, True, True, True),
}
OUT = ROOT / "eval_results"
HIGH, LOW = 0.7, 0.4


class llama:
    def __init__(self, gguf):
        self.gguf = gguf

    def __enter__(self):
        self.proc = subprocess.Popen(
            ["/opt/homebrew/bin/llama-server", "-m", str(self.gguf), "--host", "127.0.0.1", "--port", str(PORT),
             "-c", "8192", "-np", "1", "-b", "2048", "-ub", "2048"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2)
                break
            except Exception:
                time.sleep(1)
        kb.KNOWLINE_MODEL_PATH = Path(self.gguf)
        return self

    def __exit__(self, *exc):
        self.proc.terminate()
        self.proc.wait()


def question(defn):
    return types.SimpleNamespace(**{k: defn[k] for k in ("name", "description", "do", "dont")})


def run_single(config, dataset):
    gguf, twice, compact, cache = CONFIGS[config]
    cases = json.loads(Path(dataset).read_text())
    rows = []
    with llama(gguf):
        scorer = kb._Scorer(read_twice=twice, cache=cache)
        kb.compute_scan(scorer, cases[0]["text"], [question(cases[0]["q"])], LOW, HIGH)  # warm-up
        for i, case in enumerate(cases):
            started = time.perf_counter()
            _, best = kb.compute_scan(scorer, case["text"], [question(case["q"])], LOW, HIGH)
            ms = (time.perf_counter() - started) * 1000
            rows.append({"intent": case["q"]["name"], "text": case["text"], "expected": case["expected"],
                         "score": best[case["q"]["name"]], "ms": round(ms, 1)})
            if i % 50 == 0:
                print(f"{config} {i}/{len(cases)}", flush=True)
    OUT.mkdir(exist_ok=True)
    name = f"bench-{Path(dataset).stem}-{config}"
    (OUT / f"{name}.json").write_text(json.dumps({"config": config, "rows": rows}, indent=1))
    summarize(name)


def run_big(config, strategy, limit=None, shape=None):
    gguf, twice, compact, cache = CONFIGS[config]
    cases = json.loads((ROOT / "eval_big.json").read_text())
    defs = {c["q"]["name"]: c["q"] for c in json.loads((ROOT / "eval_set.json").read_text())}
    intents = ["travel", "Banking", "Refund Policy", "SONAR Defense", "P & L"]
    qs = [question(defs[n]) for n in intents]
    shape = shape or ("multiline" if strategy == "line" else "paragraph")
    rows = []
    with llama(gguf):
        scorer = kb._Scorer(read_twice=twice, cache=cache)
        kb.compute_scan(scorer, "Book a flight to Delhi tomorrow.", qs[:1], LOW, HIGH)  # warm-up
        for case in cases[:limit]:
            text = case[shape]
            started = time.perf_counter()
            _, best = kb.compute_scan(scorer, text, qs, LOW, HIGH)
            ms = (time.perf_counter() - started) * 1000
            rows.append({"id": case["id"], "present": case["intents_present"], "scores": best, "ms": round(ms, 1),
                         "lines": len(case["lines"]), "words": len(text.split())})
            print(f"{config}/{strategy}/{shape} #{case['id']} {ms/1000:.1f}s present={case['intents_present']}", flush=True)
    OUT.mkdir(exist_ok=True)
    name = f"bigbench-{strategy}-{shape}-{config}"
    (OUT / f"{name}.json").write_text(json.dumps({"config": config, "strategy": strategy, "shape": shape, "rows": rows}, indent=1))
    summarize_big(name)


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / max(len(pos) * len(neg), 1)


def summarize(name):
    rows = json.loads((OUT / f"{name}.json").read_text())["rows"]
    s, y = [r["score"] for r in rows], [r["expected"] for r in rows]
    n = len(rows)
    ok7 = sum((a >= HIGH) == b for a, b in zip(s, y))
    ok5 = sum((a >= 0.5) == b for a, b in zip(s, y))
    fp = sum(a >= HIGH and not b for a, b in zip(s, y))
    fn = sum(a < HIGH and b for a, b in zip(s, y))
    rev = sum(LOW <= a < HIGH for a in s)
    ms = [r["ms"] for r in rows]
    print(f"{name:52} n={n} acc@0.7 {ok7}/{n} ({100*ok7/n:.1f}%) acc@0.5 {100*ok5/n:.1f}% fp {fp} fn {fn} review {rev} "
          f"AUC {auc(s, y):.4f} | ms mean {sum(ms)/n:.0f} p50 {pct(ms, .5):.0f} p95 {pct(ms, .95):.0f}")


def summarize_big(name):
    rows = json.loads((OUT / f"{name}.json").read_text())["rows"]
    intents = ["travel", "Banking", "Refund Policy", "SONAR Defense", "P & L"]
    tp = fp = fn = tn = 0
    exact = 0
    for r in rows:
        ok = True
        for i in intents:
            pred, want = r["scores"][i] >= HIGH, i in r["present"]
            tp += pred and want
            fp += pred and not want
            fn += (not pred) and want
            tn += (not pred) and not want
            ok &= pred == want
        exact += ok
    n = len(rows)
    total = tp + fp + fn + tn
    ms = [r["ms"] for r in rows]
    print(f"{name:52} n={n} intent-level acc {100*(tp+tn)/total:.1f}% (tp {tp} fp {fp} fn {fn} tn {tn}) "
          f"all-5-correct {exact}/{n} | s per prompt mean {sum(ms)/n/1000:.1f} p50 {pct(ms, .5)/1000:.1f} p95 {pct(ms, .95)/1000:.1f}")


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "single":
        run_single(sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else str(ROOT / "eval_set_500.json"))
    elif mode == "big":
        run_big(sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 and sys.argv[4] != "-" else None,
                sys.argv[5] if len(sys.argv) > 5 else None)
    else:
        pat = sys.argv[2] if len(sys.argv) > 2 else ""
        for f in sorted(OUT.glob(f"*{pat}*.json")):
            (summarize_big if f.name.startswith("bigbench") else summarize)(f.stem) if f.name.startswith(("bench-", "bigbench")) else None
