"""End-to-end check of the tester's /knowline/scan with the shipped strategy: sampled single prompts + big prompts."""
import json, random, sys, time, urllib.request
base = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8000"
random.seed(7)
cases = json.load(open("eval_set_500.json"))
sample = random.sample(cases, 60)
big = json.load(open("eval_big.json"))
defs = {c["q"]["name"]: c["q"] for c in cases}
intents = ["travel", "Banking", "Refund Policy", "SONAR Defense", "P & L"]

def scan(prompt, qs):
    body = json.dumps({"raw_prompt": prompt, "questions": qs}).encode()
    t = time.perf_counter()
    d = json.load(urllib.request.urlopen(urllib.request.Request(base + "/knowline/scan", body, {"Content-Type": "application/json"}), timeout=300))
    return d, (time.perf_counter() - t) * 1000

ok, ms = 0, []
for c in sample:
    d, t = scan(c["text"], [c["q"]])
    ms.append(t); ok += (d["intent_scores"][c["q"]["name"]] >= 0.7) == c["expected"]
ms.sort()
print(f"single: {ok}/{len(sample)} correct, ms mean {sum(ms)/len(ms):.0f} p50 {ms[len(ms)//2]:.0f} p95 {ms[int(.95*(len(ms)-1))]:.0f}; scope={d['scope']} read_twice={d['read_twice']}")
tp = fp = fn = tn = 0; times = []
for b in big[:10]:
    d, t = scan(b["multiline"], [defs[i] for i in intents])
    times.append(t / 1000)
    for i in intents:
        pred, want = d["intent_scores"][i] >= 0.7, i in b["intents_present"]
        tp += pred and want; fp += pred and not want; fn += (not pred) and want; tn += (not pred) and not want
n = tp + fp + fn + tn
print(f"big (10 prompts x 5 intents): intent-level {100*(tp+tn)/n:.0f}% (tp {tp} fp {fp} fn {fn} tn {tn}), s/prompt mean {sum(times)/len(times):.1f} max {max(times):.1f}; scope={d['scope']} read_twice={d['read_twice']} passes={d['questions_asked']}")
