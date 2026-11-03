"""Compare KnowLine-4B and Jeffan-4B on the 120 labelled prompts in eval_set.json (results in eval_results/big-*.json)."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
HIGH, LOW = 0.7, 0.4  # the tester's match / review cutoffs
cases = json.loads((ROOT / "eval_set.json").read_text())
models = {"KnowLine-4B": "big-knowline", "Jeffan-4B": "big-jeffan"}


def auc(scores, labels):
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def pct(values, q):
    values = sorted(values)
    return values[min(len(values) - 1, int(round(q * (len(values) - 1))))]


results = {}
for name, label in models.items():
    rows = json.loads((ROOT / "eval_results" / f"{label}.json").read_text())["rows"]
    assert len(rows) == len(cases) and all(r["prompt"] == c["text"] for r, c in zip(rows, cases))
    results[name] = rows

labels = [c["expected"] for c in cases]
print(f"{'':12} {'acc@0.7':>9} {'acc@0.5':>9} {'prec':>6} {'recall':>7} {'F1':>6} {'false+':>7} {'false-':>7} {'review':>7} {'AUC':>6} {'best-thr acc':>13}")
for name, rows in results.items():
    s = [r["score"] for r in rows]
    tp = sum(x >= HIGH and y for x, y in zip(s, labels))
    fp = sum(x >= HIGH and not y for x, y in zip(s, labels))
    fn = sum(x < HIGH and y for x, y in zip(s, labels))
    ok7 = sum((x >= HIGH) == y for x, y in zip(s, labels))
    ok5 = sum((x >= 0.5) == y for x, y in zip(s, labels))
    prec, rec = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    review = sum(LOW <= x < HIGH for x in s)
    best = max(sum((x >= t / 100) == y for x, y in zip(s, labels)) for t in range(1, 100))
    n = len(cases)
    print(f"{name:12} {ok7:>3}/{n} {100*ok7/n:3.0f}% {ok5:>3}/{n} {prec:6.3f} {rec:7.3f} {f1:6.3f} {fp:>7} {fn:>7} {review:>7} {auc(s, labels):6.3f} {best:>9}/{n}")

print("\nPer intent, correct at 0.7 (of 24):")
intents = list(dict.fromkeys(c["q"]["name"] for c in cases))
print(f"{'':12} " + " ".join(f"{i[:13]:>13}" for i in intents))
for name, rows in results.items():
    cells = []
    for i in intents:
        idx = [k for k, c in enumerate(cases) if c["q"]["name"] == i]
        cells.append(sum((rows[k]["score"] >= HIGH) == cases[k]["expected"] for k in idx))
    print(f"{name:12} " + " ".join(f"{c:>13}" for c in cells))

print("\nLatency per prompt, end to end (ms):")
for name, rows in results.items():
    ms = [r["ms"] for r in rows]
    print(f"{name:12} mean {sum(ms)/len(ms):6.0f}  p50 {pct(ms, .5):6.0f}  p95 {pct(ms, .95):6.0f}  max {max(ms):6.0f}")

print("\nMistakes at the 0.7 cutoff (expected -> score):")
for name, rows in results.items():
    for r, c in zip(rows, cases):
        if (r["score"] >= HIGH) != c["expected"]:
            print(f"  {name:12} {'YES' if c['expected'] else 'no ':3} {r['score']:.2f}  [{c['q']['name']}] {c['text'][:80]!r}")
