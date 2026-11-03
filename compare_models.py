"""Compare checkpoints on the labelled prompts from kev_tracking.csv. Baseline = Kev 0.8B scores recorded in the CSV."""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LOW, HIGH = 0.4, 0.7  # the tester's default thresholds
ORDER = ["kev-0.8b (csv)", "kev-4b", "jeffan-4b", "knowline-4b"]

labels = json.loads((ROOT / "eval_labels.json").read_text())


def label_for(prompt):
    return next(v for k, v in labels.items() if prompt.startswith(k))


runs = {}
seen = {}
with (ROOT / "archive" / "kev_0.8b_tracking_2026-11-03.csv").open(newline="", encoding="utf-8") as file:
    for row in csv.DictReader(file):
        seen[(row["raw_prompt"], row["questions"])] = row
runs["kev-0.8b (csv)"] = [{"prompt": r["raw_prompt"], "score": max(json.loads(r["intent_scores"]).values()),
                           "ms": float(r["latency_ms"])} for r in seen.values()]
for name in ORDER[1:]:
    path = ROOT / "eval_results" / f"{name}.json"
    if path.exists():
        runs[name] = json.loads(path.read_text())["rows"]

print(f"{'model':16} {'acc@0.7':>8} {'acc@0.5':>8} {'false+':>7} {'false-':>7} {'review':>7} {'p50 ms':>8}")
detail = {}
for name, rows in runs.items():
    ok7 = ok5 = fp = fn = rev = 0
    for r in rows:
        want, s = label_for(r["prompt"]), r["score"]
        ok7 += (s >= HIGH) == want
        ok5 += (s >= 0.5) == want
        fp += (s >= HIGH) and not want
        fn += (s < HIGH) and want
        rev += LOW <= s < HIGH
    ms = sorted(r["ms"] for r in rows)[len(rows) // 2]
    n = len(rows)
    print(f"{name:16} {ok7}/{n} {100*ok7/n:4.0f}% {ok5}/{n} {fp:>5} {fn:>7} {rev:>7} {ms:>8.0f}")
    detail[name] = {r["prompt"]: r["score"] for r in rows}

print(f"\n{'expected':8} " + " ".join(f"{n[:11]:>11}" for n in runs) + "  prompt")
for prompt in detail[ORDER[0]]:
    want = label_for(prompt)
    cells = []
    for name in runs:
        s = detail[name][prompt]
        mark = "✓" if (s >= HIGH) == want else "✗"
        cells.append(f"{s:6.2f} {mark:>4}")
    print(f"{'YES' if want else 'no':8} " + " ".join(f"{c:>11}" for c in cells) + "  " + prompt[:60])
