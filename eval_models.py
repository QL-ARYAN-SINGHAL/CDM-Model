"""Run a decision-model checkpoint served by llama-server on the prompts and intents in kev_tracking.csv.

Each unique (prompt, question) row is sent exactly as kev_bridge.py would send it. Kev 0.8B's recorded
score in the CSV is the baseline; this script produces the same score for another checkpoint.

Usage: eval_models.py PORT LABEL      writes eval_results/LABEL.json
"""
import csv
import json
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "eval_results"
LABELS = ROOT / "eval_labels.json"  # {prompt prefix: bool}  true = the intent applies


def rows():
    seen = {}
    with (ROOT / "archive" / "kev_0.8b_tracking_2026-11-03.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            seen[(row["raw_prompt"], row["questions"])] = row
    return list(seen.values())


def score(port, model_id, prompt, questions):
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    qs = {}
    for i in range(len(lines)):
        for j, q in enumerate(questions):
            qs[f"l{i}__q{j}"] = {"type": "noul",
                                 "instructions": f"Judge only lines[{i}]. Use full_message only as context. {q['description']}",
                                 "criteria": {"true": q["do"], "false": q["dont"]}}
    body = {"model": model_id, "state": {"full_message": prompt, "lines": lines}, "questions": qs}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/systemone", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    started = time.perf_counter()
    with urllib.request.urlopen(req, timeout=300) as response:
        answers = json.load(response)["answers"]
    ms = (time.perf_counter() - started) * 1000
    best = max(float(answers[f"l{i}__q{j}"]["noul"]) for i in range(len(lines)) for j in range(len(questions)))
    return round(best, 4), round(ms, 1)


def main(port, label):
    model_id = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/v1/models"))["data"][0]["id"]
    out = []
    for row in rows():
        questions = json.loads(row["questions"])
        p, ms = score(port, model_id, row["raw_prompt"], questions)
        out.append({"prompt": row["raw_prompt"], "intent": questions[0]["name"], "score": p, "ms": ms})
        print(f"{label:10} {questions[0]['name'][:10]:10} {p:.3f} {ms:7.0f}ms | {row['raw_prompt'][:70]}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{label}.json").write_text(json.dumps({"model": model_id, "rows": out}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]), sys.argv[2])
