"""Frozen-dataset benchmark of one llama-server configuration (isolated server, never production).

usage: run_config.py URL LABEL SERVER_PID --cache {0,1} --warm {0,1} [--reps 3] [--seed 7] [--pilot N]

Per repetition, prompts run in a seeded shuffled order. For each prompt:
  COLD  : reset the slot (POST /slots/0?action=erase), then score Intents A, B, C in order (cache_prompt=--cache).
          Intent A's cache_n must be 0 (recorded as cold_valid); B and C may reuse A's checkpoints.
  WARM  : (only with --warm 1) the identical A, B, C scan again immediately, no reset.
Latency is the wall time of the complete three-intent scan (render + tokenize + HTTP + parse); the reset is outside the timer.
Scoring is the production path (profile_knowline.Probe.p_true == knowline_bridge._Scorer.p_true, read-once layout).
"""
import argparse
import hashlib
import json
import random
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import profile_knowline as P  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("url")
ap.add_argument("label")
ap.add_argument("pid", type=int)
ap.add_argument("--cache", type=int, required=True)
ap.add_argument("--warm", type=int, default=0)
ap.add_argument("--reps", type=int, default=3)
ap.add_argument("--seed", type=int, default=7)
ap.add_argument("--pilot", type=int, default=0)
args = ap.parse_args()

raw = (HERE / "frozen_dataset.json").read_bytes()
want = (HERE / "frozen_dataset.json.sha256").read_text().strip()
assert hashlib.sha256(raw).hexdigest() == want, "frozen dataset hash mismatch"
data = json.loads(raw)
intents, prompts = data["intents"], data["prompts"]
assert len(prompts) == 50 and len(intents) == 3
assert all(50 <= len(p["text"].split()) <= 250 for p in prompts)

out_dir = HERE / "results"
out_dir.mkdir(exist_ok=True)
out_path = out_dir / f"{args.label}{'_pilot' if args.pilot else ''}.jsonl"
probe = P.Probe(args.url)


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=10).stdout


def sysmem():
    d = {}
    for line in sh("vm_stat").splitlines():
        for k in ("Pageins", "Pageouts", "Swapins", "Swapouts"):
            if line.startswith(k + ":"):
                d[k.lower()] = int(line.split(":")[1].strip().rstrip("."))
    try:
        d["swap_used_mb"] = float(sh("sysctl -n vm.swapusage").split("used =")[1].split("M")[0])
    except Exception:
        pass
    try:
        d["free_pct"] = int(sh("memory_pressure | grep 'free percentage'").split(":")[1].strip().rstrip("%"))
    except Exception:
        pass
    try:
        d["server_rss_mb"] = round(int(sh(f"ps -o rss= -p {args.pid}").strip()) / 1024, 1)
    except Exception:
        pass
    return d


def reset_slot():
    req = urllib.request.Request(args.url + "/slots/0?action=erase", data=b"{}", method="POST",
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def scan(text):
    t0 = time.perf_counter()
    rows = []
    for d in intents:
        p, T = probe.p_true(text, P.question_of(d), False, bool(args.cache))
        rows.append({"intent": d["id"], "p_true": p, "prompt_n": T["prompt_n"], "cache_n": T["cache_n"],
                     "server_prompt_ms": T["server_prompt_ms"], "http_ms": T["http_ms"]})
    return (time.perf_counter() - t0) * 1000, rows


# warm-up (not part of the dataset)
for txt in ("Please send the quarterly numbers to the team before Friday, thanks.",
            "Reminder: the office will be closed on Monday for the public holiday."):
    reset_slot()
    scan(txt)

fh = out_path.open("w")
n_prompts = args.pilot or 50
for rep in range(1, args.reps + 1):
    order = list(range(50))
    random.Random(args.seed + rep).shuffle(order)
    for idx in order[:n_prompts]:
        pr = prompts[idx]
        for phase in (["cold", "warm"] if args.warm else ["cold"]):
            if phase == "cold":
                reset_slot()
            before = sysmem()
            ms, rows = scan(pr["text"])
            after = sysmem()
            rec = {"label": args.label, "rep": rep, "phase": phase, "id": pr["id"], "bucket": pr["bucket"],
                   "words": pr["word_count"], "e2e_ms": ms, "intents": rows,
                   "cold_valid": (rows[0]["cache_n"] == 0) if phase == "cold" else None,
                   "mem_before": before, "mem_after": after}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
    print(f"{args.label} rep {rep} done", flush=True)
fh.close()
print("DONE", out_path)
