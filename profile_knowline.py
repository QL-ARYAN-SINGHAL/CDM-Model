"""Isolated latency profiler for the KnowLine path. Read-only: it sends /completion requests to a llama-server and never
touches the production API, CSV files or model server settings.

It re-implements knowline_bridge._Scorer.p_true step by step (same render, same tokenizer, same request body, same
urllib call) with a timer around every step, and records the server's own timing block for each pass.

Usage:
  profile_knowline.py matrix  URL PID LABEL      the 8-scenario baseline (25/100/250/500 words x 1/3 intents)
  profile_knowline.py cold    URL PID LABEL      first requests after a fresh server start
  profile_knowline.py verify  URL                check the profiler returns the same score as production p_true
"""
import json
import math
import os
import random
import statistics
import subprocess
import sys
import threading
import time
import types
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
OUT = ROOT / "eval_results" / "latency_investigation"
OUT.mkdir(parents=True, exist_ok=True)
import build_eval_set as hand  # noqa: E402
import build_long_set as L  # noqa: E402
import knowline_bridge as kb  # noqa: E402

DEFS = hand.defs()
DEFS.update(L.SHORT)
NAMES = list(DEFS)
POOLS = L.pools()
MESSY_TAGS = ["btw", "tbh", "lol", "fyi", "ok so", "anyway", "not sure if this matters but"]


def messy(sentence, rng):
    r = rng.random()
    if r < 0.30:
        sentence = sentence.lower()
    if rng.random() < 0.2:
        sentence = sentence.rstrip(".")
    if rng.random() < 0.15:
        sentence = rng.choice(MESSY_TAGS) + " " + sentence
    return sentence


def make_text(words, rng, present, hard):
    """Messy workplace text of about `words` words with positive sentences for `present` intents and near-miss
    sentences (misleading keywords) for `hard` intents."""
    core = [s for n in present for s in rng.sample(POOLS[n][0], 1)] + [s for n in hard for s in rng.sample(POOLS[n][1], 1)]
    budget = max(words - sum(len(c.split()) for c in core), 10)
    sentences, total, queue = [], 0, []
    while total < budget:
        if not queue:
            queue = rng.sample(L.NEUTRAL, len(L.NEUTRAL))
        s = queue.pop()
        sentences.append(s)
        total += len(s.split())
    for c in core:
        sentences.insert(rng.randint(0, len(sentences)), c)
    return " ".join(messy(s, rng) for s in sentences)


def question_of(d):
    return {"type": "noul", "instructions": d["description"], "criteria": {"true": d["do"], "false": d["dont"]}}


class Probe:
    """p_true with a timer around every stage; mirrors knowline_bridge._Scorer.p_true (lines 135-155)."""

    def __init__(self, url):
        self.url = url.rstrip("/")
        self.sc = kb._Scorer(read_twice=False, cache=False)

    def p_true(self, state, question, twice=False, cache=False, n_probs=200):
        T = {}
        t = time.perf_counter()
        rendered = self.sc._render_request({"state": state, "questions": {"q": question}}, self.sc.tok, self.sc.layouts[twice],
                                           labels=self.sc.labels)
        text, keys = rendered["prompts"]["q"][0]
        T["render_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        ids = self.sc.tok.encode(text, add_special_tokens=False)
        T["tokenize_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        payload = json.dumps({"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": n_probs, "cache_prompt": cache}).encode()
        req = urllib.request.Request(self.url + "/completion", data=payload, headers={"Content-Type": "application/json"})
        T["json_encode_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        with urllib.request.urlopen(req, timeout=180) as response:
            raw = response.read()
        T["http_ms"] = (time.perf_counter() - t) * 1000
        t = time.perf_counter()
        data = json.loads(raw)
        top = data["completion_probabilities"][0]["top_logprobs"]
        found = {e["id"]: e["logprob"] for e in top}
        logps = [found.get(i, -math.inf) for i in self.sc.label_ids[:len(keys)]]
        peak = max(logps)
        weights = [math.exp(x - peak) for x in logps]
        p = dict(zip(keys, (w / sum(weights) for w in weights)))["true"]
        T["parse_ms"] = (time.perf_counter() - t) * 1000
        tm = data["timings"]
        T.update(prompt_n=tm["prompt_n"], cache_n=tm.get("cache_n", 0), server_prompt_ms=tm["prompt_ms"],
                 server_predicted_ms=tm["predicted_ms"], server_prompt_tps=tm["prompt_per_second"], resp_bytes=len(raw),
                 text_chars=len(text))
        T["http_overhead_ms"] = T["http_ms"] - tm["prompt_ms"] - tm["predicted_ms"]
        return p, T

    def request(self, text, intents, twice=False, cache=False, n_probs=200):
        """One scan of `intents` definitions over `text`, sequential, whole message (production compute_scan)."""
        t0 = time.perf_counter()
        passes, scores = [], []
        for d in intents:
            p, T = self.p_true(text, question_of(d), twice, cache, n_probs)
            passes.append(T)
            scores.append(p)
        return {"e2e_ms": (time.perf_counter() - t0) * 1000, "passes": passes, "scores": scores}


def sh(cmd, timeout=5):
    try:
        return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout
    except Exception:
        return ""


def vm_counters():
    out = {}
    for line in sh("vm_stat").splitlines():
        for k in ("Pageins", "Pageouts", "Swapins", "Swapouts"):
            if line.startswith(f'"{k}"') or line.startswith(k + ":"):
                out[k] = int(line.split(":")[1].strip().rstrip("."))
    swap = sh("sysctl -n vm.swapusage")
    try:
        out["swap_used_mb"] = float(swap.split("used =")[1].split("M")[0])
    except Exception:
        pass
    return out


class Sampler(threading.Thread):
    """1 Hz: GPU device utilisation (ioreg), llama-server CPU %, swap in use. Also flags production-server activity."""

    def __init__(self, pid, prod_url="http://127.0.0.1:8009"):
        super().__init__(daemon=True)
        self.pid, self.prod_url, self.stop_flag, self.rows = pid, prod_url, threading.Event(), []

    def run(self):
        while not self.stop_flag.is_set():
            gpu = sh("ioreg -r -d 1 -w0 -c IOAccelerator")
            util = None
            if '"Device Utilization %"=' in gpu:
                try:
                    util = int(gpu.split('"Device Utilization %"=')[1].split(",")[0].split("}")[0])
                except Exception:
                    pass
            cpu = sh(f"ps -o %cpu= -p {self.pid}").strip()
            busy = None
            try:
                slots = json.load(urllib.request.urlopen(self.prod_url + "/slots", timeout=1))
                busy = any(s.get("is_processing") for s in slots)
            except Exception:
                pass
            self.rows.append({"t": time.time(), "gpu_util": util, "cpu": float(cpu) if cpu else None, "prod_busy": busy})
            self.stop_flag.wait(1.0)

    def summary(self):
        self.stop_flag.set()
        g = [r["gpu_util"] for r in self.rows if r["gpu_util"] is not None]
        c = [r["cpu"] for r in self.rows if r["cpu"] is not None]
        return {"samples": len(self.rows), "gpu_util_mean": round(statistics.mean(g), 1) if g else None, "gpu_util_max": max(g) if g else None,
                "server_cpu_mean": round(statistics.mean(c), 1) if c else None, "server_cpu_max": max(c) if c else None,
                "production_busy_samples": sum(1 for r in self.rows if r["prod_busy"])}


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(round(q * (len(v) - 1))))]


def stats(v):
    return {"n": len(v), "mean": round(statistics.mean(v), 1), "min": round(min(v), 1), "p50": round(pct(v, .5), 1), "p90": round(pct(v, .9), 1),
            "p95": round(pct(v, .95), 1), "p99": round(pct(v, .99), 1), "max": round(max(v), 1)}


MATRIX = [(25, 1), (25, 3), (100, 1), (100, 3), (250, 1), (250, 3), (500, 1), (500, 3)]
ITERS = {(250, 1): 100, (250, 3): 100}  # others 50


def corpus(words, n_intents, i, rng):
    triple = rng.sample(NAMES, 3)
    asked = triple[:n_intents]
    present = [n for n in asked if rng.random() < 0.5]
    hard = [n for n in asked if n not in present and rng.random() < 0.6]
    return make_text(words, rng, present, hard), [DEFS[n] for n in asked]


def run_matrix(url, pid, label):
    probe = Probe(url)
    rng = random.Random(20261106)
    raw_path = OUT / f"baseline_raw_{label}.jsonl"
    summary = []
    with raw_path.open("w") as raw:
        for words, n in MATRIX:
            iters = ITERS.get((words, n), 50)
            for _ in range(5):  # warm-up, not recorded
                text, intents = corpus(words, n, 0, rng)
                probe.request(text, intents)
            before, sampler = vm_counters(), Sampler(pid)
            sampler.start()
            rows = []
            for i in range(iters):
                text, intents = corpus(words, n, i, rng)
                r = probe.request(text, intents)
                r.update(words=len(text.split()), n_intents=n, config=f"{words}w x {n}")
                rows.append(r)
                raw.write(json.dumps(r) + "\n")
            res = sampler.summary()
            after = vm_counters()
            e2e = [r["e2e_ms"] for r in rows]
            def stage(key):
                return statistics.mean(sum(p[key] for p in r["passes"]) for r in rows)
            row = {"config": f"{words}w x {n}", "target_words": words, "actual_words_mean": round(statistics.mean(r["words"] for r in rows)), "intents": n,
                   "prompt_tokens_per_pass_mean": round(statistics.mean(p["prompt_n"] for r in rows for p in r["passes"])),
                   "prompt_tokens_per_request_mean": round(statistics.mean(sum(p["prompt_n"] for p in r["passes"]) for r in rows)),
                   **{f"e2e_{k}": v for k, v in stats(e2e).items()},
                   "stage_render_ms": round(stage("render_ms"), 2), "stage_tokenize_ms": round(stage("tokenize_ms"), 2),
                   "stage_json_encode_ms": round(stage("json_encode_ms"), 2), "stage_http_ms": round(stage("http_ms"), 1),
                   "stage_server_prompt_ms": round(stage("server_prompt_ms"), 1), "stage_server_predicted_ms": round(stage("server_predicted_ms"), 1),
                   "stage_http_overhead_ms": round(stage("http_overhead_ms"), 1), "stage_parse_ms": round(stage("parse_ms"), 2),
                   "server_prompt_tps_mean": round(statistics.mean(p["server_prompt_tps"] for r in rows for p in r["passes"]), 0),
                   "resp_kb_mean": round(statistics.mean(p["resp_bytes"] for r in rows for p in r["passes"]) / 1024, 1),
                   "cache_n_mean": round(statistics.mean(p["cache_n"] for r in rows for p in r["passes"]), 1),
                   **{f"res_{k}": v for k, v in res.items()},
                   **{f"vm_delta_{k}": after.get(k, 0) - before.get(k, 0) for k in ("Pageins", "Pageouts", "Swapins", "Swapouts")},
                   "swap_used_mb": after.get("swap_used_mb")}
            summary.append(row)
            print(f"{row['config']:10} tokens/req {row['prompt_tokens_per_request_mean']:5}  e2e mean {row['e2e_mean']:7.0f}  p50 {row['e2e_p50']:7.0f}  p95 {row['e2e_p95']:7.0f}  "
                  f"| server prefill {row['stage_server_prompt_ms']:7.0f}  other http {row['stage_http_overhead_ms']:5.1f}  tok {row['stage_tokenize_ms']:5.2f}  render {row['stage_render_ms']:5.2f}  "
                  f"| gpu {row['res_gpu_util_mean']}% cpu {row['res_server_cpu_mean']}%  prod-busy {row['res_production_busy_samples']}  swapΔ {row['vm_delta_Swapouts']}", flush=True)
    (OUT / f"baseline_summary_{label}.json").write_text(json.dumps(summary, indent=1))


def run_cold(url, pid, label):
    probe = Probe(url)
    rng = random.Random(7)
    rows = []
    for i in range(8):
        text, intents = corpus(250, 3, i, rng)
        r = probe.request(text, intents)
        rows.append({"request_index": i, "e2e_ms": round(r["e2e_ms"], 1), "first_pass_http_ms": round(r["passes"][0]["http_ms"], 1),
                     "first_pass_server_prompt_ms": round(r["passes"][0]["server_prompt_ms"], 1), "prompt_n": r["passes"][0]["prompt_n"]})
        print(rows[-1], flush=True)
    (OUT / f"cold_start_{label}.json").write_text(json.dumps(rows, indent=1))


def run_verify(url):
    probe = Probe(url)
    kb.KNOWLINE_URL = url.rstrip("/")
    rng = random.Random(3)
    worst = 0
    for i in range(10):
        text, intents = corpus(250, 3, i, rng)
        for d in intents:
            a, _ = probe.p_true(text, question_of(d))
            b = probe.sc.p_true(text, question_of(d))
            worst = max(worst, abs(a - b))
    print("max |profiler - production p_true| over 30 scores:", worst)


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "matrix":
        run_matrix(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif mode == "cold":
        run_cold(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    elif mode == "verify":
        run_verify(sys.argv[2])
