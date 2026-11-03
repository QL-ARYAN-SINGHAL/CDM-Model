"""Read-only experiments for the KnowLine latency investigation (they only send /completion requests; nothing is changed).

  experiments_knowline.py tokens
  experiments_knowline.py cache        URL         prefix-cache reuse across 3 intents on the same message
  experiments_knowline.py concurrency  URL_NP1 URL_NP3   3 intents in parallel vs in sequence
  experiments_knowline.py nprobs       URL         n_probs 200 vs 20 vs 2: payload, parse time, score equality
  experiments_knowline.py twice        URL         read twice vs read once, same texts: latency and scores (250 words x 3 intents)
  experiments_knowline.py suite-twice  URL         read twice on eval_suite100.json, paired with the saved read-once scores
"""
import json
import random
import statistics
import sys
import threading
import time
from pathlib import Path

import profile_knowline as P

ROOT = Path(__file__).resolve().parent
OUT = P.OUT


def texts(n, words=250, seed=11):
    rng = random.Random(seed)
    return [P.corpus(words, 3, i, rng) for i in range(n)]  # (text, 3 intent definitions)


def tokens():
    probe = P.Probe("http://127.0.0.1:8035")
    tok = probe.sc.tok
    n = lambda s: len(tok.encode(s, add_special_tokens=False))
    rows = []
    for text, intents in texts(10):
        q = P.question_of(intents[0])
        render = lambda state, question, twice: probe.sc._render_request({"state": state, "questions": {"q": question}}, tok, probe.sc.layouts[twice], labels=probe.sc.labels)["prompts"]["q"][0][0]
        empty_q = {"type": "noul", "instructions": "", "criteria": {"true": "", "false": ""}}
        full1, full2 = n(render(text, q, False)), n(render(text, q, True))
        base = n(render("", empty_q, False))
        msg = n(text)
        definition = n(q["instructions"] + q["criteria"]["true"] + q["criteria"]["false"])
        rows.append({"words": len(text.split()), "full_once": full1, "full_twice": full2, "template_instruction_scaffold": base, "message": msg, "definition": definition,
                     "other": full1 - base - msg - definition})
    summary = {k: round(statistics.mean(r[k] for r in rows), 1) for k in rows[0]}
    print("token composition of one pass (mean of 10 prompts):", json.dumps(summary))
    (OUT / "token_composition.json").write_text(json.dumps({"rows": rows, "mean": summary}, indent=1))


def cache(url):
    probe = P.Probe(url)
    out = []
    for text, intents in texts(20):
        for mode, cache_flag in (("cache_prompt=false (production)", False), ("cache_prompt=true", True)):
            passes = []
            t0 = time.perf_counter()
            for d in intents:
                _, T = probe.p_true(text, P.question_of(d), False, cache_flag)
                passes.append(T)
            out.append({"mode": mode, "e2e_ms": (time.perf_counter() - t0) * 1000, "prompt_n": [p["prompt_n"] for p in passes],
                        "cache_n": [p["cache_n"] for p in passes], "server_ms": [p["server_prompt_ms"] for p in passes]})
        # exact repeat of the same 3-intent request, cache on: the upper bound of reuse
        t0 = time.perf_counter()
        passes = [probe.p_true(text, P.question_of(d), False, True)[1] for d in intents]
        out.append({"mode": "identical request repeated, cache_prompt=true", "e2e_ms": (time.perf_counter() - t0) * 1000,
                    "prompt_n": [p["prompt_n"] for p in passes], "cache_n": [p["cache_n"] for p in passes], "server_ms": [p["server_prompt_ms"] for p in passes]})
    summary = {}
    for mode in dict.fromkeys(r["mode"] for r in out):
        rs = [r for r in out if r["mode"] == mode]
        summary[mode] = {"e2e_ms_mean": round(statistics.mean(r["e2e_ms"] for r in rs), 1),
                         "tokens_processed_per_pass_mean": [round(statistics.mean(r["prompt_n"][i] for r in rs), 1) for i in range(3)],
                         "tokens_reused_per_pass_mean": [round(statistics.mean(r["cache_n"][i] for r in rs), 1) for i in range(3)],
                         "server_prompt_ms_per_pass_mean": [round(statistics.mean(r["server_ms"][i] for r in rs), 1) for i in range(3)]}
    print(json.dumps(summary, indent=1))
    (OUT / "cache_reuse.json").write_text(json.dumps({"raw": out, "summary": summary}, indent=1))


def concurrency(url1, url3):
    results = {}
    for name, url in (("-np 1 server (production setting)", url1), ("-np 3 server", url3)):
        probe = P.Probe(url)
        seq, par = [], []
        for text, intents in texts(15, seed=21):
            t0 = time.perf_counter()
            for d in intents:
                probe.p_true(text, P.question_of(d))
            seq.append((time.perf_counter() - t0) * 1000)
            errs = []
            def worker(d):
                try:
                    probe.p_true(text, P.question_of(d))
                except Exception as e:  # noqa: BLE001
                    errs.append(str(e))
            t0 = time.perf_counter()
            threads = [threading.Thread(target=worker, args=(d,)) for d in intents]
            [t.start() for t in threads]
            [t.join() for t in threads]
            par.append((time.perf_counter() - t0) * 1000)
            if errs:
                print("errors", errs[:1])
        results[name] = {"sequential_ms_mean": round(statistics.mean(seq), 1), "parallel_ms_mean": round(statistics.mean(par), 1),
                         "parallel_ms_p95": round(P.pct(par, .95), 1)}
        print(name, results[name], flush=True)
    (OUT / "concurrency.json").write_text(json.dumps(results, indent=1))


def nprobs(url):
    probe = P.Probe(url)
    rows = {k: [] for k in (200, 20, 2)}
    scores = {k: [] for k in rows}
    for text, intents in texts(30, seed=31):
        for k in rows:
            try:
                p, T = probe.p_true(text, P.question_of(intents[0]), False, False, k)
            except Exception as e:  # noqa: BLE001
                rows[k].append({"error": str(e)[:80]})
                continue
            rows[k].append(T)
            scores[k].append(p)
    summary = {}
    for k, rs in rows.items():
        ok = [r for r in rs if "error" not in r]
        summary[k] = {"ok": len(ok), "resp_kb_mean": round(statistics.mean(r["resp_bytes"] for r in ok) / 1024, 1) if ok else None,
                      "parse_ms_mean": round(statistics.mean(r["parse_ms"] for r in ok), 3) if ok else None,
                      "http_overhead_ms_mean": round(statistics.mean(r["http_overhead_ms"] for r in ok), 2) if ok else None,
                      "max_score_diff_vs_200": round(max((abs(a - b) for a, b in zip(scores[k], scores[200])), default=0), 6)}
    print(json.dumps(summary, indent=1))
    (OUT / "n_probs.json").write_text(json.dumps(summary, indent=1))


def twice(url):
    probe = P.Probe(url)
    rows = []
    for text, intents in texts(25, seed=41):
        once = probe.request(text, intents, twice=False)
        two = probe.request(text, intents, twice=True)
        rows.append({"words": len(text.split()), "once_ms": once["e2e_ms"], "twice_ms": two["e2e_ms"], "once_tokens": sum(p["prompt_n"] for p in once["passes"]),
                     "twice_tokens": sum(p["prompt_n"] for p in two["passes"]), "once_scores": once["scores"], "twice_scores": two["scores"]})
    s = {"once_ms_mean": round(statistics.mean(r["once_ms"] for r in rows), 1), "twice_ms_mean": round(statistics.mean(r["twice_ms"] for r in rows), 1),
         "once_ms_p95": round(P.pct([r["once_ms"] for r in rows], .95), 1), "twice_ms_p95": round(P.pct([r["twice_ms"] for r in rows], .95), 1),
         "once_tokens_mean": round(statistics.mean(r["once_tokens"] for r in rows)), "twice_tokens_mean": round(statistics.mean(r["twice_tokens"] for r in rows))}
    print(json.dumps(s))
    (OUT / "read_twice_latency.json").write_text(json.dumps({"rows": rows, "summary": s}, indent=1))


def suite_twice(url):
    probe = P.Probe(url)
    cases = json.loads((ROOT / "eval_suite100.json").read_text())[:40]  # first 40 prompts, to save time
    rows = []
    for i, c in enumerate(cases):
        r = probe.request(c["text"], c["intents"], twice=True)
        rows.append({"words": len(c["text"].split()), "expected": c["expected"], "scores": r["scores"], "ms": r["e2e_ms"]})
        if i % 25 == 0:
            print("suite-twice", i, flush=True)
    (OUT / "suite100_read_twice.json").write_text(json.dumps({"rows": rows}, indent=1))
    print("done", round(statistics.mean(r["ms"] for r in rows)), "ms mean")


if __name__ == "__main__":
    {"tokens": lambda: tokens(), "cache": lambda: cache(sys.argv[2]), "concurrency": lambda: concurrency(sys.argv[2], sys.argv[3]),
     "nprobs": lambda: nprobs(sys.argv[2]), "twice": lambda: twice(sys.argv[2]), "suite-twice": lambda: suite_twice(sys.argv[2])}[sys.argv[1]]()
