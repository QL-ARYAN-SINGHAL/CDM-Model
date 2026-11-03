"""Generates ../KNOWLINE_OPTIMIZATION_EXPERIMENTS.md from results/summary.json and results/scores_*_vs_A.json."""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
S = json.loads((HERE / "results" / "summary.json").read_text())
DS = json.loads((HERE / "frozen_dataset.json").read_text())
SHA = (HERE / "frozen_dataset.json.sha256").read_text().strip()
BK = ["50-75", "76-125", "126-175", "176-225", "226-250"]
sc = {l: {(x["id"], x["intent"]): x for x in json.loads((HERE / "results" / f"scores_{l}_vs_A.json").read_text())} for l in ("C", "U64")}
exp = {p["id"]: p for p in DS["prompts"]}
f = lambda v: "-" if v is None else f"{v:,.0f}"


def lat_row(name, o):
    c = o["cold"]
    return f"| {name} | {c['n']} | {f(c['mean'])} | {f(c['p50'])} | {f(c['p95'])} | {f(c['max'])} |"


def tok(o, ph):
    t = o.get(f"{ph}_tokens")
    return "-" if not t else " / ".join(f"{t[i]['processed']:.0f}" for i in "ABC"), "-" if not t else " / ".join(f"{t[i]['reused']:.0f}" for i in "ABC")


def bucket_table():
    lines = ["| Words | A cold P95 | C cold P95 | C warm P95 | -ub 64 cold P95 (partial) |", "|---|---:|---:|---:|---:|"]
    for b in BK:
        lines.append(f"| {b} | {f(S['A']['cold_by_bucket'][b]['p95'])} | {f(S['C']['cold_by_bucket'][b]['p95'])} | "
                     f"{f(S['C']['warm_by_bucket'][b]['p95'])} | {f(S['U64']['cold_by_bucket'][b]['p95'])} (n={S['U64']['cold_by_bucket'][b]['n']}) |")
    return "\n".join(lines)


def bucket_detail():
    lines = ["| Words | Config | n | mean | P50 | P95 | max |", "|---|---|---:|---:|---:|---:|---:|"]
    for b in BK:
        for l, key, name in (("A", "cold_by_bucket", "A cold"), ("C", "cold_by_bucket", "C cold"), ("C", "warm_by_bucket", "C warm"), ("U64", "cold_by_bucket", "-ub 64 cold")):
            v = S[l][key][b]
            lines.append(f"| {b} | {name} | {v['n']} | {f(v['mean'])} | {f(v['p50'])} | {f(v['p95'])} | {f(v['max'])} |")
    return "\n".join(lines)


def decisions():
    lines = ["| Prompt | Words | Intent | Provisional label | A score | C score | C diff | -ub 64 score | -ub 64 diff | A / C / -ub 64 decision (0.5) |", "|---|---:|---|---|---:|---:|---:|---:|---:|---|"]
    for p in DS["prompts"]:
        for i in "ABC":
            a = sc["C"][(p["id"], i)]["base"]
            c = sc["C"][(p["id"], i)]["cand"]
            u = sc["U64"][(p["id"], i)]["cand"]
            d = lambda x: "Yes" if x >= 0.5 else "No"
            lines.append(f"| {p['id']} | {p['word_count']} | {i} | {'Yes' if p['expected'][i] else 'No'} | {a:.4f} | {c:.4f} | {abs(c-a):.4f} | {u:.4f} | {abs(u-a):.4f} | {d(a)} / {d(c)} / {d(u)} |")
    return "\n".join(lines)


def vs(l):
    v = S[l]["vs_baseline_A"]
    return v


acc = S["A"]["accuracy_vs_provisional"]["per_intent"]
cA, cC, cU = S["A"], S["C"], S["U64"]
tA, rA = tok(cA, "cold"); tC, rC = tok(cC, "cold"); tCw, rCw = tok(cC, "warm"); tU, rU = tok(cU, "cold")

md = f"""# KnowLine-4B-Gen5.5 Q5_K_M: latency optimization experiments

Machine: Apple M4 Pro, 24 GB. Engine: llama.cpp 0.8.0 (build 11960, commit `8642541b2`). Date: 2026-11-03.
Production server (port 8009) and its bridge were never modified. Every experiment ran on an isolated server on port 8038, one server at a time.
**Nothing was deployed.**

## Scope and what was and was not run

| Config | Flags (all with `-c 8192 -np 1`, plus `--slot-save-path` for slot reset) | Status |
|---|---|---|
| A | `-b 2048 -ub 2048` (production), `cache_prompt=false` as in the bridge | Run in full: 50 prompts x 3 intents x 3 reps = 450 scores |
| C | `-b 2048 -ub 128 --checkpoint-min-step 0 --ctx-checkpoints 32`, `cache_prompt=true` | Run in full, cold and warm: 300 scans |
| `-ub 64` | `-b 2048 -ub 64 --checkpoint-min-step 0 --ctx-checkpoints 32`, `cache_prompt=true` | **Stopped early at the user's request** after 167 of 300 records (all of repetition 1, part of repetition 2). Partial. |
| B | `-b 2048 -ub 2048 --checkpoint-min-step 0 --ctx-checkpoints 32` | **Not run in full** (dropped by the user). Only a 3-prompt x 3-rep pilot (18 records, `results/B_pilot.jsonl`). |
| D | C's flags with caching disabled (micro-batch control) | **Not run** (dropped by the user). |

Because D was not run, the small score differences of C versus A cannot be split into a micro-batch effect and a cache-reuse effect.

Dataset: `knowline_opt/frozen_dataset.json`, SHA-256 `{SHA}` (verified at the start of every run). 50 unique prompts, 10 per word bucket (50-75, 76-125, 126-175, 176-225, 226-250), all within 50-250 words (`len(text.split())`), 3 fixed intents, 150 decisions per configuration. **Labels are provisional** (written by the benchmark author, not independently reviewed). No claim of verified accuracy is made. Some distractor paragraphs are reused across several prompts, a known weakness.

Scoring path: the production bridge's `p_true` (read-once layout, `n_probs=200`, softmax over the two answer labels). Latency is wall time of the complete three-intent scan (render + tokenize + HTTP + parse for A, B, C in order), excluding the slot reset. Percentiles use linear interpolation between order statistics. Cold = slot erased first (`POST /slots/0?action=erase`), Intent A must report zero reused tokens; B and C may reuse A's checkpoints. Warm = the identical scan repeated right after, no reset.

## A. Baseline (config A)

| Metric | Value |
|---|---|
| Cold scans | n = {cA['cold']['n']} (50 prompts x 3 reps) |
| Mean / P50 / P95 / max | {f(cA['cold']['mean'])} / {f(cA['cold']['p50'])} / {f(cA['cold']['p95'])} / {f(cA['cold']['max'])} ms |
| Tokens processed per scan (Intent A / B / C) | {tA} (about {cA['cold_total_processed']:.0f} total) |
| Tokens reused | {rA} |
| Accuracy vs provisional labels (threshold 0.5) | {cA['accuracy_vs_provisional']['correct']}/150; Intent B has 6 false positives, none missed |
| Memory | free memory never below {cA['memory']['min_free_pct']}%, swap used {cA['memory']['swap_used_mb_range'][0]:,.0f}-{cA['memory']['swap_used_mb_range'][1]:,.0f} MB, server RSS up to {cA['memory']['server_rss_mb_max']:,.0f} MB |

The earlier 2.31 s mean for a 250-word prompt is consistent with A's 226-250 bucket (mean {f(cA['cold_by_bucket']['226-250']['mean'])} ms). The earlier "1,259 processed tokens" for one request is plausible but not reproduced exactly: here the per-scan total is about {cA['cold_total_processed']:.0f} on average across all buckets, and the earlier token-composition file shows about 435 tokens per pass for a 262-word prompt (about 1,300 per three-pass scan).

## B. Root cause

1. Every intent is a separate prompt. The rendered text is `[scaffold + message][question block: instruction, Do, Don't][answer slot]`. The message comes first, so the three prompts share a long prefix, about {cC['cold_tokens']['B']['reused']:.0f}-{cC['cold_tokens']['C']['reused']:.0f} tokens on average, but the server recomputes it for each intent: about {cA['cold_total_processed']:.0f} tokens per scan in A, none reused.
2. The model is a hybrid/recurrent architecture. Its recurrent state cannot be rolled back to an arbitrary earlier token, so llama.cpp reuses a prefix only if a saved context checkpoint exists at or before the shared boundary.
3. With production settings (`-ub 2048`), the whole prompt is processed as one batch. In the B pilot (`-ub 2048` with `--checkpoint-min-step 0`, caching on) reuse was **0 tokens in every pass, even for an identical request sent twice**. So `--checkpoint-min-step 0` alone is not sufficient. (Pilot: 9 prompts, 18 scans; no checkpoint log lines were captured, so the mechanism is inferred from reuse counts, not observed.)
4. With `-ub 128` (config C) reuse works: B and C reuse about 214-216 tokens and process only 134 and 117. Reuse stops at a checkpoint boundary, so part of the shared prefix is still recomputed.
5. With `-ub 64` and `-b 2048` the reuse was **zero** in every measured request. An earlier script (`experiment_checkpoint.py`) with `-b 512 -ub 64` did show reuse. The cause of the difference was not investigated.
6. Per-pass server times in C (cold): Intent A about 679 ms for about 346 tokens (roughly 510 tokens/s), Intent B about 302 ms for 134 tokens, Intent C about 264 ms for 117 tokens. Intent A therefore costs about 54% of cold latency.

## C. Experiment results

| Configuration | Tokens processed (A / B / C intent) | Reused tokens | Mean latency | P95 | Accuracy vs provisional labels | Decision agreement with A |
|---|---|---|---:|---:|---:|---:|
| A: original Q5 baseline (cold) | {tA} | {rA} | {f(cA['cold']['mean'])} ms | {f(cA['cold']['p95'])} ms | {cA['accuracy_vs_provisional']['correct']}/150 | reference |
| C: `-ub 128` + checkpoints, cold | {tC} | {rC} | {f(cC['cold']['mean'])} ms | {f(cC['cold']['p95'])} ms | {cC['accuracy_vs_provisional']['correct']}/150 | {vs('C')['agree_at_0.5']}/150 |
| C: warm (identical repeat) | {tCw} | {rCw} | {f(cC['warm']['mean'])} ms | {f(cC['warm']['p95'])} ms | same scores as C cold | n/a |
| `-ub 64` + checkpoints, cold (partial, n={cU['cold']['n']}) | {tU} | {rU} | {f(cU['cold']['mean'])} ms | {f(cU['cold']['p95'])} ms | {cU['accuracy_vs_provisional']['correct']}/150 (repetition 1) | {vs('U64')['agree_at_0.5']}/150 |
| B: `-ub 2048` + checkpoints | pilot only: 440 / 442 / 427 on the first prompt | 0 / 0 / 0 | not measured on the full set | not measured | not measured | not measured |
| Shared-state execution, runtime tuning (Experiments C/D of the brief beyond the above) | not run | not run | not run | not run | not run | not run |

Cold latency distributions (all 150 cold scans for A and C; partial for `-ub 64`):

| Config | n | mean | P50 | P95 | max (ms) |
|---|---:|---:|---:|---:|---:|
{lat_row('A cold', cA)}
{lat_row('C cold', cC)}
{lat_row('-ub 64 cold (partial)', cU)}

Warm (C only): mean {f(cC['warm']['mean'])}, P50 {f(cC['warm']['p50'])}, P95 {f(cC['warm']['p95'])}, max {f(cC['warm']['max'])} ms, n = {cC['warm']['n']}. Cold and warm are never pooled.

Cold-cache validity: Intent A reused 0 tokens on {cA['cold']['n'] - cA['cold_invalid']}/{cA['cold']['n']} cold scans in A, {cC['cold']['n'] - cC['cold_invalid']}/{cC['cold']['n']} in C, {cU['cold']['n'] - cU['cold_invalid']}/{cU['cold']['n']} in `-ub 64`. Repeating the same prompt across repetitions also gave zero reuse for Intent A, so the host-memory prompt cache did not leak across the reset.

Savings of C over A (cold): tokens processed {cA['cold_total_processed']:.0f} -> {cC['cold_total_processed']:.0f} per scan (about {cC['cold_total_reused']:.0f} reused, {100*cC['cold_total_reused']/cA['cold_total_processed']:.0f}% of A's tokens); mean latency {f(cA['cold']['mean'])} -> {f(cC['cold']['mean'])} ms (about {cA['cold']['mean']-cC['cold']['mean']:.0f} ms saved); P95 {f(cA['cold']['p95'])} -> {f(cC['cold']['p95'])} ms.

## D. Accuracy comparison

Summary (repetition 1, 150 decisions, threshold 0.5; status thresholds 0.4 / 0.7):

| Comparison | Decision agreement | 0.4/0.7 status changes | Max abs score diff | Mean abs score diff |
|---|---:|---:|---:|---:|
| C vs A | {vs('C')['agree_at_0.5']}/150 | {vs('C')['status_flips_0.4_0.7']} | {vs('C')['max_abs_diff']:.6f} | {vs('C')['mean_abs_diff']:.6f} |
| `-ub 64` vs A (repetition 1 only) | {vs('U64')['agree_at_0.5']}/150 | {vs('U64')['status_flips_0.4_0.7']} | {vs('U64')['max_abs_diff']:.6f} | {vs('U64')['mean_abs_diff']:.6f} |

- Largest C-vs-A difference: 0.0011 (P42, Intent B, 0.6117 vs 0.6106). No score moves across 0.5, 0.4 or 0.7. The score differences are small and unexplained beyond different batch chunking and restored state; D was not run to attribute them.
- Within each configuration, repeated runs of the same prompt gave identical scores (spread 0.0): A and C are deterministic, so there is no run-to-run variation to report.
- Against the provisional labels (threshold 0.5), A, C and `-ub 64` all get {cA['accuracy_vs_provisional']['correct']}/150 correct with the same 6 errors, all Intent B false positives and no false negatives. Per intent (tp / fp / fn / tn; precision; recall): A {acc['A']['tp']}/{acc['A']['fp']}/{acc['A']['fn']}/{acc['A']['tn']}, 1.00, 1.00; B {acc['B']['tp']}/{acc['B']['fp']}/{acc['B']['fn']}/{acc['B']['tn']}, {acc['B']['precision']}, {acc['B']['recall']}; C {acc['C']['tp']}/{acc['C']['fp']}/{acc['C']['fn']}/{acc['C']['tn']}, 1.00, 1.00.
- The six Intent B false positives (P08, P37, P42, P44, P49, P50) are mostly prompts that quote or report an injection (a prank note, a scraped line, a defensive disclaimer). The baseline scores them 0.61-0.99. The labels are provisional, so some of these may be label errors. Not investigated further.
- These are 50 prompts. 150/150 agreement is a necessary first gate, not proof of equivalence on production traffic.

All 150 decisions (scores are P(true)):

{decisions()}

## E. Latency by word count

Cold P95 (ms), with `-ub 64` partial (n is small per bucket; not a reliable P95):

{bucket_table()}

Full distributions:

{bucket_detail()}

The longest bucket (226-250 words): A cold P95 {f(cA['cold_by_bucket']['226-250']['p95'])} ms, C cold P95 {f(cC['cold_by_bucket']['226-250']['p95'])} ms. **Not below 1,000 ms.** Only C's warm requests (P95 about {f(cC['warm']['p95'])} ms, flat across lengths) are below 1 s.

## F. Remaining bottlenecks

- Target P95 < 1,000 ms for a complete three-intent request: **not reached.** Best measured cold P95 is {f(cC['cold']['p95'])} ms (config C), about 480 ms over target; every bucket except 50-75 is above 1.1 s at P95.
- The first cold pass (Intent A) is about {679/cC['cold']['mean']*100:.0f}% of C's cold latency: about 346 tokens at roughly 510 tokens/s. No cache can remove it because the message has never been seen. It grows with prompt length (A cold P95 rises from about 1.76 s to 3.17 s in the same bucket range in config A).
- Intents B and C still process 134 and 117 tokens each (about 300 and 264 ms) because reuse stops at a checkpoint boundary and the question block is new. Together that is about 45% of cold latency.
- Warm floor in C is about 810 ms (three scans of roughly 270 ms each, about 120-130 tokens each), so even a perfect cache hit on the message leaves little headroom under 1 s with current per-request cost.
- Memory: free memory stayed at {min(cA['memory']['min_free_pct'], cC['memory']['min_free_pct'])}% or higher and no page-outs occurred in any run, but swap was 21-23 GB of 24.6 GB used. Swap-in counts were nonzero (A: {cA['memory']['swapins']:,}; C: {cC['memory']['swapins']:,}; -ub 64: {cU['memory']['swapins']:,}). A's slowest cold scan (4,790 ms, P50) had 32 swap-ins, so a memory effect on A's tail is possible but not proven. C's slowest scans had 0-12 swap-ins and were not outliers. I did not establish causation either way.
- **Environment caveat:** during the `-ub 64` run, production KnowLine (port 8009) was no longer running (see below), so memory conditions differed from A and C. This does not change the conclusion, since `-ub 64` was slower regardless.

## G. Recommended production implementation (not deployed)

Fastest configuration that preserved all 150 baseline decisions on the frozen set: **C**.

- Server flags (replace the production line in `ensure_knowline.sh`): `llama-server -m <Q5_K_M gguf> --host 127.0.0.1 --port 8009 -c 8192 -np 1 -b 2048 -ub 128 --checkpoint-min-step 0 --ctx-checkpoints 32`.
- Bridge change: enable caching, i.e. set `KNOWLINE_CACHE=1` so `p_true` sends `cache_prompt: true`, and score the three intents back to back in the order A, B, C (the current `compute_scan` already does this sequentially, with the lock held). Do not interleave other requests between the three intents: `-np 1` has a single slot and another request would evict A's checkpoints.
- Resource needs: the experiment server used up to about 4.3 GB RSS; checkpoints add a small amount per saved state (not separately measured). Memory pressure on this machine is already high.
- Operational risks: (1) concurrent requests on one slot will evict checkpoints and fall back to full cost; (2) checkpoint reuse depends on the build's `-ub`/`-b` behavior, which was not understood (see `-ub 64`), so a llama.cpp upgrade could silently change reuse, which should be monitored by logging `cache_n` per intent; (3) small score differences (max 0.0011) exist versus the baseline; (4) the P95 target is still not met.
- Rollback: restore the previous flags and set `KNOWLINE_CACHE=0`, then restart the server (needs approval). No model, quantization or threshold changes are involved.
- Before deployment: independently reviewed labels (the six Intent B disagreements should be adjudicated), a larger labeled set, and repeated runs with production traffic shape.

### Next single highest-value optimization (proposal, needs approval)

Reduce the cost of the first cold pass. It is about 54% of C's cold latency, and reuse cannot help it. Propose a prefill-throughput microbenchmark on the isolated server using Intent A alone on the frozen prompts with caching off, varying `-ub` (128/256/512/1024) and flash attention, checking tokens/s and score differences, before combining the winner with the checkpoint flags. Perfect reuse alone would not reach 1 s: with a first pass of about 680 ms and two reused passes of at least 250 ms each, the floor is about 1.2 s at the current speed, so first-pass speed is the lever that matters.

## H. Follow-up validation

- At least 200 additional independently labeled prompts across the same 50-250-word range, plus the client's real intent definitions; kept separate from this benchmark.
- Production-like traffic test: concurrent requests, mixed intent counts, bursts, and cold starts, to check eviction of the single slot's checkpoints.
- Investigate why `-ub 64` with `-b 2048` gives no reuse, and capture a verbose (`-lv 4`) log of checkpoint creation and restore. This was not done.
- Run D (C's flags with caching off) to attribute the small score differences.
- Re-measure with swap pressure reduced.

## Incident during this work

At the user's request I stopped the cloudflared tunnel for port 8011 (PID 5745, the Verity public tunnel). That tunnel ran in the foreground of `start_public_test.sh` (PID 5672); when it exited, the script's exit trap also stopped the proxy on port 8011, the Verity API it started, and **the production KnowLine server on port 8009** (PID 5696). This was an unintended side effect. Production KnowLine has not been restarted by me.

## Files

`knowline_opt/`: `build_dataset.py`, `frozen_dataset.json` (+ `.sha256`), `run_config.py`, `drive_config.sh`, `analyze.py`, `make_report.py`, `results/{{A,C,U64}}.jsonl`, `results/B_pilot.jsonl`, `results/summary.json`, `results/scores_{{C,U64}}_vs_A.json`, `logs/`.
"""
(HERE.parent / "KNOWLINE_OPTIMIZATION_EXPERIMENTS.md").write_text(md, encoding="utf-8")
print(len(md.splitlines()), "lines")
