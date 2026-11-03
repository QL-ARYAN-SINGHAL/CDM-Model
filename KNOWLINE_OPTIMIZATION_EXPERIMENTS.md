# KnowLine-4B-Gen5.5 Q5_K_M: latency optimization experiments

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

Dataset: `knowline_opt/frozen_dataset.json`, SHA-256 `936330c9bfdb6a796d197add62329f8d8f7d8542d4ae40c226674928c28cb152` (verified at the start of every run). 50 unique prompts, 10 per word bucket (50-75, 76-125, 126-175, 176-225, 226-250), all within 50-250 words (`len(text.split())`), 3 fixed intents, 150 decisions per configuration. **Labels are provisional** (written by the benchmark author, not independently reviewed). No claim of verified accuracy is made. Some distractor paragraphs are reused across several prompts, a known weakness.

Scoring path: the production bridge's `p_true` (read-once layout, `n_probs=200`, softmax over the two answer labels). Latency is wall time of the complete three-intent scan (render + tokenize + HTTP + parse for A, B, C in order), excluding the slot reset. Percentiles use linear interpolation between order statistics. Cold = slot erased first (`POST /slots/0?action=erase`), Intent A must report zero reused tokens; B and C may reuse A's checkpoints. Warm = the identical scan repeated right after, no reset.

## A. Baseline (config A)

| Metric | Value |
|---|---|
| Cold scans | n = 150 (50 prompts x 3 reps) |
| Mean / P50 / P95 / max | 2,064 / 2,018 / 3,008 / 4,790 ms |
| Tokens processed per scan (Intent A / B / C) | 346 / 348 / 333 (about 1026 total) |
| Tokens reused | 0 / 0 / 0 |
| Accuracy vs provisional labels (threshold 0.5) | 144/150; Intent B has 6 false positives, none missed |
| Memory | free memory never below 48%, swap used 22,914-22,986 MB, server RSS up to 4,019 MB |

The earlier 2.31 s mean for a 250-word prompt is consistent with A's 226-250 bucket (mean 2,748 ms). The earlier "1,259 processed tokens" for one request is plausible but not reproduced exactly: here the per-scan total is about 1026 on average across all buckets, and the earlier token-composition file shows about 435 tokens per pass for a 262-word prompt (about 1,300 per three-pass scan).

## B. Root cause

1. Every intent is a separate prompt. The rendered text is `[scaffold + message][question block: instruction, Do, Don't][answer slot]`. The message comes first, so the three prompts share a long prefix, about 214-216 tokens on average, but the server recomputes it for each intent: about 1026 tokens per scan in A, none reused.
2. The model is a hybrid/recurrent architecture. Its recurrent state cannot be rolled back to an arbitrary earlier token, so llama.cpp reuses a prefix only if a saved context checkpoint exists at or before the shared boundary.
3. With production settings (`-ub 2048`), the whole prompt is processed as one batch. In the B pilot (`-ub 2048` with `--checkpoint-min-step 0`, caching on) reuse was **0 tokens in every pass, even for an identical request sent twice**. So `--checkpoint-min-step 0` alone is not sufficient. (Pilot: 9 prompts, 18 scans; no checkpoint log lines were captured, so the mechanism is inferred from reuse counts, not observed.)
4. With `-ub 128` (config C) reuse works: B and C reuse about 214-216 tokens and process only 134 and 117. Reuse stops at a checkpoint boundary, so part of the shared prefix is still recomputed.
5. With `-ub 64` and `-b 2048` the reuse was **zero** in every measured request. An earlier script (`experiment_checkpoint.py`) with `-b 512 -ub 64` did show reuse. The cause of the difference was not investigated.
6. Per-pass server times in C (cold): Intent A about 679 ms for about 346 tokens (roughly 510 tokens/s), Intent B about 302 ms for 134 tokens, Intent C about 264 ms for 117 tokens. Intent A therefore costs about 54% of cold latency.

## C. Experiment results

| Configuration | Tokens processed (A / B / C intent) | Reused tokens | Mean latency | P95 | Accuracy vs provisional labels | Decision agreement with A |
|---|---|---|---:|---:|---:|---:|
| A: original Q5 baseline (cold) | 346 / 348 / 333 | 0 / 0 / 0 | 2,064 ms | 3,008 ms | 144/150 | reference |
| C: `-ub 128` + checkpoints, cold | 346 / 134 / 117 | 0 / 214 / 216 | 1,253 ms | 1,480 ms | 144/150 | 150/150 |
| C: warm (identical repeat) | 130 / 132 / 117 | 216 / 216 / 216 | 813 ms | 839 ms | same scores as C cold | n/a |
| `-ub 64` + checkpoints, cold (partial, n=84) | 345 / 347 / 332 | 0 / 0 / 0 | 2,269 ms | 3,380 ms | 144/150 (repetition 1) | 150/150 |
| B: `-ub 2048` + checkpoints | pilot only: 440 / 442 / 427 on the first prompt | 0 / 0 / 0 | not measured on the full set | not measured | not measured | not measured |
| Shared-state execution, runtime tuning (Experiments C/D of the brief beyond the above) | not run | not run | not run | not run | not run | not run |

Cold latency distributions (all 150 cold scans for A and C; partial for `-ub 64`):

| Config | n | mean | P50 | P95 | max (ms) |
|---|---:|---:|---:|---:|---:|
| A cold | 150 | 2,064 | 2,018 | 3,008 | 4,790 |
| C cold | 150 | 1,253 | 1,242 | 1,480 | 1,573 |
| -ub 64 cold (partial) | 84 | 2,269 | 2,224 | 3,380 | 5,115 |

Warm (C only): mean 813, P50 809, P95 839, max 933 ms, n = 150. Cold and warm are never pooled.

Cold-cache validity: Intent A reused 0 tokens on 150/150 cold scans in A, 150/150 in C, 84/84 in `-ub 64`. Repeating the same prompt across repetitions also gave zero reuse for Intent A, so the host-memory prompt cache did not leak across the reset.

Savings of C over A (cold): tokens processed 1026 -> 597 per scan (about 429 reused, 42% of A's tokens); mean latency 2,064 -> 1,253 ms (about 812 ms saved); P95 3,008 -> 1,480 ms.

## D. Accuracy comparison

Summary (repetition 1, 150 decisions, threshold 0.5; status thresholds 0.4 / 0.7):

| Comparison | Decision agreement | 0.4/0.7 status changes | Max abs score diff | Mean abs score diff |
|---|---:|---:|---:|---:|
| C vs A | 150/150 | 0 | 0.001093 | 0.000043 |
| `-ub 64` vs A (repetition 1 only) | 150/150 | 0 | 0.000329 | 0.000008 |

- Largest C-vs-A difference: 0.0011 (P42, Intent B, 0.6117 vs 0.6106). No score moves across 0.5, 0.4 or 0.7. The score differences are small and unexplained beyond different batch chunking and restored state; D was not run to attribute them.
- Within each configuration, repeated runs of the same prompt gave identical scores (spread 0.0): A and C are deterministic, so there is no run-to-run variation to report.
- Against the provisional labels (threshold 0.5), A, C and `-ub 64` all get 144/150 correct with the same 6 errors, all Intent B false positives and no false negatives. Per intent (tp / fp / fn / tn; precision; recall): A 13/0/0/37, 1.00, 1.00; B 12/6/0/32, 0.667, 1.0; C 14/0/0/36, 1.00, 1.00.
- The six Intent B false positives (P08, P37, P42, P44, P49, P50) are mostly prompts that quote or report an injection (a prank note, a scraped line, a defensive disclaimer). The baseline scores them 0.61-0.99. The labels are provisional, so some of these may be label errors. Not investigated further.
- These are 50 prompts. 150/150 agreement is a necessary first gate, not proof of equivalence on production traffic.

All 150 decisions (scores are P(true)):

| Prompt | Words | Intent | Provisional label | A score | C score | C diff | -ub 64 score | -ub 64 diff | A / C / -ub 64 decision (0.5) |
|---|---:|---|---|---:|---:|---:|---:|---:|---|
| P01 | 67 | A | Yes | 0.9882 | 0.9882 | 0.0000 | 0.9882 | 0.0000 | Yes / Yes / Yes |
| P01 | 67 | B | No | 0.0121 | 0.0122 | 0.0000 | 0.0121 | 0.0000 | No / No / No |
| P01 | 67 | C | No | 0.0206 | 0.0206 | 0.0000 | 0.0206 | 0.0000 | No / No / No |
| P02 | 58 | A | No | 0.0033 | 0.0033 | 0.0000 | 0.0033 | 0.0000 | No / No / No |
| P02 | 58 | B | Yes | 0.9993 | 0.9993 | 0.0000 | 0.9993 | 0.0000 | Yes / Yes / Yes |
| P02 | 58 | C | No | 0.0023 | 0.0023 | 0.0000 | 0.0023 | 0.0000 | No / No / No |
| P03 | 58 | A | No | 0.0020 | 0.0020 | 0.0000 | 0.0020 | 0.0000 | No / No / No |
| P03 | 58 | B | No | 0.0585 | 0.0585 | 0.0000 | 0.0585 | 0.0001 | No / No / No |
| P03 | 58 | C | Yes | 0.9878 | 0.9878 | 0.0000 | 0.9878 | 0.0000 | Yes / Yes / Yes |
| P04 | 59 | A | No | 0.0011 | 0.0011 | 0.0000 | 0.0011 | 0.0000 | No / No / No |
| P04 | 59 | B | No | 0.0009 | 0.0009 | 0.0000 | 0.0009 | 0.0000 | No / No / No |
| P04 | 59 | C | No | 0.0015 | 0.0015 | 0.0000 | 0.0015 | 0.0000 | No / No / No |
| P05 | 58 | A | No | 0.0110 | 0.0110 | 0.0000 | 0.0110 | 0.0000 | No / No / No |
| P05 | 58 | B | No | 0.0130 | 0.0130 | 0.0000 | 0.0130 | 0.0000 | No / No / No |
| P05 | 58 | C | No | 0.0212 | 0.0212 | 0.0000 | 0.0212 | 0.0000 | No / No / No |
| P06 | 60 | A | No | 0.0018 | 0.0018 | 0.0000 | 0.0018 | 0.0000 | No / No / No |
| P06 | 60 | B | No | 0.3814 | 0.3809 | 0.0005 | 0.3814 | 0.0000 | No / No / No |
| P06 | 60 | C | No | 0.0035 | 0.0035 | 0.0000 | 0.0035 | 0.0000 | No / No / No |
| P07 | 58 | A | No | 0.0034 | 0.0034 | 0.0000 | 0.0034 | 0.0000 | No / No / No |
| P07 | 58 | B | No | 0.0666 | 0.0666 | 0.0000 | 0.0666 | 0.0000 | No / No / No |
| P07 | 58 | C | No | 0.2248 | 0.2248 | 0.0000 | 0.2248 | 0.0000 | No / No / No |
| P08 | 54 | A | Yes | 0.9973 | 0.9973 | 0.0000 | 0.9973 | 0.0000 | Yes / Yes / Yes |
| P08 | 54 | B | No | 0.7100 | 0.7099 | 0.0001 | 0.7100 | 0.0000 | Yes / Yes / Yes |
| P08 | 54 | C | Yes | 0.9755 | 0.9755 | 0.0000 | 0.9755 | 0.0000 | Yes / Yes / Yes |
| P09 | 58 | A | No | 0.0099 | 0.0099 | 0.0000 | 0.0099 | 0.0000 | No / No / No |
| P09 | 58 | B | Yes | 0.9987 | 0.9987 | 0.0000 | 0.9987 | 0.0000 | Yes / Yes / Yes |
| P09 | 58 | C | No | 0.0053 | 0.0053 | 0.0000 | 0.0053 | 0.0000 | No / No / No |
| P10 | 62 | A | No | 0.2179 | 0.2179 | 0.0000 | 0.2179 | 0.0000 | No / No / No |
| P10 | 62 | B | No | 0.0320 | 0.0321 | 0.0001 | 0.0320 | 0.0000 | No / No / No |
| P10 | 62 | C | No | 0.0016 | 0.0016 | 0.0000 | 0.0016 | 0.0000 | No / No / No |
| P11 | 93 | A | Yes | 0.9907 | 0.9908 | 0.0000 | 0.9907 | 0.0000 | Yes / Yes / Yes |
| P11 | 93 | B | No | 0.0168 | 0.0168 | 0.0000 | 0.0168 | 0.0000 | No / No / No |
| P11 | 93 | C | No | 0.0194 | 0.0194 | 0.0000 | 0.0194 | 0.0000 | No / No / No |
| P12 | 80 | A | No | 0.0014 | 0.0014 | 0.0000 | 0.0014 | 0.0000 | No / No / No |
| P12 | 80 | B | No | 0.1595 | 0.1597 | 0.0002 | 0.1596 | 0.0001 | No / No / No |
| P12 | 80 | C | Yes | 0.9990 | 0.9990 | 0.0000 | 0.9990 | 0.0000 | Yes / Yes / Yes |
| P13 | 87 | A | No | 0.0020 | 0.0020 | 0.0000 | 0.0020 | 0.0000 | No / No / No |
| P13 | 87 | B | Yes | 0.9870 | 0.9869 | 0.0001 | 0.9870 | 0.0000 | Yes / Yes / Yes |
| P13 | 87 | C | No | 0.0020 | 0.0020 | 0.0000 | 0.0020 | 0.0000 | No / No / No |
| P14 | 86 | A | No | 0.0038 | 0.0038 | 0.0000 | 0.0038 | 0.0000 | No / No / No |
| P14 | 86 | B | No | 0.0402 | 0.0402 | 0.0000 | 0.0402 | 0.0000 | No / No / No |
| P14 | 86 | C | No | 0.0086 | 0.0085 | 0.0000 | 0.0086 | 0.0000 | No / No / No |
| P15 | 80 | A | No | 0.0191 | 0.0191 | 0.0000 | 0.0191 | 0.0000 | No / No / No |
| P15 | 80 | B | No | 0.0145 | 0.0145 | 0.0000 | 0.0145 | 0.0000 | No / No / No |
| P15 | 80 | C | No | 0.0238 | 0.0238 | 0.0000 | 0.0238 | 0.0000 | No / No / No |
| P16 | 78 | A | No | 0.0320 | 0.0320 | 0.0000 | 0.0320 | 0.0000 | No / No / No |
| P16 | 78 | B | No | 0.0009 | 0.0009 | 0.0000 | 0.0009 | 0.0000 | No / No / No |
| P16 | 78 | C | No | 0.0021 | 0.0021 | 0.0000 | 0.0021 | 0.0000 | No / No / No |
| P17 | 91 | A | No | 0.0017 | 0.0017 | 0.0000 | 0.0017 | 0.0000 | No / No / No |
| P17 | 91 | B | No | 0.0459 | 0.0460 | 0.0001 | 0.0459 | 0.0000 | No / No / No |
| P17 | 91 | C | No | 0.1874 | 0.1877 | 0.0003 | 0.1874 | 0.0000 | No / No / No |
| P18 | 87 | A | Yes | 0.9576 | 0.9575 | 0.0001 | 0.9576 | 0.0000 | Yes / Yes / Yes |
| P18 | 87 | B | Yes | 0.9972 | 0.9972 | 0.0000 | 0.9972 | 0.0000 | Yes / Yes / Yes |
| P18 | 87 | C | No | 0.0364 | 0.0363 | 0.0000 | 0.0364 | 0.0000 | No / No / No |
| P19 | 81 | A | No | 0.0189 | 0.0189 | 0.0000 | 0.0189 | 0.0000 | No / No / No |
| P19 | 81 | B | Yes | 0.9995 | 0.9995 | 0.0000 | 0.9995 | 0.0000 | Yes / Yes / Yes |
| P19 | 81 | C | Yes | 0.9983 | 0.9983 | 0.0000 | 0.9983 | 0.0000 | Yes / Yes / Yes |
| P20 | 76 | A | No | 0.0012 | 0.0012 | 0.0000 | 0.0012 | 0.0000 | No / No / No |
| P20 | 76 | B | No | 0.0104 | 0.0104 | 0.0000 | 0.0104 | 0.0000 | No / No / No |
| P20 | 76 | C | No | 0.0133 | 0.0133 | 0.0000 | 0.0133 | 0.0000 | No / No / No |
| P21 | 174 | A | Yes | 0.9468 | 0.9468 | 0.0000 | 0.9468 | 0.0000 | Yes / Yes / Yes |
| P21 | 174 | B | No | 0.0624 | 0.0624 | 0.0000 | 0.0624 | 0.0000 | No / No / No |
| P21 | 174 | C | No | 0.0198 | 0.0197 | 0.0000 | 0.0198 | 0.0000 | No / No / No |
| P22 | 138 | A | No | 0.0013 | 0.0013 | 0.0000 | 0.0013 | 0.0000 | No / No / No |
| P22 | 138 | B | No | 0.2509 | 0.2507 | 0.0003 | 0.2511 | 0.0001 | No / No / No |
| P22 | 138 | C | No | 0.0259 | 0.0259 | 0.0000 | 0.0259 | 0.0000 | No / No / No |
| P23 | 137 | A | No | 0.0016 | 0.0016 | 0.0000 | 0.0016 | 0.0000 | No / No / No |
| P23 | 137 | B | No | 0.1256 | 0.1255 | 0.0000 | 0.1256 | 0.0000 | No / No / No |
| P23 | 137 | C | Yes | 0.9795 | 0.9796 | 0.0000 | 0.9795 | 0.0000 | Yes / Yes / Yes |
| P24 | 156 | A | No | 0.0099 | 0.0099 | 0.0000 | 0.0099 | 0.0000 | No / No / No |
| P24 | 156 | B | Yes | 0.9984 | 0.9984 | 0.0000 | 0.9984 | 0.0000 | Yes / Yes / Yes |
| P24 | 156 | C | No | 0.0022 | 0.0022 | 0.0000 | 0.0022 | 0.0000 | No / No / No |
| P25 | 135 | A | No | 0.0136 | 0.0136 | 0.0000 | 0.0136 | 0.0000 | No / No / No |
| P25 | 135 | B | No | 0.0529 | 0.0528 | 0.0000 | 0.0529 | 0.0000 | No / No / No |
| P25 | 135 | C | No | 0.0162 | 0.0162 | 0.0000 | 0.0162 | 0.0000 | No / No / No |
| P26 | 141 | A | Yes | 0.9969 | 0.9969 | 0.0000 | 0.9969 | 0.0000 | Yes / Yes / Yes |
| P26 | 141 | B | No | 0.0157 | 0.0157 | 0.0000 | 0.0157 | 0.0000 | No / No / No |
| P26 | 141 | C | Yes | 0.9988 | 0.9988 | 0.0000 | 0.9988 | 0.0000 | Yes / Yes / Yes |
| P27 | 132 | A | Yes | 0.9391 | 0.9391 | 0.0000 | 0.9391 | 0.0000 | Yes / Yes / Yes |
| P27 | 132 | B | No | 0.2973 | 0.2967 | 0.0007 | 0.2973 | 0.0000 | No / No / No |
| P27 | 132 | C | No | 0.0028 | 0.0028 | 0.0000 | 0.0028 | 0.0000 | No / No / No |
| P28 | 138 | A | No | 0.4431 | 0.4431 | 0.0000 | 0.4428 | 0.0003 | No / No / No |
| P28 | 138 | B | No | 0.2114 | 0.2116 | 0.0001 | 0.2115 | 0.0001 | No / No / No |
| P28 | 138 | C | No | 0.0099 | 0.0099 | 0.0000 | 0.0099 | 0.0000 | No / No / No |
| P29 | 147 | A | No | 0.0056 | 0.0056 | 0.0000 | 0.0056 | 0.0000 | No / No / No |
| P29 | 147 | B | No | 0.0431 | 0.0431 | 0.0000 | 0.0431 | 0.0000 | No / No / No |
| P29 | 147 | C | Yes | 0.9654 | 0.9654 | 0.0000 | 0.9654 | 0.0000 | Yes / Yes / Yes |
| P30 | 154 | A | No | 0.0025 | 0.0025 | 0.0000 | 0.0025 | 0.0000 | No / No / No |
| P30 | 154 | B | Yes | 0.9989 | 0.9989 | 0.0000 | 0.9989 | 0.0000 | Yes / Yes / Yes |
| P30 | 154 | C | No | 0.0120 | 0.0121 | 0.0000 | 0.0121 | 0.0000 | No / No / No |
| P31 | 198 | A | Yes | 0.9780 | 0.9780 | 0.0000 | 0.9780 | 0.0000 | Yes / Yes / Yes |
| P31 | 198 | B | No | 0.4118 | 0.4111 | 0.0006 | 0.4116 | 0.0001 | No / No / No |
| P31 | 198 | C | No | 0.0236 | 0.0236 | 0.0000 | 0.0236 | 0.0000 | No / No / No |
| P32 | 190 | A | No | 0.0058 | 0.0058 | 0.0000 | 0.0058 | 0.0000 | No / No / No |
| P32 | 190 | B | Yes | 0.9978 | 0.9978 | 0.0000 | 0.9978 | 0.0000 | Yes / Yes / Yes |
| P32 | 190 | C | No | 0.0382 | 0.0382 | 0.0000 | 0.0382 | 0.0000 | No / No / No |
| P33 | 198 | A | No | 0.0018 | 0.0018 | 0.0000 | 0.0018 | 0.0000 | No / No / No |
| P33 | 198 | B | No | 0.0401 | 0.0401 | 0.0001 | 0.0401 | 0.0000 | No / No / No |
| P33 | 198 | C | Yes | 0.9988 | 0.9988 | 0.0000 | 0.9988 | 0.0000 | Yes / Yes / Yes |
| P34 | 197 | A | No | 0.0080 | 0.0080 | 0.0000 | 0.0080 | 0.0000 | No / No / No |
| P34 | 197 | B | No | 0.0015 | 0.0015 | 0.0000 | 0.0015 | 0.0000 | No / No / No |
| P34 | 197 | C | No | 0.0472 | 0.0471 | 0.0001 | 0.0472 | 0.0000 | No / No / No |
| P35 | 181 | A | Yes | 0.9658 | 0.9659 | 0.0000 | 0.9659 | 0.0000 | Yes / Yes / Yes |
| P35 | 181 | B | Yes | 0.9991 | 0.9991 | 0.0000 | 0.9991 | 0.0000 | Yes / Yes / Yes |
| P35 | 181 | C | Yes | 0.9830 | 0.9830 | 0.0000 | 0.9830 | 0.0000 | Yes / Yes / Yes |
| P36 | 212 | A | No | 0.0016 | 0.0016 | 0.0000 | 0.0016 | 0.0000 | No / No / No |
| P36 | 212 | B | No | 0.0008 | 0.0008 | 0.0000 | 0.0008 | 0.0000 | No / No / No |
| P36 | 212 | C | No | 0.0241 | 0.0241 | 0.0000 | 0.0241 | 0.0000 | No / No / No |
| P37 | 185 | A | No | 0.0016 | 0.0016 | 0.0000 | 0.0016 | 0.0000 | No / No / No |
| P37 | 185 | B | No | 0.9903 | 0.9903 | 0.0000 | 0.9903 | 0.0000 | Yes / Yes / Yes |
| P37 | 185 | C | No | 0.0243 | 0.0243 | 0.0000 | 0.0243 | 0.0000 | No / No / No |
| P38 | 192 | A | No | 0.0014 | 0.0014 | 0.0000 | 0.0014 | 0.0000 | No / No / No |
| P38 | 192 | B | No | 0.0197 | 0.0197 | 0.0000 | 0.0197 | 0.0000 | No / No / No |
| P38 | 192 | C | Yes | 0.9977 | 0.9977 | 0.0000 | 0.9977 | 0.0000 | Yes / Yes / Yes |
| P39 | 186 | A | Yes | 0.9691 | 0.9691 | 0.0000 | 0.9691 | 0.0000 | Yes / Yes / Yes |
| P39 | 186 | B | No | 0.0269 | 0.0269 | 0.0000 | 0.0269 | 0.0000 | No / No / No |
| P39 | 186 | C | No | 0.0443 | 0.0443 | 0.0000 | 0.0443 | 0.0000 | No / No / No |
| P40 | 181 | A | No | 0.0193 | 0.0193 | 0.0000 | 0.0193 | 0.0000 | No / No / No |
| P40 | 181 | B | Yes | 0.9994 | 0.9994 | 0.0000 | 0.9994 | 0.0000 | Yes / Yes / Yes |
| P40 | 181 | C | No | 0.0252 | 0.0252 | 0.0000 | 0.0252 | 0.0000 | No / No / No |
| P41 | 239 | A | Yes | 0.9970 | 0.9970 | 0.0000 | 0.9970 | 0.0000 | Yes / Yes / Yes |
| P41 | 239 | B | No | 0.0237 | 0.0238 | 0.0000 | 0.0237 | 0.0000 | No / No / No |
| P41 | 239 | C | No | 0.0212 | 0.0211 | 0.0000 | 0.0212 | 0.0000 | No / No / No |
| P42 | 230 | A | No | 0.0022 | 0.0022 | 0.0000 | 0.0022 | 0.0000 | No / No / No |
| P42 | 230 | B | No | 0.6117 | 0.6106 | 0.0011 | 0.6117 | 0.0000 | Yes / Yes / Yes |
| P42 | 230 | C | No | 0.0041 | 0.0041 | 0.0000 | 0.0041 | 0.0000 | No / No / No |
| P43 | 235 | A | No | 0.0119 | 0.0119 | 0.0000 | 0.0119 | 0.0000 | No / No / No |
| P43 | 235 | B | No | 0.1359 | 0.1359 | 0.0000 | 0.1359 | 0.0000 | No / No / No |
| P43 | 235 | C | Yes | 0.9991 | 0.9991 | 0.0000 | 0.9991 | 0.0000 | Yes / Yes / Yes |
| P44 | 240 | A | No | 0.0018 | 0.0018 | 0.0000 | 0.0018 | 0.0000 | No / No / No |
| P44 | 240 | B | No | 0.8539 | 0.8542 | 0.0003 | 0.8539 | 0.0000 | Yes / Yes / Yes |
| P44 | 240 | C | No | 0.0016 | 0.0016 | 0.0000 | 0.0016 | 0.0000 | No / No / No |
| P45 | 240 | A | Yes | 0.9868 | 0.9868 | 0.0000 | 0.9868 | 0.0000 | Yes / Yes / Yes |
| P45 | 240 | B | Yes | 0.9981 | 0.9981 | 0.0000 | 0.9981 | 0.0000 | Yes / Yes / Yes |
| P45 | 240 | C | Yes | 0.9705 | 0.9706 | 0.0000 | 0.9705 | 0.0000 | Yes / Yes / Yes |
| P46 | 234 | A | No | 0.0103 | 0.0103 | 0.0000 | 0.0103 | 0.0000 | No / No / No |
| P46 | 234 | B | No | 0.0148 | 0.0148 | 0.0000 | 0.0148 | 0.0000 | No / No / No |
| P46 | 234 | C | Yes | 0.9983 | 0.9983 | 0.0000 | 0.9983 | 0.0000 | Yes / Yes / Yes |
| P47 | 242 | A | No | 0.0023 | 0.0023 | 0.0000 | 0.0023 | 0.0000 | No / No / No |
| P47 | 242 | B | Yes | 0.9195 | 0.9196 | 0.0001 | 0.9195 | 0.0000 | Yes / Yes / Yes |
| P47 | 242 | C | No | 0.0022 | 0.0022 | 0.0000 | 0.0022 | 0.0000 | No / No / No |
| P48 | 234 | A | No | 0.3684 | 0.3684 | 0.0000 | 0.3684 | 0.0000 | No / No / No |
| P48 | 234 | B | No | 0.0145 | 0.0144 | 0.0000 | 0.0145 | 0.0000 | No / No / No |
| P48 | 234 | C | No | 0.0075 | 0.0075 | 0.0000 | 0.0075 | 0.0000 | No / No / No |
| P49 | 230 | A | Yes | 0.9953 | 0.9953 | 0.0000 | 0.9953 | 0.0000 | Yes / Yes / Yes |
| P49 | 230 | B | No | 0.6530 | 0.6537 | 0.0007 | 0.6530 | 0.0000 | Yes / Yes / Yes |
| P49 | 230 | C | Yes | 0.9951 | 0.9951 | 0.0000 | 0.9951 | 0.0000 | Yes / Yes / Yes |
| P50 | 235 | A | No | 0.0016 | 0.0016 | 0.0000 | 0.0016 | 0.0000 | No / No / No |
| P50 | 235 | B | No | 0.9644 | 0.9644 | 0.0000 | 0.9644 | 0.0000 | Yes / Yes / Yes |
| P50 | 235 | C | No | 0.0225 | 0.0225 | 0.0000 | 0.0225 | 0.0000 | No / No / No |

## E. Latency by word count

Cold P95 (ms), with `-ub 64` partial (n is small per bucket; not a reliable P95):

| Words | A cold P95 | C cold P95 | C warm P95 | -ub 64 cold P95 (partial) |
|---|---:|---:|---:|---:|
| 50-75 | 1,761 | 1,103 | 824 | 2,287 (n=16) |
| 76-125 | 2,039 | 1,245 | 840 | 2,131 (n=18) |
| 126-175 | 2,421 | 1,308 | 834 | 3,098 (n=17) |
| 176-225 | 2,828 | 1,561 | 828 | 3,169 (n=16) |
| 226-250 | 3,172 | 1,522 | 843 | 4,195 (n=17) |

Full distributions:

| Words | Config | n | mean | P50 | P95 | max |
|---|---|---:|---:|---:|---:|---:|
| 50-75 | A cold | 30 | 1,477 | 1,423 | 1,761 | 1,874 |
| 50-75 | C cold | 30 | 1,067 | 1,062 | 1,103 | 1,110 |
| 50-75 | C warm | 30 | 810 | 808 | 824 | 842 |
| 50-75 | -ub 64 cold | 16 | 1,634 | 1,531 | 2,287 | 2,438 |
| 76-125 | A cold | 30 | 1,675 | 1,628 | 2,039 | 2,152 |
| 76-125 | C cold | 30 | 1,129 | 1,128 | 1,245 | 1,249 |
| 76-125 | C warm | 30 | 818 | 813 | 840 | 890 |
| 76-125 | -ub 64 cold | 18 | 1,795 | 1,726 | 2,131 | 2,859 |
| 126-175 | A cold | 30 | 2,032 | 1,976 | 2,421 | 2,862 |
| 126-175 | C cold | 30 | 1,245 | 1,241 | 1,308 | 1,314 |
| 126-175 | C warm | 30 | 812 | 809 | 834 | 838 |
| 126-175 | -ub 64 cold | 17 | 2,313 | 2,102 | 3,098 | 3,872 |
| 176-225 | A cold | 30 | 2,390 | 2,329 | 2,828 | 3,010 |
| 176-225 | C cold | 30 | 1,367 | 1,355 | 1,561 | 1,573 |
| 176-225 | C warm | 30 | 810 | 807 | 828 | 840 |
| 176-225 | -ub 64 cold | 16 | 2,553 | 2,466 | 3,169 | 3,325 |
| 226-250 | A cold | 30 | 2,748 | 2,646 | 3,172 | 4,790 |
| 226-250 | C cold | 30 | 1,456 | 1,461 | 1,522 | 1,543 |
| 226-250 | C warm | 30 | 815 | 810 | 843 | 933 |
| 226-250 | -ub 64 cold | 17 | 3,056 | 2,825 | 4,195 | 5,115 |

The longest bucket (226-250 words): A cold P95 3,172 ms, C cold P95 1,522 ms. **Not below 1,000 ms.** Only C's warm requests (P95 about 839 ms, flat across lengths) are below 1 s.

## F. Remaining bottlenecks

- Target P95 < 1,000 ms for a complete three-intent request: **not reached.** Best measured cold P95 is 1,480 ms (config C), about 480 ms over target; every bucket except 50-75 is above 1.1 s at P95.
- The first cold pass (Intent A) is about 54% of C's cold latency: about 346 tokens at roughly 510 tokens/s. No cache can remove it because the message has never been seen. It grows with prompt length (A cold P95 rises from about 1.76 s to 3.17 s in the same bucket range in config A).
- Intents B and C still process 134 and 117 tokens each (about 300 and 264 ms) because reuse stops at a checkpoint boundary and the question block is new. Together that is about 45% of cold latency.
- Warm floor in C is about 810 ms (three scans of roughly 270 ms each, about 120-130 tokens each), so even a perfect cache hit on the message leaves little headroom under 1 s with current per-request cost.
- Memory: free memory stayed at 48% or higher and no page-outs occurred in any run, but swap was 21-23 GB of 24.6 GB used. Swap-in counts were nonzero (A: 2,945; C: 120,856; -ub 64: 75,364). A's slowest cold scan (4,790 ms, P50) had 32 swap-ins, so a memory effect on A's tail is possible but not proven. C's slowest scans had 0-12 swap-ins and were not outliers. I did not establish causation either way.
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

`knowline_opt/`: `build_dataset.py`, `frozen_dataset.json` (+ `.sha256`), `run_config.py`, `drive_config.sh`, `analyze.py`, `make_report.py`, `results/{A,C,U64}.jsonl`, `results/B_pilot.jsonl`, `results/summary.json`, `results/scores_{C,U64}_vs_A.json`, `logs/`.
