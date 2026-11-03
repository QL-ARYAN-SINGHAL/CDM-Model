"""10-prompt check of the -ub 128 strategy on the live Verity KnowLine server (8009), same scoring code as the bridge.
Per prompt, back to back, no slot reset (production-like): TREATMENT = cache_prompt true (what the bridge sends with
KNOWLINE_CACHE=1), then CONTROL = cache_prompt false (what the running Verity bridge sends today). Treatment runs first so the
control's leftover state cannot help it. Nothing is written to the tracker CSV.
"""
import json
import statistics as S
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import profile_knowline as P  # noqa: E402

data = json.loads((HERE / "frozen_dataset.json").read_text())
intents, prompts = data["intents"], {p["id"]: p for p in data["prompts"]}
ids = ["P01", "P02", "P11", "P12", "P21", "P22", "P31", "P32", "P41", "P42"]
probe = P.Probe("http://127.0.0.1:8009")


def scan(text, cache):
    t0 = time.perf_counter()
    rows = [probe.p_true(text, P.question_of(d), False, cache) for d in intents]
    return (time.perf_counter() - t0) * 1000, [(p, T["prompt_n"], T["cache_n"]) for p, T in rows]


scan("Warm-up text, please ignore: the meeting moved to Thursday afternoon.", False)
out = []
for pid in ids:
    t_ms, t_rows = scan(prompts[pid]["text"], True)
    c_ms, c_rows = scan(prompts[pid]["text"], False)
    out.append({"id": pid, "words": prompts[pid]["word_count"], "treat_ms": t_ms, "ctrl_ms": c_ms,
                "treat_tokens": [(r[1], r[2]) for r in t_rows], "ctrl_tokens": [(r[1], r[2]) for r in c_rows],
                "max_score_diff": max(abs(a[0] - b[0]) for a, b in zip(t_rows, c_rows)),
                "decisions_same": all((a[0] >= 0.5) == (b[0] >= 0.5) for a, b in zip(t_rows, c_rows))})
    o = out[-1]
    print(pid, o["words"], "w | cache ON", round(t_ms), "ms", o["treat_tokens"], "| cache OFF", round(c_ms), "ms | diff", round(o["max_score_diff"], 5), "| same decisions", o["decisions_same"], flush=True)
t, c = [o["treat_ms"] for o in out], [o["ctrl_ms"] for o in out]
print(f"cache ON  (-ub128 strategy): mean {S.mean(t):.0f} median {S.median(t):.0f} max {max(t):.0f} ms")
print(f"cache OFF (today's bridge) : mean {S.mean(c):.0f} median {S.median(c):.0f} max {max(c):.0f} ms")
print(f"all 30 decisions identical: {all(o['decisions_same'] for o in out)}; max score diff {max(o['max_score_diff'] for o in out):.5f}")
(HERE / "results" / "verity_10prompt_check.json").write_text(json.dumps(out, indent=1))
