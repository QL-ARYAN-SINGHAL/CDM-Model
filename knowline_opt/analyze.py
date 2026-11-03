"""Summarize results/{A,B,C,D}.jsonl against the unchanged baseline A and the provisional labels."""
import json
import statistics as S
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
labels = sys.argv[1:] or ["A", "B", "C", "D"]
data = json.loads((HERE / "frozen_dataset.json").read_text())
exp = {p["id"]: p["expected"] for p in data["prompts"]}
BUCKETS = ["50-75", "76-125", "126-175", "176-225", "226-250"]
TH_BIN, TH_LOW, TH_HIGH = 0.5, 0.4, 0.7


def pct(v, q):
    v = sorted(v)
    k = (len(v) - 1) * q
    f = int(k)
    return v[f] + (v[min(f + 1, len(v) - 1)] - v[f]) * (k - f)


def load(l):
    return [json.loads(x) for x in (HERE / "results" / f"{l}.jsonl").open()]


def status(p):
    return "match" if p >= TH_HIGH else "review" if p >= TH_LOW else "no_match"


R = {l: load(l) for l in labels}
out = {}


def lat(rows):
    v = [r["e2e_ms"] for r in rows]
    return {"n": len(v), "mean": round(S.mean(v), 1), "p50": round(pct(v, .5), 1), "p95": round(pct(v, .95), 1), "max": round(max(v), 1)} if v else None


def mem(rows):
    d = lambda k: sum(r["mem_after"].get(k, 0) - r["mem_before"].get(k, 0) for r in rows)
    return {"pageins": d("pageins"), "pageouts": d("pageouts"), "swapins": d("swapins"), "swapouts": d("swapouts"),
            "min_free_pct": min(r["mem_after"].get("free_pct", 100) for r in rows),
            "swap_used_mb_range": [min(r["mem_before"].get("swap_used_mb", 0) for r in rows), max(r["mem_after"].get("swap_used_mb", 0) for r in rows)],
            "server_rss_mb_max": max(r["mem_after"].get("server_rss_mb", 0) for r in rows)}


for l, rows in R.items():
    cold = [r for r in rows if r["phase"] == "cold"]
    warm = [r for r in rows if r["phase"] == "warm"]
    o = {"cold": lat(cold), "warm": lat(warm), "cold_invalid": sum(1 for r in cold if not r["cold_valid"]),
         "cold_by_bucket": {b: lat([r for r in cold if r["bucket"] == b]) for b in BUCKETS},
         "warm_by_bucket": {b: lat([r for r in warm if r["bucket"] == b]) for b in BUCKETS} if warm else None}
    for name, grp in (("cold", cold), ("warm", warm)):
        if grp:
            o[f"{name}_tokens"] = {i: {"processed": round(S.mean(r["intents"][k]["prompt_n"] for r in grp), 1),
                                       "reused": round(S.mean(r["intents"][k]["cache_n"] for r in grp), 1)} for k, i in enumerate("ABC")}
            o[f"{name}_total_processed"] = round(S.mean(sum(x["prompt_n"] for x in r["intents"]) for r in grp), 1)
            o[f"{name}_total_reused"] = round(S.mean(sum(x["cache_n"] for x in r["intents"]) for r in grp), 1)
    o["memory"] = mem(rows)
    o["slowest_cold"] = sorted(((round(r["e2e_ms"]), r["id"], r["rep"], r["mem_after"].get("swapins", 0) - r["mem_before"].get("swapins", 0)) for r in cold), reverse=True)[:5]
    # accuracy vs provisional labels (rep 1 cold, threshold 0.5)
    r1 = [r for r in cold if r["rep"] == 1]
    acc = {}
    for k, i in enumerate("ABC"):
        tp = fp = fn = tn = 0
        for r in r1:
            pred, gold = r["intents"][k]["p_true"] >= TH_BIN, bool(exp[r["id"]][i])
            tp += pred and gold; fp += pred and not gold; fn += (not pred) and gold; tn += (not pred) and not gold
        acc[i] = {"tp": tp, "fp": fp, "fn": fn, "tn": tn, "precision": round(tp / (tp + fp), 3) if tp + fp else None, "recall": round(tp / (tp + fn), 3) if tp + fn else None}
    o["accuracy_vs_provisional"] = {"per_intent": acc, "correct": sum(a["tp"] + a["tn"] for a in acc.values()), "of": 150}
    # repeatability across reps (same config)
    byk = {}
    for r in cold:
        for k, i in enumerate("ABC"):
            byk.setdefault((r["id"], i), []).append(r["intents"][k]["p_true"])
    o["rep_to_rep_max_spread"] = max(max(v) - min(v) for v in byk.values())
    out[l] = o

# agreement with baseline A
if "A" in R:
    base = {}
    for r in R["A"]:
        if r["phase"] == "cold":
            for k, i in enumerate("ABC"):
                base.setdefault((r["id"], i), []).append(r["intents"][k]["p_true"])
    base1 = {key: v[0] for key, v in base.items()}
    for l, rows in R.items():
        diffs, flips, status_flips, crossings, table = [], 0, 0, [], []
        for r in rows:
            if r["phase"] != "cold" or r["rep"] != 1:
                continue
            for k, i in enumerate("ABC"):
                p, b = r["intents"][k]["p_true"], base1[(r["id"], i)]
                d = abs(p - b)
                diffs.append(d)
                if (p >= TH_BIN) != (b >= TH_BIN):
                    flips += 1
                if status(p) != status(b):
                    status_flips += 1
                    crossings.append((r["id"], i, round(b, 4), round(p, 4)))
                table.append({"id": r["id"], "intent": i, "base": round(b, 6), "cand": round(p, 6), "abs_diff": round(d, 6)})
        out[l]["vs_baseline_A"] = {"decisions": len(diffs), "binary_flips_at_0.5": flips, "agree_at_0.5": len(diffs) - flips,
                                   "status_flips_0.4_0.7": status_flips, "status_flip_cases": crossings,
                                   "max_abs_diff": round(max(diffs), 6), "mean_abs_diff": round(S.mean(diffs), 6)}
        (HERE / "results" / f"scores_{l}_vs_A.json").write_text(json.dumps(table, indent=1))

(HERE / "results" / "summary.json").write_text(json.dumps(out, indent=1))
for l, o in out.items():
    print("==", l, "cold", o["cold"], "warm", o["warm"], "invalid_cold", o["cold_invalid"])
    print("   tokens cold", o.get("cold_tokens"), "warm", o.get("warm_tokens"))
    print("   cold p95 by bucket", {b: (v or {}).get("p95") for b, v in o["cold_by_bucket"].items()})
    if o["warm_by_bucket"]:
        print("   warm p95 by bucket", {b: (v or {}).get("p95") for b, v in o["warm_by_bucket"].items()})
    print("   vs A", o.get("vs_baseline_A"))
    print("   acc", o["accuracy_vs_provisional"]["correct"], "/150", {i: (a["fp"], a["fn"]) for i, a in o["accuracy_vs_provisional"]["per_intent"].items()}, "rep spread", round(o["rep_to_rep_max_spread"], 6))
    print("   mem", o["memory"], "slowest", o["slowest_cold"][:3])
