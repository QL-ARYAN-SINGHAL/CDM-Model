"""Paired accuracy comparison of read once (current production layout) and read twice (Gen5.5 layout), same weights.

Three paired data sets, all from saved runs of the same model file (KnowLine-4B-Gen5.5 Q5_K_M):
  A. eval_set_500.json     500 single-intent prompts       bench-eval_set_500-q5-once-compact vs q5-twice-compact
  B. eval_big.json         40 big prompts x 5 intents      bigbench-whole-paragraph-q5-once-compact vs q5-twice-compact
  C. eval_suite100.json    first 40 of 100 prompts x 3     suite100-knowline-4b-q5 (read once) vs suite100_read_twice
"""
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
R = ROOT / "eval_results"
OUT = R / "latency_investigation"
HI = 0.7


def load(p):
    return json.loads(Path(p).read_text())


def decisions_A():
    a = load(R / "bench-eval_set_500-q5-once-compact.json")["rows"]
    b = load(R / "bench-eval_set_500-q5-twice-compact.json")["rows"]
    return [(x["intent"], x["expected"], x["score"], y["score"], x["text"][:70]) for x, y in zip(a, b)]


def decisions_B():
    cases = load(ROOT / "eval_big.json")
    a = load(R / "bigbench-whole-paragraph-q5-once-compact.json")["rows"]
    b = load(R / "bigbench-whole-paragraph-q5-twice-compact.json")["rows"]
    out = []
    for x, y in zip(a, b):
        for i in x["scores"]:
            out.append((i, i in x["present"], x["scores"][i], y["scores"][i], f"big prompt #{x['id']}"))
    return out


def decisions_C():
    cases = load(ROOT / "eval_suite100.json")
    a = load(R / "suite100-knowline-4b-q5.json")["rows"]
    b = load(OUT / "suite100_read_twice.json")["rows"]
    out = []
    for c, x, y in zip(cases, a, b):  # zip stops at the 40 read-twice rows
        for d, e, s1, s2 in zip(c["intents"], c["expected"], x["scores"], y["scores"]):
            out.append((d["name"], e, s1, s2, c["text"][:70]))
    return out


def metrics(dec, idx):
    tp = sum(d[idx] >= HI and d[1] for d in dec); fp = sum(d[idx] >= HI and not d[1] for d in dec)
    fn = sum(d[idx] < HI and d[1] for d in dec); tn = sum(d[idx] < HI and not d[1] for d in dec)
    return {"precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4), "fp_rate": round(fp / max(fp + tn, 1), 4),
            "fn_rate": round(fn / max(fn + tp, 1), 4), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "confidently_wrong(>=0.9 or <=0.1)": sum((d[idx] >= 0.9 and not d[1]) or (d[idx] <= 0.1 and d[1]) for d in dec)}


def analyse(name, dec):
    diffs = [abs(d[2] - d[3]) for d in dec]
    crossing = [d for d in dec if (d[2] >= HI) != (d[3] >= HI)]
    per_intent = defaultdict(list)
    for d in dec:
        per_intent[d[0]].append(d)
    res = {"decisions": len(dec), "read_once": metrics(dec, 2), "read_twice": metrics(dec, 3),
           "decision_disagreements": len(crossing), "disagreement_rate": round(len(crossing) / len(dec), 4),
           "mean_abs_prob_diff": round(sum(diffs) / len(diffs), 4), "max_abs_prob_diff": round(max(diffs), 4),
           "prob_diff_gt_0.1": sum(x > 0.1 for x in diffs),
           "threshold_crossings": [{"intent": d[0], "expected": d[1], "once": round(d[2], 3), "twice": round(d[3], 3), "text": d[4]} for d in crossing],
           "per_intent": {k: {"n": len(v), "once": {m: metrics(v, 2)[m] for m in ("precision", "recall")},
                              "twice": {m: metrics(v, 3)[m] for m in ("precision", "recall")}} for k, v in per_intent.items()}}
    print(f"\n{name}: {len(dec)} decisions, {len(crossing)} disagree at 0.7, mean |dp| {res['mean_abs_prob_diff']}, max |dp| {res['max_abs_prob_diff']}")
    for tag in ("read_once", "read_twice"):
        m = res[tag]
        print(f"  {tag:10} precision {m['precision']} recall {m['recall']} FP {m['fp']} FN {m['fn']} confidently-wrong {m['confidently_wrong(>=0.9 or <=0.1)']}")
    return res


if __name__ == "__main__":
    results = {"A_500_single": analyse("A. 500 single prompts", decisions_A()), "B_40_big_x5": analyse("B. 40 big prompts x 5 intents", decisions_B())}
    if (OUT / "suite100_read_twice.json").exists():
        results["C_suite100_first40_x3"] = analyse("C. suite100 first 40 prompts x 3 intents", decisions_C())
    (OUT / "read_twice_accuracy.json").write_text(json.dumps(results, indent=1))
