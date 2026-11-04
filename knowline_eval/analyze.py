"""Summarise results/*.jsonl against suite100.json labels -> results/summary.json + printed tables."""
import json, glob, statistics as st, collections
suite = {p["id"]: p for p in json.load(open("suite100.json"))["prompts"]}
lab = {(p["id"], i["name"]): i["label"] for p in suite.values() for i in p["intents"]}
rows = [json.loads(l) for f in sorted(glob.glob("results/*.jsonl")) for l in open(f)]

def pct(v, q):
    v = sorted(v); k = (len(v) - 1) * q; f = int(k); c = min(f + 1, len(v) - 1)
    return v[f] + (v[c] - v[f]) * (k - f)

def auc(pairs):  # pairs: (score, is_positive)
    pos = [s for s, y in pairs if y]; neg = [s for s, y in pairs if not y]
    if not pos or not neg: return float("nan")
    return sum((p > n) + 0.5 * (p == n) for p in pos for n in neg) / (len(pos) * len(neg))

def best_thr(pairs):
    return max(((sum((s >= t) == y for s, y in pairs) / len(pairs)), t) for t in [x / 100 for x in range(5, 96, 5)])

out = {}
for key in sorted({(r["model"], r["schema"]) for r in rows}):
    rs = [r for r in rows if (r["model"], r["schema"]) == key]
    lat = [r["latency_ms"] for r in rs]
    res = {"prompts": len(rs), "lat_mean_ms": round(st.mean(lat)), "lat_p50_ms": round(pct(lat, .5)), "lat_p95_ms": round(pct(lat, .95))}
    if key[1] == "E":
        ok = [max(r["dist"], key=r["dist"].get) == r["expected"] for r in rs]
        res.update(choice_acc=round(sum(ok) / len(ok), 3), n_choice_prompts=len(ok),
                   en_acc=round(sum(o for o, r in zip(ok, rs) if r["lang"] == "en") / max(1, sum(r["lang"] == "en" for r in rs)), 3))
    else:
        pairs, by = [], collections.defaultdict(list)
        for r in rs:
            for n, s in r["scores"].items():
                l = lab[(r["id"], n)]; pairs.append((s, l == "P")); by[l].append(s); by[("lang", r["lang"])].append((s >= .5) == (l == "P"))
        tp = sum(s >= .5 and y for s, y in pairs); fp = sum(s >= .5 and not y for s, y in pairs); fn = sum(s < .5 and y for s, y in pairs)
        prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
        ba, bt = best_thr(pairs)
        res.update(n_scores=len(pairs), acc=round(sum((s >= .5) == y for s, y in pairs) / len(pairs), 3), precision=round(prec, 3), recall=round(rec, 3),
                   f1=round(2 * prec * rec / max(1e-9, prec + rec), 3), auc=round(auc(pairs), 3), best_thr_acc=round(ba, 3), best_thr=bt,
                   near_miss_fp_rate=round(sum(s >= .5 for s in by["N"]) / len(by["N"]), 3), negative_fp_rate=round(sum(s >= .5 for s in by["X"]) / len(by["X"]), 3),
                   en_acc=round(st.mean(by[("lang", "en")]), 3), nonen_acc=round(st.mean(by[("lang", "non-en")]), 3),
                   lat_per_intent_ms=round(sum(lat) / sum(len(r["scores"]) for r in rs)))
        # prompt-level: every intent of the prompt right at 0.5
        res["prompt_all_correct"] = round(sum(all((s >= .5) == (lab[(r["id"], n)] == "P") for n, s in r["scores"].items()) for r in rs) / len(rs), 3)
    out["%s/%s" % key] = res
json.dump(out, open("results/summary.json", "w"), indent=1)
cols = ["acc", "f1", "auc", "near_miss_fp_rate", "negative_fp_rate", "prompt_all_correct", "choice_acc", "lat_mean_ms", "lat_p50_ms", "lat_p95_ms"]
print("%-8s" % "run" + "".join("%20s" % c for c in cols))
for k, v in out.items(): print("%-8s" % k + "".join("%20s" % v.get(c, "-") for c in cols))
