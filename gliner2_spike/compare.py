"""Report for one GLiNER2.5-Decide run: intent P/R/F1, exact-row accuracy, extraction P/R/F1 (raw and gated by own intent), per-type table. usage: python compare.py --gliner DIR > REPORT.md"""
import argparse, json, os
ap = argparse.ArgumentParser(); ap.add_argument("--gliner", required=True); A = ap.parse_args()
g = json.load(open(os.path.join(A.gliner, "metrics.json")))
print("# GLiNER2.5-Decide: intent + extract (fine-tuned)\n")
for sp in ("gold", "test"):
    if sp not in g: continue
    a = g[sp]["intent@0.5"]
    print(f"## {sp} ({g[sp]['rows']} rows)\n\n| detector | P | R | F1 | best threshold (F1) |\n|---|---|---|---|---|")
    for f in ("pii", "phi", "pci"):
        t, _ = g[sp]["best_intent_thresholds"][f]
        print(f"| {f.upper()} | {a[f]['precision']} | {a[f]['recall']} | {a[f]['f1']} | {t} |")
    print(f"\nexact-row accuracy {a['exact_row_accuracy']}; false-positive rate on None rows {a['none_fp_rate']}\n")
    for k, label in (("extract_raw", "extraction, all predicted types"), ("extract_gated_by_own_intent", "extraction, gated by own intent")):
        m = g[sp][k]; print(f"- {label}: P {m['micro']['precision']} / R {m['micro']['recall']} / F1 {m['micro']['f1']}; None-intent rows that still extract: {m['intent0_rows_with_any_extraction']}")
    print("\n| type | P | R | F1 | support |\n|---|---|---|---|---|")
    for t, m in sorted(g[sp]["extract_gated_by_own_intent"]["per_type"].items(), key=lambda kv: kv[1]["f1"]):
        print(f"| {t} | {m['precision']} | {m['recall']} | {m['f1']} | {m['tp'] + m['fn']} |")
    print()
if "latency_ms_single_request" in g: print("single-request latency:", g["latency_ms_single_request"])
