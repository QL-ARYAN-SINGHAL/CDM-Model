"""Step 3: score a trained model on the dual-task test split and the 300 gold rows (intent P/R, extraction exact-span P/R/F1, 'no intent -> no extraction'), with progress + ETA.
usage: python evaluate.py --data DIR --model DIR --out DIR [--max_test N] [--device cuda]"""
import argparse, json, os, sys, time, datetime
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
try:
    import truststore; truststore.inject_into_ssl()
except Exception: pass
import torch
from gliner2 import AutoExtractor

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--model", required=True); ap.add_argument("--out", required=True); ap.add_argument("--max_test", type=int, default=0); ap.add_argument("--max_gold", type=int, default=0)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu"); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--threshold", type=float, default=0.5)
ap.add_argument("--chunk", type=int, default=0, help="ask extract labels in groups of this size (0 = all 45 at once)")
ap.add_argument("--inference", choices=("joint", "separate"), default="joint", help="joint matches the training schema; separate reproduces legacy evaluation")
A = ap.parse_args(); os.makedirs(A.out, exist_ok=True)
if A.inference == "joint" and A.chunk:
    ap.error("--chunk requires --inference separate; joint uses the full combined training schema")
say = lambda m: print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)
kw = {"quantize": True} if A.device.startswith("cuda") else {}      # fp16 inference on GPU
model = AutoExtractor.from_pretrained(A.model, map_location=A.device, **kw); model.eval()
test = [json.loads(l) for l in open(os.path.join(A.data, "dual_test.jsonl"))]
if A.max_test: test = test[:A.max_test]
gold = [r for r in (json.loads(l) for l in open(os.path.join(A.data, "dual_test.jsonl"))) if r.get("gold")]
if A.max_gold: gold = gold[:A.max_gold]
res = {"inference_schema": A.inference}
def predict(texts, batch_size):
    if A.inference == "joint": return common.predict_joint(model, texts, batch_size=batch_size, threshold=A.threshold)
    return (common.predict_intent(model, texts, batch_size=batch_size),
            common.predict_entities(model, texts, batch_size=batch_size, threshold=A.threshold, chunk=A.chunk))
if len(gold) >= 20:        # single-request latency (one text per call, intent + all 45 extract labels), the number that matters for a guardrail
    lat = []
    for r in gold[:100]:
        t = time.time(); predict([r["text"]], batch_size=1); lat.append((time.time() - t) * 1000)
    lat.sort(); res["latency_ms_single_request"] = {"p50": round(lat[len(lat) // 2], 1), "p95": round(lat[int(len(lat) * .95)], 1), "n": len(lat), "device": A.device}; say(f"single-request latency p50 {res['latency_ms_single_request']['p50']} ms / p95 {res['latency_ms_single_request']['p95']} ms")
for name, rows in (("gold", gold), ("test", test)):
    t0 = time.time(); probs, ents = [], []; B = A.batch * 4
    for i in range(0, len(rows), B):
        chunk = rows[i:i + B]; texts = [r["text"] for r in chunk]
        chunk_probs, chunk_ents = predict(texts, batch_size=A.batch)
        probs += chunk_probs; ents += chunk_ents
        done = i + len(chunk); el = time.time() - t0; say(f"{name}: {done}/{len(rows)} ({100 * done / len(rows):.0f}%) | {done / el:.1f} rows/s | elapsed {common.fmt_eta(el)} | ETA {common.fmt_eta(el / done * (len(rows) - done))}")
    m = {"rows": len(rows), "seconds": round(time.time() - t0, 1), "rows_per_s": round(len(rows) / (time.time() - t0), 2), "intent@0.5": common.intent_metrics(rows, probs, A.threshold),
         "best_intent_thresholds": {f: common.best_threshold(rows, probs, f) for f in common.FAMILIES},
         "extract_raw": common.extract_metrics(rows, ents), "extract_gated_by_own_intent": common.extract_metrics(rows, ents, gate_probs=probs, thr=A.threshold)}
    res[name] = m
    with open(os.path.join(A.out, f"preds_{name}.jsonl"), "w") as f:
        for r, p, e in zip(rows, probs, ents): f.write(json.dumps({"id": r["id"], "intent_prob": p, "entities": [list(x) for x in e]}) + "\n")
    json.dump(res, open(os.path.join(A.out, "metrics.json"), "w"), indent=1); say(f"{name} intent F1 " + str({f: m['intent@0.5'][f]['f1'] for f in common.FAMILIES}) + f" | extract F1 raw {m['extract_raw']['micro']['f1']} gated {m['extract_gated_by_own_intent']['micro']['f1']}")
say("EVAL DONE")
