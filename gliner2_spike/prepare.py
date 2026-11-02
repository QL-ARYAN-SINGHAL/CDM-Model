"""Step 1: dual-task rows -> GLiNER2 train / val records (entities + classifications), with a preflight that runs GLiNER2's own collator on every record and drops the ones it cannot align.
usage: python prepare.py --data DIR --work DIR --model fastino/gliner2-base-v1 --train dual_train_100.jsonl [--max_train N] [--max_val N]"""
import argparse, json, os, random, sys, time
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common

ap = argparse.ArgumentParser()
ap.add_argument("--data", required=True); ap.add_argument("--work", required=True); ap.add_argument("--model", default="fastino/gliner2-base-v1")
ap.add_argument("--train", default="dual_train_100.jsonl"); ap.add_argument("--val", default="dual_val.jsonl")
ap.add_argument("--max_train", type=int, default=0); ap.add_argument("--max_val", type=int, default=3000); ap.add_argument("--max_labels", type=int, default=16); ap.add_argument("--full_prob", type=float, default=0.25)
ap.add_argument("--no_preflight", action="store_true")
A = ap.parse_args(); os.makedirs(A.work, exist_ok=True)
try:
    import truststore; truststore.inject_into_ssl()
except Exception: pass
from gliner2 import AutoExtractor

def load(name, cap, seed):
    rows = [json.loads(l) for l in open(os.path.join(A.data, name))]
    if cap and len(rows) > cap: random.Random(seed).shuffle(rows); rows = rows[:cap]
    return rows

t0 = time.time(); print("loading", A.model, flush=True)
model = AutoExtractor.from_pretrained(A.model, map_location="cpu")
arch = getattr(model, "architecture", "span"); max_len = getattr(getattr(model, "config", None), "max_len", None)
print(f"architecture={arch} max_len={max_len} params={sum(p.numel() for p in model.parameters()) / 1e6:.0f}M", flush=True)
rep = {"model": A.model, "architecture": arch, "max_len": max_len}
for tag, name, cap in (("val", A.val, A.max_val), ("train", A.train, A.max_train)):
    rows = load(name, cap, 1); recs = [common.to_record(r, max_labels=A.max_labels, full_prob=A.full_prob) for r in rows]; kept = recs; bad = Counter()
    if not A.no_preflight:
        proc = model.processor; proc.change_mode(is_training=True); kept = []; t1 = time.time()
        for i, rec in enumerate(recs):
            try: proc.collate_fn_train([(rec["input"], rec["output"])], max_len=max_len, architecture=arch, max_gold_per_query=32, on_capacity_exceeded="raise", dedupe_contained_entity_spans=True); kept.append(rec)
            except Exception as e: bad[(type(e).__name__ + ": " + str(e)[:90]).replace("\n", " ")] += 1
            if (i + 1) % 5000 == 0: print(f"  preflight {tag} {i + 1}/{len(recs)} kept {len(kept)} | {time.time() - t1:.0f}s", flush=True)
    with open(os.path.join(A.work, f"{tag}.jsonl"), "w") as f: [f.write(json.dumps(r, ensure_ascii=False) + "\n") for r in kept]
    rep[tag] = {"source": name, "rows": len(rows), "kept": len(kept), "dropped": len(recs) - len(kept), "drop_reasons": dict(bad.most_common(8)),
                "labels_per_example_p50": sorted(len(r["output"]["entities"]) for r in kept)[len(kept) // 2] if kept else 0}
    print(tag, rep[tag], flush=True)
json.dump(rep, open(os.path.join(A.work, "prepare_report.json"), "w"), indent=1); print(f"prepare done in {time.time() - t0:.0f}s", flush=True)
