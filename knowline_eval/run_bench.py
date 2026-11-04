"""Score suite100.json against one KnowLine model (llama-server) under several question schemas.
usage: run_bench.py --model 4b|2b --schemas A,B,C,D,E [--read-twice] [--out results/4b.jsonl]
Schemas (all 'noul' yes/no questions except E):
  A  instructions=question, criteria true=Do false=Don't   (current repo schema)
  B  instructions=question only (no Do/Don't)
  C  instructions=question, criteria true=Do only
  D  question + 'Yes only if: Do No if: Don't' folded into instructions, no criteria
  E  ONE 'choice' question per prompt listing every intent plus 'none' (single label, 1 forward pass)
"""
import argparse, json, math, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
os.environ.setdefault("KNOWLINE_CACHE", "1")
import knowline_bridge as kb

def q_noul(schema, it):
    q, do, dont = it["question"], it["do"], it["dont"]
    if schema == "A": return {"type": "noul", "instructions": q, "criteria": {"true": do, "false": dont}}
    if schema == "B": return {"type": "noul", "instructions": q}
    if schema == "C": return {"type": "noul", "instructions": q, "criteria": {"true": do}}
    if schema == "D": return {"type": "noul", "instructions": f"{q} Answer Yes only if: {do} Answer No if: {dont}"}
    raise ValueError(schema)

def choice_dist(sc, state, intents, read_twice):
    crit = {it["name"]: it["question"] for it in intents}; crit["none"] = "None of the above intents is present."
    question = {"type": "choice", "instructions": "Which ONE intent does the text express most clearly?", "criteria": crit}
    r = sc._render_request({"state": state, "questions": {"q": question}}, sc.tok, sc.layouts[read_twice], labels=sc.labels)
    text, keys = r["prompts"]["q"][0]
    ids = sc.tok.encode(text, add_special_tokens=False)
    data = kb._request("/completion", {"prompt": ids, "n_predict": 1, "temperature": 0, "n_probs": 200, "cache_prompt": sc.cache}, timeout=180, base=sc.base)
    top = {e["id"]: e["logprob"] for e in data["completion_probabilities"][0]["top_logprobs"]}
    lp = [top.get(i, -math.inf) for i in sc.label_ids[:len(keys)]]
    m = max(lp); w = [math.exp(x - m) for x in lp]; z = sum(w)
    return dict(zip(keys, (x / z for x in w)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["2b"])
    ap.add_argument("--schemas", default="A,B,C,D,E")
    ap.add_argument("--read-twice", action="store_true")
    ap.add_argument("--no-cache", action="store_true", help="cache_prompt=false (the original bridge setting)")
    ap.add_argument("--tag", default="", help="label stored in results (default: model)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suite", default=str(Path(__file__).parent / "suite100.json"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    suite = json.load(open(a.suite))["prompts"]
    if a.limit: suite = suite[:a.limit]
    sc = kb._Scorer(base=kb.MODELS[a.model]["url"], read_twice=a.read_twice, cache=not a.no_cache)
    tag = a.tag or (a.model + ("+rt" if a.read_twice else ""))
    sc.p_true(suite[0]["text"], q_noul("A", suite[0]["intents"][0]))   # warm-up, not recorded
    out = open(a.out, "a")
    for schema in a.schemas.split(","):
        for p in suite:
            t0 = time.perf_counter()
            rec = {"model": tag, "schema": schema, "id": p["id"], "lang": p["lang"], "words": p["words"]}
            if schema == "E":
                pos = [i["name"] for i in p["intents"] if i["label"] == "P"]
                if len(pos) > 1: continue
                dist = choice_dist(sc, p["text"], p["intents"], a.read_twice)
                rec.update(expected=(pos[0] if pos else "none"), dist=dist)
            else:
                rec["scores"] = {i["name"]: sc.p_true(p["text"], q_noul(schema, i)) for i in p["intents"]}
            rec["latency_ms"] = round((time.perf_counter() - t0) * 1000, 1)
            out.write(json.dumps(rec) + "\n"); out.flush()
        print(tag, schema, "done", flush=True)

if __name__ == "__main__":
    main()
