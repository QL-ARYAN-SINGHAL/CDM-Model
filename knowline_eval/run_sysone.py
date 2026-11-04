"""Score suite100.json against any server that speaks /v1/systemone (llama-server with a decision GGUF, optiq serve, ...).
One request per prompt carrying ALL its intents as questions (how these models are meant to be served); latency = that request.
usage: run_sysone.py --url http://127.0.0.1:8011 --tag aurelis2b --model NAME --schemas A,B,C,D,E --out results/aurelis2b.jsonl"""
import argparse, json, time, urllib.request
from pathlib import Path
from run_bench import q_noul

def post(url, body, timeout=300):
    req = urllib.request.Request(url + "/v1/systemone", data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r: return json.load(r)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True); ap.add_argument("--tag", required=True); ap.add_argument("--model", default="m")
    ap.add_argument("--schemas", default="A,B,C,D,E"); ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--suite", default=str(Path(__file__).parent / "suite100.json")); ap.add_argument("--out", required=True)
    a = ap.parse_args()
    suite = json.load(open(a.suite))["prompts"]
    if a.limit: suite = suite[:a.limit]
    post(a.url, {"model": a.model, "state": suite[0]["text"], "questions": {"w": q_noul("A", suite[0]["intents"][0])}})  # warm-up
    out = open(a.out, "a")
    for schema in a.schemas.split(","):
        for p in suite:
            rec = {"model": a.tag, "schema": schema, "id": p["id"], "lang": p["lang"], "words": p["words"]}
            if schema == "E":
                pos = [i["name"] for i in p["intents"] if i["label"] == "P"]
                if len(pos) > 1: continue
                crit = {i["name"]: i["question"] for i in p["intents"]}; crit["none"] = "None of the above intents is present."
                qs = {"q": {"type": "choice", "instructions": "Which ONE intent does the text express most clearly?", "criteria": crit}}
            else:
                qs = {i["name"]: q_noul(schema, i) for i in p["intents"]}
            t0 = time.perf_counter(); resp = post(a.url, {"model": a.model, "state": p["text"], "questions": qs}); dt = (time.perf_counter() - t0) * 1000
            ans = resp["answers"]
            if schema == "E":
                rec.update(expected=(pos[0] if pos else "none"), dist=ans["q"]["probabilities"])
            else:
                rec["scores"] = {n: float(ans[n]["noul"]) for n in qs}
            rec["latency_ms"] = round(dt, 1)
            out.write(json.dumps(rec) + "\n"); out.flush()
        print(a.tag, schema, "done", flush=True)

if __name__ == "__main__":
    main()
