"""Score kev_tracking.csv prompts with decision models that need their own prompt layout (KnowLine, Jeffan).

Request body is the one kev_bridge.py builds (state = {full_message, lines}, one noul question per line with the
bridge's instructions and the row's Do/Don't). Each model renders it with its own published layout:

  knowline     KnowLine-4B-Gen5.5 layout (read twice, label-token readout) -> Q5_K_M GGUF on llama-server
  jeffan-gguf  Jeffan-4B-2010 answer-code layout                           -> Q5_K_M GGUF on llama-server (causal mask)
  jeffan-hf    Jeffan-4B-2010 answer-code layout, bidirectional full-attention mask, bf16 weights in transformers

Usage: eval_alt_models.py MODE LABEL [PORT]      writes eval_results/LABEL.json
"""
import csv
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
CAND = ROOT / "models" / "candidates"
RESULTS = ROOT / "eval_results"


def unique_rows():
    import os
    if os.environ.get("EVAL_SET"):  # a labelled set built by build_eval_set.py: {"q": definition, "text", "expected"}
        cases = json.loads(Path(os.environ["EVAL_SET"]).read_text())
        return [{"raw_prompt": c["text"], "questions": json.dumps([c["q"]])} for c in cases]
    seen = {}
    with (ROOT / "archive" / "kev_0.8b_tracking_2026-11-03.csv").open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            seen[(row["raw_prompt"], row["questions"])] = row
    return list(seen.values())


def bridge_request(prompt, questions):
    lines = [line.strip() for line in prompt.splitlines() if line.strip()]
    qs = {}
    for i in range(len(lines)):
        for j, q in enumerate(questions):
            qs[f"l{i}__q{j}"] = {"type": "noul",
                                 "instructions": f"Judge only lines[{i}]. Use full_message only as context. {q['description']}",
                                 "criteria": {"true": q["do"], "false": q["dont"]}}
    return {"state": {"full_message": prompt, "lines": lines}, "questions": qs}


def llama_logprobs(port, token_ids, wanted_ids):
    """Next-token log-probabilities of wanted_ids after the prompt (token ids), via llama-server /completion."""
    body = {"prompt": token_ids, "n_predict": 1, "temperature": 0, "n_probs": 200, "cache_prompt": False}
    req = urllib.request.Request(f"http://127.0.0.1:{port}/completion", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as response:
        data = json.load(response)
    top = data["completion_probabilities"][0]
    entries = top.get("top_logprobs") or top.get("top_probs") or []
    found = {}
    for e in entries:
        value = e.get("logprob", math.log(e["prob"]) if e.get("prob") else -math.inf)
        found[e["id"]] = value
    return [found.get(i, -math.inf) for i in wanted_ids]


def softmax(values):
    peak = max(values)
    weights = [math.exp(v - peak) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


# ----------------------------------------------------------------------------------------------- KnowLine
def make_knowline(port):
    sys.path.insert(0, str(ROOT / "knowline_src"))
    from transformers import AutoTokenizer
    import knowline_server as ks
    from knowline_layout import Layout, render_request
    tok = AutoTokenizer.from_pretrained(str(ROOT / "knowline_src"))
    _, probe, _ = ks.render(tok, "x", {"q": {"type": "noul"}}, ["A", "B"])
    labels, ids = ks.find_labels(tok, probe["q"][0])
    layout = Layout(read_twice=True)

    def score(request):
        r = render_request(request, tok, layout, labels=labels)
        best = 0.0
        for qid in request["questions"]:
            text, keys = r["prompts"][qid][0]
            token_ids = tok.encode(text, add_special_tokens=False)
            logps = llama_logprobs(port, token_ids, ids[:len(keys)])
            probs = dict(zip(keys, softmax(logps)))
            best = max(best, probs["true"])
        return best
    return score


# ----------------------------------------------------------------------------------------------- Jeffan
def make_jeffan(mode, port):
    import importlib.util
    from transformers import AutoTokenizer
    # load render.py alone: the package __init__ imports the Decision Index engine, which needs CUDA and an extra package
    spec = importlib.util.spec_from_file_location("jeffan_render", CAND / "jeffan-bf16" / "inference" / "jeffan_engine" / "render.py")
    render_module = importlib.util.module_from_spec(spec)
    sys.modules["jeffan_render"] = render_module
    spec.loader.exec_module(render_module)
    CodeRenderer, from_wire = render_module.CodeRenderer, render_module.from_wire
    tok = AutoTokenizer.from_pretrained(str(CAND / "jeffan-bf16"))
    renderer = CodeRenderer(tok, max_len=65536, reads=1)
    if mode == "jeffan-hf":
        import torch
        from transformers import AutoModelForCausalLM
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        lm = AutoModelForCausalLM.from_pretrained(str(CAND / "jeffan-bf16"), dtype=torch.bfloat16,
                                                  attn_implementation="sdpa").to(device).eval()
        code_ids = torch.tensor(renderer.code_ids[:2], device=device)

        def forward(ids):
            x = torch.tensor([ids], device=device)
            n = len(ids)
            mask = {"full_attention": torch.ones(1, 1, n, n, dtype=torch.bool, device=device), "linear_attention": None}
            with torch.inference_mode():
                h = lm.model(input_ids=x, attention_mask=mask, use_cache=False).last_hidden_state[0, -1]
                logits = lm.get_output_embeddings().weight[code_ids].float() @ h.float()
            return torch.log_softmax(logits, -1).cpu().tolist()
    else:
        def forward(ids):
            return llama_logprobs(port, ids, renderer.code_ids[:2])

    def score(request):
        best = 0.0
        for qid, q in request["questions"].items():
            row = from_wire(request["state"], q)
            ids = renderer.render(row) if hasattr(renderer, "render") else renderer.assemble(renderer.pieces(row))
            probs = softmax(forward(ids))  # options in request order: false, true
            best = max(best, probs[1])
        return best
    return score


def main(mode, label, port):
    score = make_knowline(port) if mode == "knowline" else make_jeffan(mode, port)
    out = []
    rows = unique_rows()
    first = json.loads(rows[0]["questions"])
    score(bridge_request(rows[0]["raw_prompt"], first))  # warm-up, not timed
    for row in rows:
        questions = json.loads(row["questions"])
        started = time.perf_counter()
        p = score(bridge_request(row["raw_prompt"], questions))
        ms = (time.perf_counter() - started) * 1000
        out.append({"prompt": row["raw_prompt"], "intent": questions[0]["name"], "score": round(p, 4), "ms": round(ms, 1)})
        print(f"{label:12} {questions[0]['name'][:10]:10} {p:.3f} {ms:7.0f}ms | {row['raw_prompt'][:70]}", flush=True)
    RESULTS.mkdir(exist_ok=True)
    (RESULTS / f"{label}.json").write_text(json.dumps({"model": label, "rows": out}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 8031)
