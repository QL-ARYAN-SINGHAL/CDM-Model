"""Build eval_suite100.json: 100 prompts of 50-250 words, each checked against 3 intents (300 yes/no decisions).

Six intents (travel, Banking, Refund Policy, SONAR Defense, P & L, Secrets & credentials) with short definitions. Each
prompt is ordinary workplace text; it contains the content of 0-3 of its three asked intents (inserted sentences), and the
absent ones are sometimes given a near-miss sentence on the same topic. The label of every (prompt, intent) comes from
what was inserted.  {"text", "intents": [3 definitions], "expected": [3 bools]}
"""
import json
import random
from pathlib import Path

import build_eval_set as hand
import build_long_set as L

ROOT = Path(__file__).resolve().parent
rng = random.Random(20261105)
L.rng = rng  # the long-set helpers draw from this generator too (deterministic suite)


def text_of_size(words, core):
    sentences = [rng.choice(L.OPENERS)]
    total, queue = 0, []
    while total < words:
        if not queue:
            queue = rng.sample(L.NEUTRAL, len(L.NEUTRAL))
        s = queue.pop()
        sentences.append(s)
        total += len(s.split())
    for c in core:
        sentences.insert(rng.randint(1, len(sentences)), c)
    sentences.append(rng.choice(L.CLOSERS))
    out, para = [], []
    for s in sentences:
        para.append(s)
        if len(para) >= rng.randint(3, 5):
            out.append(" ".join(para))
            para = []
    if para:
        out.append(" ".join(para))
    return "\n\n".join(out)


def main():
    defs = hand.defs()
    defs.update(L.SHORT)
    pools = L.pools()
    names = list(pools)
    cases = []
    for _ in range(100):
        asked = rng.sample(names, 3)
        k = rng.choices([0, 1, 2, 3], weights=[10, 35, 35, 20])[0]
        present = set(rng.sample(asked, k))
        core = []
        for n in asked:
            if n in present:
                core += rng.sample(pools[n][0], rng.choice([1, 2]))
            elif rng.random() < 0.6:
                core += rng.sample(pools[n][1], 1)
        words = rng.randint(35, 205)
        text = text_of_size(max(words - sum(len(c.split()) for c in core), 20), core)
        cases.append({"text": text, "intents": [defs[n] for n in asked], "expected": [n in present for n in asked]})
    (ROOT / "eval_suite100.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))
    w = sorted(len(c["text"].split()) for c in cases)
    pos = sum(sum(c["expected"]) for c in cases)
    print(f"100 prompts, 300 decisions ({pos} positive); words min {w[0]}, median {w[50]}, max {w[-1]}; "
          f"prompts by present intents: " + str({k: sum(sum(c['expected']) == k for c in cases) for k in range(4)}))


if __name__ == "__main__":
    main()
