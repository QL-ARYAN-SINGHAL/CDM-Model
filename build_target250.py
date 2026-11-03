"""eval_target250.json: 40 prompts of about 250 words (230-270), each checked against 3 intents; the production target size."""
import json
import random
from pathlib import Path

import build_eval_set as hand
import build_long_set as L
import build_suite100 as S

ROOT = Path(__file__).resolve().parent
rng = random.Random(20261107)
S.rng = rng
L.rng = rng
defs = hand.defs()
defs.update(L.SHORT)
pools = L.pools()
names = list(pools)
cases = []
for _ in range(40):
    asked = rng.sample(names, 3)
    present = set(rng.sample(asked, rng.choices([0, 1, 2, 3], weights=[10, 35, 35, 20])[0]))
    core = []
    for n in asked:
        if n in present:
            core += rng.sample(pools[n][0], rng.choice([1, 2]))
        elif rng.random() < 0.6:
            core += rng.sample(pools[n][1], 1)
    text = S.text_of_size(max(rng.randint(205, 235) - sum(len(c.split()) for c in core), 20), core)
    cases.append({"text": text, "intents": [defs[n] for n in asked], "expected": [n in present for n in asked]})
(ROOT / "eval_target250.json").write_text(json.dumps(cases, indent=1, ensure_ascii=False))
w = sorted(len(c["text"].split()) for c in cases)
print(f"{len(cases)} prompts, words min {w[0]}, median {w[20]}, max {w[-1]}; decisions {sum(sum(c['expected']) for c in cases)}/{3*len(cases)} positive")
