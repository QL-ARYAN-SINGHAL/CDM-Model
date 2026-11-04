"""Merge suite_src/part*.py into suite100.json and validate the user's constraints."""
import json, sys, collections
sys.path.insert(0, "suite_src")
import part1, part2, part3, part4, patch
prompts = part1.PROMPTS + part2.PROMPTS + part3.PROMPTS + part4.PROMPTS
import re
def count_words(t):
    # whitespace words; CJK/kana/hangul chars ~1.6 chars per word; Thai ~5 chars per word
    cjk = len(re.findall(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af]', t)); thai = len(re.findall(r'[\u0e00-\u0e7f]', t))
    rest = re.sub(r'[\u3040-\u30ff\u3400-\u9fff\uac00-\ud7af\u0e00-\u0e7f]', ' ', t)
    return len(rest.split()) + round(cjk / 1.6) + round(thai / 5)
for p in prompts:
    if p["id"] in patch.LONG: p["text"] = p["text"] + " " + " ".join(patch.LONG[p["id"]].split())
    if p["id"] in patch.REPLACE:
        old, (n, l, q, d, o) = patch.REPLACE[p["id"]]
        k = next(i for i, it in enumerate(p["intents"]) if it["name"] == old)
        p["intents"][k] = {"name": n, "label": l, "question": q, "do": d, "dont": o}
    if p["id"] in patch.X:
        n, q, d, o = patch.X[p["id"]]; p["intents"].append({"name": n, "label": "X", "question": q, "do": d, "dont": o})
errs = []
names = collections.Counter(i["name"] for p in prompts for i in p["intents"])
ids = collections.Counter(p["id"] for p in prompts)
for p in prompts:
    p["words"] = count_words(p["text"])
    p["ascii_ratio"] = round(sum(c.isascii() for c in p["text"]) / len(p["text"]), 2)
    p["lang"] = "en" if p["ascii_ratio"] > 0.97 else "non-en"
    if not 30 <= p["words"] <= 250: errs.append(f'{p["id"]}: {p["words"]} words')
    if len(p["intents"]) < 3: errs.append(f'{p["id"]}: only {len(p["intents"])} intents')
    if len({i["name"] for i in p["intents"]}) != len(p["intents"]): errs.append(f'{p["id"]}: dup within prompt')
    if not any(i["label"] == "P" for i in p["intents"]) and not any(i["label"]=="N" for i in p["intents"]): errs.append(f'{p["id"]}: no P/N')
errs += [f"dup intent {n} x{c}" for n, c in names.items() if c > 1]
errs += [f"dup id {n}" for n, c in ids.items() if c > 1]
lab = collections.Counter(i["label"] for p in prompts for i in p["intents"])
print("prompts", len(prompts), "intents", sum(lab.values()), dict(lab))
w = [p["words"] for p in prompts]; print("words min/med/max", min(w), sorted(w)[len(w)//2], max(w))
print("non-en prompts", sum(p["lang"]=="non-en" for p in prompts), " single-positive prompts", sum(sum(i["label"]=="P" for i in p["intents"])==1 for p in prompts), " no-positive", sum(not any(i["label"]=="P" for i in p["intents"]) for p in prompts))
import collections as C; print("word bins", sorted(C.Counter(min(p["words"]//50*50,200) for p in prompts).items())); print("intents/prompt", sorted(C.Counter(len(p["intents"]) for p in prompts).items())); print("regions", len({p["region"] for p in prompts}))
print("ERRORS:", errs or "none")
if not errs:
    json.dump({"version": "suite100-v1", "label_status": "provisional (author-written, not independently reviewed)", "prompts": prompts}, open("suite100.json", "w"), ensure_ascii=False, indent=1)
