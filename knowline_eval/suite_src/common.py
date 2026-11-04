# Each prompt: P(id, region, text, (intent_name, label, question, do, dont), ...)
# label: "P" positive (intent clearly present), "N" near-miss (topic present, intent NOT met), "X" negative (unrelated)
def P(pid, region, text, *intents):
    return {"id": pid, "region": region, "text": " ".join(text.split()),
            "intents": [{"name": n, "label": l, "question": q, "do": d, "dont": o} for n, l, q, d, o in intents]}
