"""Summarize saved 50-case outputs; smoke expectations need manual validity review."""
import csv
import json
from collections import Counter
from pathlib import Path

root = Path(__file__).parent / "audit_50"
new = json.loads((root / "new.json").read_text())
new5 = json.loads((root / "new5.json").read_text())
laya = json.loads((root / "laya.json").read_text())
assert len(new) == len(new5) == len(laya) == 50

with (root / "comparison.csv").open("w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["id", "category", "prompt", "expected_intents_smoke", "expected_entity_smoke",
                     "dual_intents", "dual_pii", "dual_phi", "dual_pci", "dual_entities",
                     "laya_intents", "laya_pii", "laya_phi", "laya_pci", "entity_exact_match"])
    for a, b in zip(new, laya):
        expected = a["expected_entity"]
        match = bool(expected and any(e["type"] == expected[0] and e["text"] == expected[1] for e in a["entities"]))
        writer.writerow([a["id"], a["category"], a["prompt"], ";".join(a["expected_intents"]),
                         ": ".join(expected) if expected else "", ";".join(a["intents"]),
                         *[a["scores"][k] for k in ("pii", "phi", "pci")],
                         json.dumps(a["entities"]), ";".join(b["intents"]),
                         *[b["scores"][k] for k in ("pii", "phi", "pci")], int(match)])

def count_exact(rows):
    return sum(set(r["intents"]) == set(r["expected_intents"]) for r in rows)

expected_mentions = [r for r in new if r["expected_entity"]]
matched = [r for r in expected_mentions if any(e["type"] == r["expected_entity"][0] and e["text"] == r["expected_entity"][1] for e in r["entities"])]
matched5 = sum(any(e["type"] == r["expected_entity"][0] and e["text"] == r["expected_entity"][1] for e in r["entities"]) for r in new5 if r["expected_entity"])
missed = Counter(r["expected_entity"][0] for r in expected_mentions if r not in matched)
categories = sorted({r["category"] for r in new})
lines = ["# 50 prompt smoke audit", "", "Hand authored prompts and labels. Some identifier values may be synthetic, invalid, or out of policy; mismatches require manual review before becoming training data.",
         "", "Models: dual-task checkpoint in `../best`; Laya intent checkpoint in `/Users/app/Downloads/Laya-Finetune/laya_finetuned`.",
         "Dual threshold 0.5 for each binary intent; Laya threshold 0.7. Extraction uses all 45 trained labels and threshold 0.5. Laya has no extraction head.",
         "", f"Dual intent exact match against smoke expectations: {count_exact(new)}/50. Laya: {count_exact(laya)}/50.",
         f"Dual extraction exact type and span: {len(matched)}/{len(expected_mentions)} positive prompts. This is a strict smoke check, not validated recall.",
         f"Five-label extraction chunks: {matched5}/{len(expected_mentions)} strict matches. Both schemas emitted no entities on the 10 negatives. Chunking changes the inference task and is not equivalent to published evaluation.",
         f"Models agree on intent flags: {sum(set(a['intents']) == set(b['intents']) for a,b in zip(new,laya))}/50.",
         "", "## By prompt category", "", "| Category | Cases | Dual exact | Laya exact |", "|---|---:|---:|---:|"]
for cat in categories:
    a = [r for r in new if r["category"] == cat]
    b = [r for r in laya if r["category"] == cat]
    lines.append(f"| {cat} | {len(a)} | {count_exact(a)} | {count_exact(b)} |")
lines += ["", "## Observations", "", "- Greeting and Sarah email classify correctly in both raw model APIs. No PCI false positives among the 10 negative/near-miss prompts in either model.",
          "- Several synthetic PCI positives get no PCI flag from either model. Payment validity and dataset label policy must be checked before treating these as false negatives.",
          "- Extraction is schema sensitive: all 45 labels produced no EMAIL for Sarah prompt, while EMAIL plus NAME labels produced EMAIL confidence 0.9996 in direct probe. Keep schema fixed when comparing with published evaluation.",
          "- Previous UI PCI results came from entity-derived priority logic and a fabricated 0.99 confidence, not the trained classification output.",
          "- Laya used a different intent training file (`laya_train_25.jsonl`) from the dual-task checkpoint (`v5_train/dual_train_100.jsonl`); they are related corpora but not identical training runs.",
          "- Requested taxonomy includes aliases: TELEPHONE=PHONE, DRIVER_ID=DRIVING_LICENSE, IN_PERMANENT_ACCOUNT_NUMBER=PAN_NUMBER, IN_VOTER_NUMBER=VOTER_ID, CREDIT_DEBIT_NUMBER=CARD_NUMBER, CREDIT_DEBIT_CVV=CVV, CREDIT_DEBIT_EXPIRY=CARD_EXPIRY, BANK_ROUTING=US_ROUTING_NO. These are not separately trained types.",
          "", "## Extraction cases needing review", "", "| Type | Strict misses |", "|---|---:|"]
for typ, n in missed.most_common():
    lines.append(f"| {typ} | {n} |")
(root / "README.md").write_text("\n".join(lines) + "\n")
print(root / "comparison.csv")
print(root / "README.md")
