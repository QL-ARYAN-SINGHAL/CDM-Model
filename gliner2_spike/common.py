"""Shared pieces for the GLiNER2 spike: dual-task rows (data/final/v5_train) -> GLiNER2 training records, inference wrappers, metrics.
Task layout per example: three binary classification tasks (pii / phi / pci, labels yes|no) + entity labels with pinned character spans.
Teaching 'no intent -> no extraction': types that occur in the text but are not targets (gate closed, masked / invalid values) are declared with an EMPTY mention list = trained negative.
Types the dataset cannot judge (ignore / ignore_types) are left out of that example's schema, so they get neither a positive nor a negative."""
import hashlib, json, math, os, random, re, time
from collections import Counter, defaultdict

NATURAL = {  # extract type -> label text the model sees (natural words generalise better than CAPS_UNDERSCORE)
 "NAME": "person name", "ADDRESS": "postal address", "DATE": "date", "PHONE": "phone number", "FAX": "fax number", "EMAIL": "email address", "SSN": "social security number",
 "DRIVING_LICENSE": "driving licence number", "PASSPORT_NUMBER": "passport number", "VOTER_ID": "voter id", "PASSWORD": "password", "USERNAME": "username", "PAN_NUMBER": "pan number",
 "AADHAAR_NUMBER": "aadhaar number", "IN_NREGA": "nrega job card number", "CA_SOCIAL_INSURANCE_NUMBER": "canadian social insurance number", "UK_NATIONAL_INSURANCE_NUMBER": "uk national insurance number",
 "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER": "uk taxpayer reference", "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER": "us itin", "IP_ADDRESS": "ip address", "MAC_ADDRESS": "mac address",
 "AWS_ACCESS_KEY": "aws access key id", "AWS_SECRET_KEY": "aws secret key", "LICENSE_PLATE": "license plate", "VEHICLE_IDENTIFICATION_NUMBER": "vehicle identification number", "AGE": "age",
 "URL": "profile url", "CERTIFICATE_LICENSE_NUMBER": "professional license number", "DEVICE_IDENTIFIER": "device identifier", "MEDICAL_RECORD_NUMBER": "medical record number",
 "CA_HEALTH_NUMBER": "canadian health card number", "HEALTH_PLAN_BENEFICIARY_NUMBER": "health plan beneficiary number", "UK_NATIONAL_HEALTH_SERVICE_NUMBER": "nhs number",
 "CARD_NUMBER": "payment card number", "CVV": "card cvv", "CARD_EXPIRY": "card expiry date", "PIN": "card pin", "ACCOUNT_NUMBER": "bank account number",
 "INTERNATIONAL_BANK_ACCOUNT_NUMBER": "iban", "US_ROUTING_NO": "routing number", "UPI_ID": "upi id", "IFSC_CODE": "ifsc code", "UTR_NUMBER": "utr number", "SWIFT_CODE": "swift code", "WALLET_ID": "wallet id"}
BACK = {v: k for k, v in NATURAL.items()}
FAMILIES = ("pii", "phi", "pci")
FAM_TYPES = {  # same split as gen/dual_task.py (NAME and DATE are shared: PII or PHI)
 "pii": {"ADDRESS", "PHONE", "FAX", "EMAIL", "SSN", "DRIVING_LICENSE", "PASSPORT_NUMBER", "VOTER_ID", "PASSWORD", "USERNAME", "PAN_NUMBER", "AADHAAR_NUMBER", "IN_NREGA", "CA_SOCIAL_INSURANCE_NUMBER",
         "UK_NATIONAL_INSURANCE_NUMBER", "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER", "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "IP_ADDRESS", "MAC_ADDRESS", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "LICENSE_PLATE",
         "VEHICLE_IDENTIFICATION_NUMBER", "AGE", "URL", "CERTIFICATE_LICENSE_NUMBER", "DEVICE_IDENTIFIER"},
 "phi": {"MEDICAL_RECORD_NUMBER", "CA_HEALTH_NUMBER", "HEALTH_PLAN_BENEFICIARY_NUMBER", "UK_NATIONAL_HEALTH_SERVICE_NUMBER"},
 "pci": {"CARD_NUMBER", "CVV", "CARD_EXPIRY", "PIN", "ACCOUNT_NUMBER", "INTERNATIONAL_BANK_ACCOUNT_NUMBER", "US_ROUTING_NO", "UPI_ID", "IFSC_CODE", "UTR_NUMBER", "SWIFT_CODE", "WALLET_ID"}}
SHARED = {"NAME", "DATE"}
ALL_TYPES = sorted(NATURAL)
assert set(ALL_TYPES) == set().union(*FAM_TYPES.values()) | SHARED, "label tables out of sync"
CLS_LABELS = ["yes", "no"]


def family_open(typ, intent):
    """is extraction allowed for this type given an intent dict {pii,phi,pci} of 0/1"""
    return bool(intent["pii"] or intent["phi"]) if typ in SHARED else any(intent[f] and typ in FAM_TYPES[f] for f in FAMILIES)


def _rng(key): return random.Random(int(hashlib.md5(key.encode()).hexdigest()[:12], 16))


def to_record(row, max_labels=16, full_prob=0.25, seed=0):
    """dual-task row -> GLiNER2 raw record {'input', 'output'}. Schema = positives + hard negatives (+ random negatives up to max_labels); 25% of rows carry the full label set."""
    r = _rng(f"{seed}|{row['id']}"); text = row["text"]
    dropped = set(row.get("ignore_types", [])) | {g["type"] for g in row.get("ignore", [])}
    pos = defaultdict(list)
    for g in sorted(row["targets"], key=lambda g: g["start"]):
        if g["type"] not in dropped: pos[g["type"]].append({"text": text[g["start"]:g["end"]], "start": g["start"], "end": g["end"]})
    hard = [t for t in row.get("neg_types", []) if t not in pos and t not in dropped]
    rest = [t for t in ALL_TYPES if t not in pos and t not in dropped and t not in hard]
    if r.random() < full_prob: negs = hard + rest
    else:
        k = max(0, max_labels - len(pos) - len(hard)); negs = hard + r.sample(rest, min(k, len(rest)))
    ents = {NATURAL[t]: v for t, v in pos.items()}
    for t in negs: ents[NATURAL[t]] = []
    keys = list(ents); r.shuffle(keys); ents = {k: ents[k] for k in keys}
    cls = [{"task": f, "labels": CLS_LABELS, "true_label": "yes" if row["labels"][f] else "no"} for f in FAMILIES]
    return {"input": text, "output": {"entities": ents, "classifications": cls}}


def write_jsonl(rows, path, **kw):
    n = 0
    with open(path, "w") as f:
        for row in rows: f.write(json.dumps(to_record(row, **kw), ensure_ascii=False) + "\n"); n += 1
    return n


# ----------------------------------------------------------------------------------------------- inference
def cls_schema(): return {f: CLS_LABELS for f in FAMILIES}


def predict_joint(model, texts, batch_size=16, threshold=0.5):
    """Use the combined training schema; return independent intent scores and raw spans."""
    schema = model.create_schema().entities([NATURAL[t] for t in ALL_TYPES])
    for family in FAMILIES:
        schema.classification(family, CLS_LABELS)
    results = model.batch_extract(texts, schema, batch_size=batch_size,
                                  threshold=threshold, include_spans=True, include_confidence=True)
    probabilities, mentions = [], []
    for result in results:
        scores = {}
        for family in FAMILIES:
            answer = result[family]
            scores[family] = float(answer["confidence"] if answer["label"] == "yes" else 1.0 - answer["confidence"])
        probabilities.append(scores)
        mentions.append([(BACK[label], m["start"], m["end"], float(m["confidence"]))
                         for label, values in result.get("entities", {}).items() if label in BACK
                         for m in values])
    return probabilities, mentions


def predict_intent(model, texts, batch_size=32):
    """-> list of {pii,phi,pci} probabilities of 'yes'"""
    out = []
    res = model.batch_classify_text(texts, cls_schema(), batch_size=batch_size, include_confidence=True)
    for r in res:
        d = {}
        for f in FAMILIES:
            x = r[f]; lab, conf = (x["label"], x["confidence"]) if isinstance(x, dict) else (x, 1.0)
            d[f] = conf if lab == "yes" else 1.0 - conf
        out.append(d)
    return out


def predict_entities(model, texts, types=None, batch_size=16, threshold=0.5, chunk=0):
    """-> list of [(type, start, end, confidence)]; chunk > 0 asks the labels in groups of that size"""
    types = types or ALL_TYPES; groups = [types] if not chunk else [types[i:i + chunk] for i in range(0, len(types), chunk)]
    out = [[] for _ in texts]
    for g in groups:
        res = model.batch_extract_entities(texts, [NATURAL[t] for t in g], batch_size=batch_size, include_spans=True, include_confidence=True, threshold=threshold)
        for i, r in enumerate(res):
            for lab, ms in r["entities"].items():
                for m in ms:
                    if lab in BACK: out[i].append((BACK[lab], m["start"], m["end"], m.get("confidence", 1.0)))
    return out


# ----------------------------------------------------------------------------------------------- metrics
def prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0; r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0, "tp": tp, "fp": fp, "fn": fn}


def intent_metrics(rows, probs, thr=0.5):
    out = {}
    for f in FAMILIES:
        tp = sum(1 for r, p in zip(rows, probs) if r["labels"][f] and p[f] >= thr); fp = sum(1 for r, p in zip(rows, probs) if not r["labels"][f] and p[f] >= thr)
        fn = sum(1 for r, p in zip(rows, probs) if r["labels"][f] and p[f] < thr); out[f] = prf(tp, fp, fn)
    acc = sum(all((p[f] >= thr) == bool(r["labels"][f]) for f in FAMILIES) for r, p in zip(rows, probs)) / max(1, len(rows))
    none_rows = [(r, p) for r, p in zip(rows, probs) if not any(r["labels"].values())]
    out["exact_row_accuracy"] = round(acc, 4); out["none_fp_rate"] = round(sum(any(p[f] >= thr for f in FAMILIES) for _, p in none_rows) / max(1, len(none_rows)), 4)
    return out


def best_threshold(rows, probs, f):
    best = (0.5, -1)
    for t in [i / 20 for i in range(1, 20)]:
        m = prf(sum(1 for r, p in zip(rows, probs) if r["labels"][f] and p[f] >= t), sum(1 for r, p in zip(rows, probs) if not r["labels"][f] and p[f] >= t), sum(1 for r, p in zip(rows, probs) if r["labels"][f] and p[f] < t))
        if m["f1"] > best[1]: best = (t, m["f1"])
    return best


def _masked(row):
    """(types the row cannot judge, ignore ranges)"""
    return set(row.get("ignore_types", [])), row.get("ignore", [])


def extract_metrics(rows, preds, gate_probs=None, thr=0.5):
    """exact (type, start, end) match micro P/R/F1 + per type. gate_probs: apply 'no intent -> no extraction' per family using the model's own intent scores."""
    per = defaultdict(lambda: [0, 0, 0]); tp = fp = fn = 0; rows_with_pred_on_none = 0; none_rows = 0
    for i, (row, pr) in enumerate(zip(rows, preds)):
        skip_t, ign = _masked(row)
        gold = {(g["type"], g["start"], g["end"]) for g in row["targets"] if g["type"] not in skip_t}
        pset = set()
        for t, s, e, c in pr:
            if t in skip_t or any(s < g["end"] and e > g["start"] and g["type"] == t for g in ign): continue
            if gate_probs is not None and not family_open(t, {f: gate_probs[i][f] >= thr for f in FAMILIES}): continue
            pset.add((t, s, e))
        if not any(row["labels"].values()):
            none_rows += 1; rows_with_pred_on_none += bool(pset)
        for k in gold & pset: per[k[0]][0] += 1
        for k in pset - gold: per[k[0]][1] += 1
        for k in gold - pset: per[k[0]][2] += 1
        tp += len(gold & pset); fp += len(pset - gold); fn += len(gold - pset)
    out = {"micro": prf(tp, fp, fn), "per_type": {t: prf(*v) for t, v in sorted(per.items())}, "intent0_rows_with_any_extraction": round(rows_with_pred_on_none / max(1, none_rows), 4)}
    return out


def fmt_eta(sec):
    sec = int(max(0, sec)); h, m = sec // 3600, sec % 3600 // 60
    return f"{h}h{m:02d}m" if h else f"{m}m{sec % 60:02d}s"
