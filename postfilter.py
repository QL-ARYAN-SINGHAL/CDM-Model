"""Rule filter applied after the model: drops extractions whose value breaks the character-class or pattern rules, then drops an intent only when
the filter removed the last real value of that family. Messy separators (spaces, line breaks, - / . etc.) are tolerated.
  layer 1 (always): digit-only entities must not contain letters; NAME must not contain digits.
  layer 2 (STRICT_TYPES): length / format rules from the approved pattern sheet (value_rules.py), checked on the value with separators removed.
Switch off with VERITY_POSTFILTER=0. Aadhaar accepts 12 digits (first digit 2-9) and, if AADHAAR_ALLOW_16=1 (default), 16-digit groups (Virtual ID)."""
import os, re
import value_rules as R

ENABLED = os.environ.get("VERITY_POSTFILTER", "1") != "0"
ALLOW_16 = os.environ.get("AADHAAR_ALLOW_16", "1") != "0"
SEPS = " \t\r\n-–—_/.,()+:·|\\"          # tolerated inside digit-only values
DIGIT_ONLY = {"AADHAAR_NUMBER", "CVV", "CARD_NUMBER", "ACCOUNT_NUMBER", "PHONE", "FAX", "SSN", "US_ROUTING_NO", "UK_NATIONAL_HEALTH_SERVICE_NUMBER",
              "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "CA_SOCIAL_INSURANCE_NUMBER", "PIN", "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER"}
ALPHA_ONLY = {"NAME"}
FAMILY = {  # same split as common.FAM_TYPES; NAME / DATE are shared (open under PII or PHI)
 "pii": {"ADDRESS", "PHONE", "FAX", "EMAIL", "SSN", "DRIVING_LICENSE", "PASSPORT_NUMBER", "VOTER_ID", "PASSWORD", "USERNAME", "PAN_NUMBER", "AADHAAR_NUMBER", "IN_NREGA", "CA_SOCIAL_INSURANCE_NUMBER",
         "UK_NATIONAL_INSURANCE_NUMBER", "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER", "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "IP_ADDRESS", "MAC_ADDRESS", "AWS_ACCESS_KEY", "AWS_SECRET_KEY", "LICENSE_PLATE",
         "VEHICLE_IDENTIFICATION_NUMBER", "AGE", "URL", "CERTIFICATE_LICENSE_NUMBER", "DEVICE_IDENTIFIER"},
 "phi": {"MEDICAL_RECORD_NUMBER", "CA_HEALTH_NUMBER", "HEALTH_PLAN_BENEFICIARY_NUMBER", "UK_NATIONAL_HEALTH_SERVICE_NUMBER"},
 "pci": {"CARD_NUMBER", "CVV", "CARD_EXPIRY", "PIN", "ACCOUNT_NUMBER", "INTERNATIONAL_BANK_ACCOUNT_NUMBER", "US_ROUTING_NO", "UPI_ID", "IFSC_CODE", "UTR_NUMBER", "SWIFT_CODE", "WALLET_ID"}}
SHARED = {"NAME", "DATE"}
D = lambda s: re.sub(r"\D", "", s)
NOWS = lambda s: re.sub(r"\s+", "", s)
COMPACT = lambda s: re.sub(r"[\s\-–—_]", "", s)

def _phone(v):
    d = D(v); canon = ("+" if v.lstrip().startswith("+") else "") + d
    return R.v_phone(canon) or R.v_phone(v) or bool(re.fullmatch(r"\d{3}[- ]\d{4}", v.strip()))
def _aadhaar(v):
    d = D(v); return (len(d) == 12 and d[0] in "23456789") or (ALLOW_16 and len(d) == 16)
STRICT = {
 "AADHAAR_NUMBER": _aadhaar, "CVV": lambda v: len(D(v)) in (3, 4), "CARD_NUMBER": lambda v: 13 <= len(D(v)) <= 19 and D(v)[0] in "23456",
 "ACCOUNT_NUMBER": lambda v: 8 <= len(D(v)) <= 18, "PHONE": _phone, "FAX": _phone, "SSN": lambda v: R.v_ssn(D(v)), "US_ROUTING_NO": lambda v: R.v_routing(D(v)),
 "UK_NATIONAL_HEALTH_SERVICE_NUMBER": lambda v: len(D(v)) == 10, "US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER": lambda v: R.v_itin(D(v)), "CA_SOCIAL_INSURANCE_NUMBER": lambda v: R.v_sin(D(v)),
 "PIN": lambda v: len(D(v)) == 4, "UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER": lambda v: len(D(v)) == 10,
 "PAN_NUMBER": lambda v: R.v_pan(COMPACT(v)), "IFSC_CODE": lambda v: R.v_ifsc(COMPACT(v)), "VOTER_ID": lambda v: R.v_voter(COMPACT(v)),
 "EMAIL": lambda v: R.v_email(NOWS(v)), "UPI_ID": lambda v: R.v_upi(NOWS(v)), "UK_NATIONAL_INSURANCE_NUMBER": lambda v: R.v_ni(" ".join(v.split())),
 "SWIFT_CODE": lambda v: R.v_swift(COMPACT(v)), "INTERNATIONAL_BANK_ACCOUNT_NUMBER": lambda v: R.v_iban(NOWS(v)), "VEHICLE_IDENTIFICATION_NUMBER": lambda v: R.v_vin(NOWS(v)),
 "MAC_ADDRESS": lambda v: R.v_mac(NOWS(v)), "IN_NREGA": lambda v: R.v_nrega(NOWS(v)), "CA_HEALTH_NUMBER": lambda v: R.v_cahn(" ".join(v.split()))}

def _trim(text, e):
    """cut trailing/leading sentence punctuation and non-Latin letters glued to the span (e.g. '4829 1567 9034이고'); returns a new entity dict"""
    s, t = e["start"], e["text"]; a, b = 0, len(t)
    while b > a and (t[b - 1] in ".,;:!?\"'”’)]}" or (ord(t[b - 1]) > 127 and t[b - 1].isalpha())): b -= 1
    while a < b and (t[a] in "\"'“‘([{" ): a += 1
    return e if (a, b) == (0, len(t)) else {**e, "text": t[a:b], "start": s + a, "end": s + b}

def check(etype, value):
    """-> None if the value is acceptable, else the reason it is dropped"""
    if etype in DIGIT_ONLY:
        if re.search(r"[^\d" + re.escape(SEPS) + "]", value): return "letters in a digit-only entity"
        if not re.search(r"\d", value): return "no digits"
    if etype in ALPHA_ONLY and re.search(r"\d", value): return "digits in a name"
    if etype in STRICT and not STRICT[etype](value): return "breaks the format rule"
    return None

def apply(text, entities, intents):
    """entities: model's gated entity dicts (type, text, start, end, ...); intents: iterable of 'PII'/'PHI'/'PCI'. Returns (entities, intents, dropped)."""
    if not ENABLED: return entities, list(intents), []
    kept, dropped = [], []
    for e in entities:
        e2 = _trim(text, e) if e["type"] in DIGIT_ONLY | set(STRICT) | ALPHA_ONLY else e
        why = check(e2["type"], e2["text"])
        (dropped if why else kept).append({**e2, "reason": why} if why else e2)
    new_intents = list(intents)
    for fam in ("pii", "phi", "pci"):
        if fam.upper() not in new_intents: continue
        lost_here = any(d["type"] in FAMILY[fam] for d in dropped)
        alive = any(k["type"] in FAMILY[fam] for k in kept)
        if lost_here and not alive: new_intents.remove(fam.upper())   # the filter removed the last real value of this family
    live = {i.lower() for i in new_intents}
    kept = [k for k in kept if (k["type"] in SHARED and (live & {"pii", "phi"})) or any(k["type"] in FAMILY[f] for f in live)]
    return kept, new_intents, dropped
