"""50 hand-authored smoke cases; run each model with its own Python environment.

  /Users/app/Downloads/results/venv/bin/python audit_50.py new
  /Users/app/Intent-FineTune/.venv/bin/python audit_50.py laya
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
OUT = ROOT / "audit_50"
OUT.mkdir(exist_ok=True)

# Expected labels are smoke-test expectations, not adjudicated dataset gold.
# Each positive has one expected exact entity value; negatives have none.
CASES = [
    ("negative", "Hi How are you", [], None),
    ("negative", "Can you send the report tomorrow?", [], None),
    ("negative", "We should review the draft after lunch.", [], None),
    ("negative", "Please enter your email address here.", [], None),
    ("negative", "Do you know your card number?", [], None),
    ("negative", "The website says contact us for details.", [], None),
    ("negative", "Name: [REDACTED], SSN: XXX-XX-XXXX", [], None),
    ("negative", "My payment card ends in 1234.", [], None),
    ("near_miss", "The MAC address format is AA:BB:CC:DD:EE:FF; this is an example.", [], None),
    ("near_miss", "A sample routing number might look like 021000021.", [], None),
    ("positive", "Send the confirmation email to Sarah's address, which is sarah.thompson92@example.com", ["PII"], ("EMAIL", "sarah.thompson92@example.com")),
    ("positive", "My name is Sarah Thompson.", ["PII"], ("NAME", "Sarah Thompson")),
    ("positive", "Deliver to my address: 18 Oak Street, Boston, MA 02116.", ["PII"], ("ADDRESS", "18 Oak Street, Boston, MA 02116")),
    ("positive", "My phone number is +1 415 555 0138.", ["PII"], ("PHONE", "+1 415 555 0138")),
    ("positive", "Our fax number is +1 212 555 0188.", ["PII"], ("FAX", "+1 212 555 0188")),
    ("positive", "My SSN is 123-45-6789.", ["PII"], ("SSN", "123-45-6789")),
    ("positive", "My driver's license number is D1234567.", ["PII"], ("DRIVING_LICENSE", "D1234567")),
    ("positive", "My passport number is K12345678.", ["PII"], ("PASSPORT_NUMBER", "K12345678")),
    ("positive", "My voter ID is ABC1234567.", ["PII"], ("VOTER_ID", "ABC1234567")),
    ("positive", "My password is Summer2026!x.", ["PII"], ("PASSWORD", "Summer2026!x")),
    ("positive", "My username is sarah_t92.", ["PII"], ("USERNAME", "sarah_t92")),
    ("positive", "My PAN number is ABCDE1234F.", ["PII"], ("PAN_NUMBER", "ABCDE1234F")),
    ("positive", "My Aadhaar number is 2345 6789 0123.", ["PII"], ("AADHAAR_NUMBER", "2345 6789 0123")),
    ("positive", "My Canadian social insurance number is 046 454 286.", ["PII"], ("CA_SOCIAL_INSURANCE_NUMBER", "046 454 286")),
    ("positive", "My UK national insurance number is QQ 12 34 56 C.", ["PII"], ("UK_NATIONAL_INSURANCE_NUMBER", "QQ 12 34 56 C")),
    ("positive", "My ITIN is 912-70-1234.", ["PII"], ("US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER", "912-70-1234")),
    ("positive", "My device IP address is 192.168.1.25.", ["PII"], ("IP_ADDRESS", "192.168.1.25")),
    ("positive", "My laptop MAC address is 3C:22:FB:11:22:33.", ["PII"], ("MAC_ADDRESS", "3C:22:FB:11:22:33")),
    ("positive", "My AWS access key ID is AKIAIOSFODNN7EXAMPLE.", ["PII"], ("AWS_ACCESS_KEY", "AKIAIOSFODNN7EXAMPLE")),
    ("positive", "My car license plate is 7ABC123.", ["PII"], ("LICENSE_PLATE", "7ABC123")),
    ("positive", "My car VIN is 1HGCM82633A004352.", ["PII"], ("VEHICLE_IDENTIFICATION_NUMBER", "1HGCM82633A004352")),
    ("positive", "My professional license number is RN123456.", ["PII"], ("CERTIFICATE_LICENSE_NUMBER", "RN123456")),
    ("positive", "My profile link is https://example.com/users/sarah92.", ["PII"], ("URL", "https://example.com/users/sarah92")),
    ("messy", "hey um my email's sarah.thompson92@example.com pls reply there", ["PII"], ("EMAIL", "sarah.thompson92@example.com")),
    ("messy", "can u txt me on +1 415 555 0138 thx", ["PII"], ("PHONE", "+1 415 555 0138")),
    ("positive", "My medical record number is MRN-12345678.", ["PHI"], ("MEDICAL_RECORD_NUMBER", "MRN-12345678")),
    ("positive", "My Canadian health card number is 1234-567-890-AB.", ["PHI"], ("CA_HEALTH_NUMBER", "1234-567-890-AB")),
    ("positive", "My health plan beneficiary number is H123456789.", ["PHI"], ("HEALTH_PLAN_BENEFICIARY_NUMBER", "H123456789")),
    ("positive", "My NHS number is 943 476 5919.", ["PHI"], ("UK_NATIONAL_HEALTH_SERVICE_NUMBER", "943 476 5919")),
    ("positive", "My card number is 4111 1111 1111 1111.", ["PCI"], ("CARD_NUMBER", "4111 1111 1111 1111")),
    ("positive", "My card CVV is 123.", ["PCI"], ("CVV", "123")),
    ("positive", "My card expiry date is 12/29.", ["PCI"], ("CARD_EXPIRY", "12/29")),
    ("positive", "My bank account number is 123456789012.", ["PCI"], ("ACCOUNT_NUMBER", "123456789012")),
    ("positive", "My US bank routing number is 021000021.", ["PCI"], ("US_ROUTING_NO", "021000021")),
    ("positive", "My UPI ID is sarah@okaxis.", ["PCI"], ("UPI_ID", "sarah@okaxis")),
    ("positive", "My IFSC code is HDFC0001234.", ["PCI"], ("IFSC_CODE", "HDFC0001234")),
    ("positive", "My UTR number is 1234567890123456.", ["PCI"], ("UTR_NUMBER", "1234567890123456")),
    ("positive", "My SWIFT code is CHASUS33XXX.", ["PCI"], ("SWIFT_CODE", "CHASUS33XXX")),
    ("positive", "My wallet ID is wallet_123456789.", ["PCI"], ("WALLET_ID", "wallet_123456789")),
    ("messy", "uh my card is 4111 1111 1111 1111 pls use it", ["PCI"], ("CARD_NUMBER", "4111 1111 1111 1111")),
]
assert len(CASES) == 50

def base_row(i, case):
    category, prompt, expected_intents, expected_entity = case
    return {"id": i + 1, "category": category, "prompt": prompt,
            "expected_intents": expected_intents,
            "expected_entity": expected_entity}

if sys.argv[1] in ("new", "new5"):
    import torch
    from gliner2 import AutoExtractor
    sys.path.insert(0, str(ROOT / "gliner2_spike"))
    import common
    model = AutoExtractor.from_pretrained(str(ROOT / "best")); model.eval()
    rows = []
    with torch.no_grad():
        for offset in range(0, 50, 10):
            part = CASES[offset:offset + 10]
            texts = [c[1] for c in part]
            probs = common.predict_intent(model, texts, batch_size=10)
            ents = common.predict_entities(model, texts, batch_size=10, threshold=0.5,
                                           chunk=5 if sys.argv[1] == "new5" else 0)
            for j, (case, p, es) in enumerate(zip(part, probs, ents)):
                row = base_row(offset + j, case)
                row.update({"scores": p, "intents": [k.upper() for k, v in p.items() if v >= 0.5],
                            "entities": [{"type": t, "text": case[1][s:e], "score": round(float(c), 4), "start": s, "end": e} for t, s, e, c in es]})
                rows.append(row)
    target = OUT / ("new5.json" if sys.argv[1] == "new5" else "new.json")
elif sys.argv[1] == "laya":
    import torch
    import laya
    agent = laya.Agent('/Users/app/Downloads/Laya-Finetune/laya_finetuned', device='mps' if torch.backends.mps.is_available() else 'cpu')
    Q = {"pii": "Does the text contain an actual personal identifier value (ID number, phone, email, address, etc.) that belongs to a named person or to the writer? Fiction and label-only mentions don't count.",
         "phi": "Does the text contain a health ID value, or a diagnosis, medication or test tied to a named person? Fiction and label-only mentions don't count.",
         "pci": "Does the text contain a usable payment or bank value (card, account, IBAN, UPI, routing, IFSC, SWIFT, UTR, wallet) shown with its label (UPI needs none)? Masked, invalid and label-only mentions don't count."}
    schema = {k: {"type": "noul", "instructions": v} for k, v in Q.items()}
    rows = []
    with torch.no_grad():
        for i, case in enumerate(CASES):
            a = agent.predict(case[1], schema)['answers']
            p = {k: float(a[k]['noul']) for k in Q}
            row = base_row(i, case)
            row.update({"scores": p, "intents": [k.upper() for k, v in p.items() if v >= 0.7]})
            rows.append(row)
    target = OUT / "laya.json"
else:
    raise SystemExit('usage: audit_50.py new|new5|laya')

target.write_text(json.dumps(rows, indent=2))
print(target)
