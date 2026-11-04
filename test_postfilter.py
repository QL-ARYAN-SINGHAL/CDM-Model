"""python3 test_postfilter.py  -> runs the filter on hand-written cases (messy separators kept, letters / wrong lengths dropped)"""
import postfilter as P
ent = lambda t, v, s=0: {"type": t, "text": v, "start": s, "end": s + len(v), "score": 1.0}
def fam(t): return next((f.upper() for f in ("pii", "phi", "pci") if t in P.FAMILY[f]), "PII")   # NAME / DATE -> PII
def run(t, v): return P.apply("x " + v, [ent(t, v, 2)], [fam(t)])
KEEP = [("AADHAAR_NUMBER", "2345 6789 0123"), ("AADHAAR_NUMBER", "2345 - 6789 - 0123"), ("AADHAAR_NUMBER", "2345-6789-0123"), ("AADHAAR_NUMBER", "2345\n6789\n0123"), ("AADHAAR_NUMBER", "2345 / 6789 / 0123"),
        ("AADHAAR_NUMBER", "9867 - 5615 - 9216- 1281"), ("CVV", "123"), ("CVV", "4521"), ("PAN_NUMBER", "ABCPD1234E"), ("PAN_NUMBER", "abcpd1234e"), ("PAN_NUMBER", "ABCPD 1234 E"), ("PAN_NUMBER", "ABCPD-1234-E"),
        ("PHONE", "+91\n99111\n31674"), ("PHONE", "+91 98765 43210"), ("PHONE", "(704) 553-0182"), ("EMAIL", "testing@gmail.com"), ("EMAIL", "priya.k\n@\nexample.com"), ("UPI_ID", "amit@okaxis"), ("CARD_NUMBER", "4111 1111 1111 1111"),
        ("CARD_NUMBER", "4532-0151-1283-0366"), ("ACCOUNT_NUMBER", "9876 8762 1231 0876"), ("SSN", "137-56-9781"), ("IFSC_CODE", "HDFC0001234"), ("NAME", "Rahul Sharma")]
DROP = [("AADHAAR_NUMBER", "asdnjkkjnda"), ("AADHAAR_NUMBER", "83839das"), ("AADHAAR_NUMBER", "2345678901"), ("AADHAAR_NUMBER", "2345 6789 012"), ("AADHAAR_NUMBER", "0123 4567 8901"), ("CVV", "12"), ("CVV", "12345"),
        ("CVV", "12a"), ("PAN_NUMBER", "ABCPD1234EXTRA"), ("PAN_NUMBER", "ABCPD12345"), ("PAN_NUMBER", "asdnjkkjnda"), ("PAN_NUMBER", "ABCDE2132F"), ("PHONE", "asdnjkkjnda"), ("PHONE", "98765abcde"), ("EMAIL", "asdnjkkjnda"),
        ("EMAIL", "name@gmail"), ("UPI_ID", "Suny@Aryan"), ("UPI_ID", "asdnjkkjnda"), ("CARD_NUMBER", "asdnjkkjnda"), ("ACCOUNT_NUMBER", "asdnjkkjnda"), ("SSN", "666-12-3456"), ("NAME", "Rahul123"), ("IFSC_CODE", "HDFC1001234")]
for t, v in KEEP:
    k, i, d = run(t, v); assert k and not d, ("should keep", t, v, d)
for t, v in DROP:
    k, i, d = run(t, v); assert d and not k, ("should drop", t, v, k)
    if t != "NAME": assert i == [], ("intent should be removed when its last value is dropped", t, v, i)
k, i, d = P.apply("x", [ent("AADHAAR_NUMBER", "asdnjkkjnda", 0), ent("CARD_NUMBER", "4111 1111 1111 1111", 0)], ["PII", "PCI"]); assert i == ["PCI"] and len(k) == 1   # one family survives
k, i, d = P.apply("Pooja is nice. aadhar asdnjkkjnda", [ent("NAME", "Pooja", 0), ent("AADHAAR_NUMBER", "asdnjkkjnda", 22)], ["PII"]); assert k == [] and i == []   # NAME closes with the intent
k, i, d = P.apply("4829 1567 9034이고", [ent("AADHAAR_NUMBER", "4829 1567 9034이고", 0)], ["PII"]); assert k and k[0]["text"] == "4829 1567 9034"   # non-Latin tail trimmed
print(f"postfilter OK: {len(KEEP)} values kept, {len(DROP)} dropped")
