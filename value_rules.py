"""Value rules (validators only), copied from v9_repair/values.py (approved pattern sheet). Used by postfilter.py; no checksums."""
import random, re, string, datetime as dt

UPI_HANDLES = "okaxis oksbi okhdfcbank okicici ybl okybl ibl axl paytm ptyes pthdfc apl upi sbi hdfcbank icici axisbank kotak".split()
PAN_4TH = "PCFHABGLJT"
D = lambda s: re.sub(r"\D", "", s)
MASH = ["asdnjkkjnda", "qwertyuiop", "asdfghjkl", "zxcvbnm", "hjkhjkhjk", "lkjhgfdsa", "xyzabc", "abcdefgh", "sdfsdfsdf", "ghjkl", "poiuytre", "mnbvcxz", "kjdsfkjsd", "aaaaaaaa", "qazwsxedc", "hfgjdkslq", "wxyzwxyz", "bnmbnm", "tyutyu", "rtyrtyrty"]
WORDS = ["hello", "pending", "not available", "tbd", "unknown", "will share later", "n/a", "xxxxx", "none", "confidential", "see attachment", "asap", "yes", "ok", "check", "dont have", "forgot", "later", "na", "nil", "******", "test", "dummy value", "abc", "lorem ipsum", "private", "redacted", "-"]
def gib(r): return r.choice(MASH) if r.random() < .6 else "".join(r.choice(string.ascii_lowercase) for _ in range(r.randint(5, 13)))
def word(r): return r.choice(WORDS)
def rd(r, n, first=None):
    s = "".join(r.choice("0123456789") for _ in range(n))
    return (r.choice(first) + s[1:]) if first else s
def rl(r, n, up=True): return "".join(r.choice(string.ascii_uppercase if up else string.ascii_lowercase) for _ in range(n))
def luhn_fix(d):
    s = 0
    for i, ch in enumerate(reversed(d)):
        n = int(ch) * (2 if i % 2 == 0 else 1); s += n - 9 if n > 9 else n
    return d + str((10 - s % 10) % 10)
def trivial(d): return len(set(d)) == 1 or d in "01234567890123456789" or d in "98765432109876543210"

# ------------------------------------------------------------------------------------------------ validators
def _sep(v, chars=" -"): return re.sub("[" + re.escape(chars) + "]", "", v)
def v_aadhaar(v):
    m = re.fullmatch(r"(\d{4})(\s*[-–/_.]?\s*)(\d{4})(\s*[-–/_.]?\s*)(\d{4})", v.strip())
    return bool(m) and m.group(1)[0] in "23456789"
def v_cvv(v): return bool(re.fullmatch(r"\d{3,4}", v.strip()))
def v_card(v):
    if not re.fullmatch(r"[\d \-]+", v.strip()): return False
    d = D(v); return 13 <= len(d) <= 19 and d[0] in "23456" and not trivial(d)
def v_account(v):
    if not re.fullmatch(r"[\d \-]+", v.strip()): return False
    d = D(v); return 8 <= len(d) <= 18 and not trivial(d) and len(set(d)) > 2
def v_phone(v):
    s = v.strip()
    if re.search(r"[A-Za-z]", s) or not re.fullmatch(r"\+?[\d \-().]+", s): return False
    d = D(s); plus = s.startswith("+")
    if re.fullmatch(r"\d{3}[- ]\d{4}", s): return True   # 7-digit local number, kept as is
    if not 8 <= len(d) <= 15: return False
    if plus and d.startswith("91"): return len(d) == 12 and d[2] in "123456789"
    if plus and d.startswith("1"): return len(d) == 11 and d[1] in "23456789" and d[4] in "23456789"
    if plus and d.startswith("44"): return len(d) in (11, 12) and d[2] != "0"
    if plus: return True
    if len(d) == 12 and d.startswith("91"): return d[2] in "123456789"
    if len(d) == 11 and d.startswith("0"): return True
    if len(d) == 11 and d.startswith("1"): return d[1] in "23456789" and d[4] in "23456789"
    if len(d) == 10: return d[0] in "23456789"
    return len(d) >= 8
def v_ssn(v):
    if not re.fullmatch(r"\d{3}[- .]?\d{2}[- .]?\d{4}", v.strip()): return False
    d = D(v); return "001" <= d[:3] <= "899" and d[:3] != "666" and d[3:5] != "00" and d[5:] != "0000"
def v_routing(v):
    if not re.fullmatch(r"[\d \-]+", v.strip()): return False
    d = D(v); return len(d) == 9 and (1 <= int(d[:2]) <= 12 or 21 <= int(d[:2]) <= 32 or 61 <= int(d[:2]) <= 72 or d[:2] == "80")
def v_nhs(v): return bool(re.fullmatch(r"\d{3}[- ]?\d{3}[- ]?\d{4}", v.strip()))
def v_itin(v):
    if not re.fullmatch(r"\d{3}[- ]?\d{2}[- ]?\d{4}", v.strip()): return False
    d = D(v); m = int(d[3:5]); return d[0] == "9" and (50 <= m <= 65 or 70 <= m <= 88 or 90 <= m <= 92 or 94 <= m <= 99)
def v_sin(v): return bool(re.fullmatch(r"[1-79]\d{2}[- ]?\d{3}[- ]?\d{3}", v.strip()))
def v_pin(v): return bool(re.fullmatch(r"\d{4}", v.strip()))
def v_ukutr(v): return bool(re.fullmatch(r"\d{5} ?\d{5}", v.strip()))
def v_utr(v):
    s = v.strip(); return bool(re.fullmatch(r"\d{12}|[A-Za-z0-9]{16}|[A-Za-z0-9]{22}", s)) and not (s.isalpha())
def v_expiry(v):
    s = v.strip(); m = re.fullmatch(r"(\d{2})\s?[/-]\s?(\d{2}|\d{4})", s)
    if m: return 1 <= int(m.group(1)) <= 12
    return bool(re.fullmatch(r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{4}", s))
def v_pan(v):
    m = re.fullmatch(r"([A-Za-z]{5})[ \-]?(\d{4})[ \-]?([A-Za-z])", v.strip()); return bool(m) and m.group(1)[3].upper() in PAN_4TH
def v_ifsc(v): return bool(re.fullmatch(r"[A-Za-z]{4}[ \-]?0[A-Za-z0-9]{6}", v.strip()))
def v_voter(v): return bool(re.fullmatch(r"[A-Za-z]{3} ?\d{7}", v.strip()))
def v_dl(v): return bool(re.fullmatch(r"[A-Za-z]{2}[ -]?\d{2}[ -]?\d{11}", v.strip()))
def v_passport(v): return bool(re.fullmatch(r"[A-Za-z]\d{7}|\d{9}|[A-Za-z]{2}\d{7}|[A-Za-z]\d{8}", v.strip()))
def v_vin(v): return bool(re.fullmatch(r"[A-HJ-NPR-Za-hj-npr-z0-9]{17}", v.strip()))
def v_plate(v):
    s = v.strip()
    if re.fullmatch(r"[A-Za-z]{2}[ -]?\d{1,2}[ -]?[A-Za-z]{1,3}[ -]?\d{4}|\d{2}[ -]?BH[ -]?\d{4}[ -]?[A-Za-z]{1,2}|[A-Za-z]{2}\d{2} ?[A-Za-z]{3}", s, re.I): return True
    return bool(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9 \-]{2,9}[A-Za-z0-9]", s)) and bool(re.search(r"\d", s)) and bool(re.search(r"[A-Za-z]", s))
IBAN_LEN = dict(GB=22, DE=22, FR=27, ES=24, IT=27, NL=18, BE=16, CH=21, IE=22, AT=20, PT=25, SE=24, NO=15, DK=18, PL=28, AE=23, SA=24, FI=18, LU=20, GR=27, CZ=24, HU=28, TR=26)
def v_iban(v):
    s = v.replace(" ", "").upper()
    return bool(re.fullmatch(r"[A-Z]{2}\d{2}[A-Z0-9]+", s)) and IBAN_LEN.get(s[:2]) == len(s)
def v_swift(v):
    s = v.strip(); return bool(re.fullmatch(r"[A-Za-z]{6}[A-Za-z0-9]{2}([A-Za-z0-9]{3})?", s)) and "XX" not in s[:6].upper()
def v_ni(v):
    s = v.strip().upper()
    if not re.fullmatch(r"[A-CEGHJ-PR-TW-Z][A-CEGHJ-NPR-TW-Z] ?\d{2} ?\d{2} ?\d{2} ?[A-D]", s): return False
    return s[:2] not in ("BG", "GB", "NK", "KN", "TN", "NT", "ZZ")
def v_mac(v): return bool(re.fullmatch(r"([0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}|[0-9A-Fa-f]{12}|([0-9A-Fa-f]{4}\.){2}[0-9A-Fa-f]{4}", v.strip()))
def v_nrega(v): return bool(re.fullmatch(r"[A-Za-z]{2}-\d{2}-\d{3}-\d{3}-\d{3}/\d{2,5}|[A-Za-z]{2}-\d{2}-\d{3}-\d{3}-\d{6}|[A-Za-z]{2}\d{14}", v.strip()))
def v_cahn(v): return bool(re.fullmatch(r"\d{4}[- ]?\d{3}[- ]?\d{3}( ?[A-Za-z]{2})?", v.strip()))
def v_ip(v):
    p = v.strip().split(".")
    return len(p) == 4 and all(re.fullmatch(r"0|[1-9]\d{0,2}", x) and int(x) <= 255 for x in p)
def v_date(v):
    s = v.strip()
    for f in ("%d/%m/%Y", "%m/%d/%Y", "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d.%m.%Y", "%d %B %Y", "%B %d, %Y", "%d %b %Y", "%d%m%Y", "%m%d%Y", "%d/%m/%y", "%m/%d/%y"):
        try: dt.datetime.strptime(s, f); return True
        except ValueError: pass
    return False
def v_wallet(v): return bool(re.fullmatch(r"(WAL|WALLET)[ -]?\d{6,11}", v.strip(), re.I))
def v_mrn(v): return bool(re.fullmatch(r"([A-Za-z]{2,4}(-[A-Za-z]{2,4})?-?)?\d{6,10}", v.strip()))
def v_hpbn(v): return bool(re.fullmatch(r"([A-Za-z]{2,4}-?)?\d{8,10}", v.strip()))
def v_cert(v): return bool(re.fullmatch(r"([A-Za-z]{2,4}(-[A-Za-z]{2,4})?-?)?\d{5,8}", v.strip()))
def v_device(v): return bool(re.fullmatch(r"\d{14,16}|[A-Za-z]{2,4}(-[A-Za-z]{2,3})?-?\d{6,11}(-[A-Za-z]{2})?", v.strip()))
def v_email(v):
    s = v.strip()
    return bool(re.fullmatch(r"[A-Za-z0-9](?:[A-Za-z0-9._%+\-]{0,62}[A-Za-z0-9_])?@(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,}", s)) and ".." not in s
def v_upi(v):
    m = re.fullmatch(r"([A-Za-z0-9._\-]{1,50})@([A-Za-z]+)", v.strip()); return bool(m) and m.group(2).lower() in UPI_HANDLES
def v_url(v): return bool(re.fullmatch(r"(https?://)?(www\.)?([A-Za-z0-9\-]+\.)+[A-Za-z]{2,}(/\S*)?", v.strip()))
def v_age(v): return bool(re.fullmatch(r"\d{1,2}", v.strip())) and int(v) >= 1

VALID = dict(AADHAAR_NUMBER=v_aadhaar, CVV=v_cvv, CARD_NUMBER=v_card, ACCOUNT_NUMBER=v_account, PHONE=v_phone, FAX=v_phone, SSN=v_ssn, US_ROUTING_NO=v_routing,
    UK_NATIONAL_HEALTH_SERVICE_NUMBER=v_nhs, US_INDIVIDUAL_TAX_IDENTIFICATION_NUMBER=v_itin, CA_SOCIAL_INSURANCE_NUMBER=v_sin, PIN=v_pin, UK_UNIQUE_TAXPAYER_REFERENCE_NUMBER=v_ukutr,
    UTR_NUMBER=v_utr, CARD_EXPIRY=v_expiry, PAN_NUMBER=v_pan, IFSC_CODE=v_ifsc, VOTER_ID=v_voter, DRIVING_LICENSE=v_dl, PASSPORT_NUMBER=v_passport, VEHICLE_IDENTIFICATION_NUMBER=v_vin,
    LICENSE_PLATE=v_plate, INTERNATIONAL_BANK_ACCOUNT_NUMBER=v_iban, SWIFT_CODE=v_swift, UK_NATIONAL_INSURANCE_NUMBER=v_ni, MAC_ADDRESS=v_mac, IN_NREGA=v_nrega, CA_HEALTH_NUMBER=v_cahn,
    IP_ADDRESS=v_ip, DATE=v_date, WALLET_ID=v_wallet, MEDICAL_RECORD_NUMBER=v_mrn, HEALTH_PLAN_BENEFICIARY_NUMBER=v_hpbn, CERTIFICATE_LICENSE_NUMBER=v_cert, DEVICE_IDENTIFIER=v_device,
    EMAIL=v_email, UPI_ID=v_upi, URL=v_url, AGE=v_age)
def is_valid(entity, value): return VALID[entity](value)

