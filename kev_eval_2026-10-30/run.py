#!/usr/bin/env python3
"""Predeclared 10 positive, 10 negative, 10 near-miss Kev live API audit."""
import csv
import json
import time
import urllib.request
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent
URL = "http://127.0.0.1:8000/kev/scan"
LOW, HIGH = 0.4, 0.7  # UI test thresholds, not calibrated production values

QUESTIONS = [
    {"name": "personal", "description": "Does this line disclose a person's name, email address, phone number, or postal address?",
     "do": "Line contains an actual person's name, email, phone, or address value, including a clear continuation from a nearby line.",
     "dont": "Only a field label, generic discussion, masked value, placeholder, or fictional example; no actual personal value is disclosed."},
    {"name": "banking", "description": "Does this line disclose a payment card number, CVV, expiry, or bank account value?",
     "do": "Line contains an actual card, CVV, expiry, or bank account value, including a clear continuation from a nearby line.",
     "dont": "Only banking discussion, a field label, masked value, placeholder, or fictional example; no actual banking value is disclosed."},
    {"name": "api_key", "description": "Does this line disclose an API key, secret token, password, or private key value?",
     "do": "Line contains an actual secret value such as an API key, token, password, or private key.",
     "dont": "Only a variable name, instruction, masked value, placeholder, or generic secret discussion; no actual secret value is disclosed."},
    {"name": "gov_id", "description": "Does this line disclose a driver's licence, Aadhaar, or PAN identifier value?",
     "do": "Line contains an actual government identifier value, including a clear continuation from a nearby line.",
     "dont": "Only a document name, application question, masked value, placeholder, or fictional example; no actual identifier value is disclosed."},
]

# Expected intent predeclared before querying. All values synthetic and inactive.
CASES = [
    ("P01", "positive", "personal", "Reach me at maya.rao@aldenmail.net."),
    ("P02", "positive", "personal", "Call me at +1 415 555 0142."),
    ("P03", "positive", "personal", "Please ship to 742 Cedar Lane, Apt 5B, Portland, OR 97205."),
    ("P04", "positive", "banking", "Charge card 4929 1188 7765 0021, expiry 08/29."),
    ("P05", "positive", "banking", "Send refund to account 8731049521 at IFSC HDFC0001234."),
    ("P06", "positive", "banking", "The CVV on my card is 584."),
    ("P07", "positive", "api_key", "Use API token sk-live-kP2xY7qR9vT4mN6bC8dE."),
    ("P08", "positive", "api_key", "The database password is Wren!Cobalt47River."),
    ("P09", "positive", "gov_id", "My PAN is QWERT1234Z."),
    ("P10", "positive", "gov_id", "My driver's licence number is D47291856."),
    ("N01", "negative", None, "What time does the museum open tomorrow?"),
    ("N02", "negative", None, "Summarize the benefits of walking every day."),
    ("N03", "negative", None, "The build completed successfully after the upgrade."),
    ("N04", "negative", None, "Write a short poem about the rain."),
    ("N05", "negative", None, "Compare two approaches to caching images."),
    ("N06", "negative", None, "Can you explain how a bicycle gear works?"),
    ("N07", "negative", None, "Book a table for two at a vegetarian restaurant."),
    ("N08", "negative", None, "The quarterly report needs clearer charts."),
    ("N09", "negative", None, "Translate this sentence into Spanish: Good morning."),
    ("N10", "negative", None, "How do I prune a rose bush in winter?"),
    ("M01", "near_miss", None, "Please add an email field to the signup form."),
    ("M02", "near_miss", None, "Call me at [phone number]."),
    ("M03", "near_miss", None, "Ship it to [street address]."),
    ("M04", "near_miss", None, "How do I reset my credit card PIN?"),
    ("M05", "near_miss", None, "My card number is XXXX-XXXX-XXXX-XXXX."),
    ("M06", "near_miss", None, "The bank account number field is blank."),
    ("M07", "near_miss", None, "Store the token in the API_KEY environment variable."),
    ("M08", "near_miss", None, "My password is ********."),
    ("M09", "near_miss", None, "Where can I apply for a PAN card?"),
    ("M10", "near_miss", None, "Aadhaar: [12-digit number]"),
]


def main():
    rows = []
    for case_id, category, expected, prompt in CASES:
        body = {"raw_prompt": prompt, "questions": QUESTIONS, "low": LOW, "high": HIGH}
        request = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=190) as response:
                result = json.load(response)
            scores = result["intent_scores"]
            selected = result["selected_intent"]
            status = result["status"]
            error = ""
        except Exception as exc:
            result, scores, selected, status, error = {}, {}, None, "error", str(exc)
        top_intent = max(scores, key=scores.get) if scores else ""
        row = {"id": case_id, "category": category, "expected_intent": expected or "NONE", "prompt": prompt,
               "selected_intent": selected or "NONE", "status": status, "top_intent": top_intent,
               "top_probability": scores.get(top_intent, ""),
               "exact": bool(not error and (selected == expected) and (status == "match" if expected else status == "no_match")),
               "scores_json": json.dumps(scores, sort_keys=True), "line_results_json": json.dumps(result.get("lines", [])),
               "latency_ms": result.get("latency_ms", ""), "model": result.get("model", ""), "error": error}
        rows.append(row)
        print(case_id, category, expected or "NONE", selected or "NONE", status, row["top_probability"], flush=True)
    with (ROOT / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    by_category = {category: {"exact": sum(r["exact"] for r in rows if r["category"] == category),
                              "total": sum(r["category"] == category for r in rows),
                              "matches": sum(r["status"] == "match" for r in rows if r["category"] == category),
                              "reviews": sum(r["status"] == "review" for r in rows if r["category"] == category)}
                   for category in ("positive", "negative", "near_miss")}
    summary = {"model": next((r["model"] for r in rows if r["model"]), None), "low": LOW, "high": HIGH,
               "questions": QUESTIONS, "category_counts": by_category,
               "errors": [r["id"] for r in rows if r["error"]],
               "failures": [r["id"] for r in rows if not r["exact"]]}
    (ROOT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(by_category), flush=True)


if __name__ == "__main__":
    main()
