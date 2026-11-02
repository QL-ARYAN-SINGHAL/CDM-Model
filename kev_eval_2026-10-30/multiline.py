"""Two separate line-grid probes; not included in 30-case category counts."""
import json
import urllib.request
from pathlib import Path

from run import QUESTIONS, LOW, HIGH

CASES = [
    {"id": "L01", "expected_lines": [None, "banking"],
     "raw_prompt": "My bank account details follow.\nAccount number: 8731049521, IFSC HDFC0001234."},
    {"id": "L02", "expected_lines": ["personal", "banking"],
     "raw_prompt": "Reach me at maya.rao@aldenmail.net.\nCharge card 4929 1188 7765 0021, expiry 08/29."},
]

out = []
for case in CASES:
    body = {"raw_prompt": case["raw_prompt"], "questions": QUESTIONS, "low": LOW, "high": HIGH}
    request = urllib.request.Request("http://127.0.0.1:8000/kev/scan", data=json.dumps(body).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=190) as response:
        result = json.load(response)
    out.append({**case, "result": result})
    print(case["id"], [line["selected_intent"] for line in result["lines"]], flush=True)
Path(__file__).with_name("multiline_results.json").write_text(json.dumps(out, indent=2) + "\n")
