"""Access-key protected live viewer for the Verity and KnowLine tracking CSVs.

Reads the CSV files directly on every request, so new prompts appear without a restart.
Serves only tracker_view.html, /api/data and the two CSV downloads.
"""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from http.cookies import SimpleCookie
from pathlib import Path
from urllib.parse import urlparse, parse_qs
import csv
import hmac
import io
import json
import os
import secrets
import sys

ROOT = Path(__file__).resolve().parent
KEY_FILE = ROOT / "tracker_access_key"
FILES = {"verity": ROOT / "verity_v4_tracking.csv", "knowline": ROOT / "knowline_tracking.csv"}
PROMPT_FIELD = {"verity": "prompt", "knowline": "raw_prompt"}
csv.field_size_limit(sys.maxsize)


def load_key():
    if os.environ.get("TRACKER_ACCESS_KEY"):
        return os.environ["TRACKER_ACCESS_KEY"]
    if not KEY_FILE.exists():
        KEY_FILE.write_text(secrets.token_urlsafe(24) + "\n")
        KEY_FILE.chmod(0o600)
    return KEY_FILE.read_text().strip()


ACCESS_KEY = load_key()


def read_rows(model):
    path = FILES[model]
    if not path.exists():
        return []
    for _ in range(3):  # the model servers append to these files; retry on a half-written row
        try:
            with path.open(newline="", encoding="utf-8") as file:
                rows = list(csv.DictReader(file))
            break
        except (csv.Error, UnicodeDecodeError):
            rows = []
    rows.sort(key=lambda row: row.get("timestamp_utc") or "", reverse=True)
    return rows


LOGIN = """<!doctype html><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Tracker access</title><body style="font-family:system-ui;background:#0d1420;color:#edf2fa;display:grid;place-items:center;height:100vh;margin:0">
<form method="get" action="/" style="display:grid;gap:12px;width:min(340px,90vw)"><h2 style="margin:0">Model tracker</h2>
<input name="key" type="password" placeholder="Access key" autofocus style="padding:12px;border-radius:8px;border:1px solid #2a394d;background:#101925;color:#edf2fa;font:inherit">
<button style="padding:12px;border:0;border-radius:8px;background:#acc9ff;font-weight:600;font:inherit">Open</button></form>"""


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type, extra=None):
        if isinstance(body, str):
            body = body.encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        for name, value in (extra or []):
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(body)

    def authorized(self, query):
        cookie = SimpleCookie(self.headers.get("Cookie", ""))
        candidates = [query.get("key", [""])[0], self.headers.get("X-Access-Key", ""),
                      cookie["tk"].value if "tk" in cookie else ""]
        return any(c and hmac.compare_digest(c.encode(), ACCESS_KEY.encode()) for c in candidates)

    def do_GET(self):
        url = urlparse(self.path)
        query = parse_qs(url.query)
        if not self.authorized(query):
            if url.path == "/":
                return self.reply(401, LOGIN, "text/html; charset=utf-8")
            return self.reply(401, '{"detail":"Access key required"}', "application/json")
        if url.path == "/":
            page = (ROOT / "tracker_view.html").read_text(encoding="utf-8")
            extra = []
            if "key" in query:  # keep the key out of the address bar and history
                extra.append(("Set-Cookie", f"tk={ACCESS_KEY}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000"))
                return self.reply(303, "", "text/plain", extra + [("Location", "/")])
            return self.reply(200, page, "text/html; charset=utf-8")
        if url.path == "/api/data":
            model = query.get("model", [""])[0]
            if model not in FILES:
                return self.reply(400, '{"detail":"model must be verity or knowline"}', "application/json")
            rows = read_rows(model)
            return self.reply(200, json.dumps({"model": model, "count": len(rows), "rows": rows}), "application/json")
        if url.path in ("/download/verity.csv", "/download/knowline.csv"):
            model = url.path.split("/")[-1].split(".")[0]
            path = FILES[model]
            body = path.read_bytes() if path.exists() else b""
            return self.reply(200, body, "text/csv; charset=utf-8",
                              [("Content-Disposition", f'attachment; filename="{path.name}"')])
        return self.reply(404, '{"detail":"Not found"}', "application/json")

    def log_message(self, fmt, *args):  # never log the query string (it can hold the key)
        sys.stderr.write("%s %s\n" % (self.address_string(), self.requestline.split("?")[0]))


if __name__ == "__main__":
    port = int(os.environ.get("TRACKER_PORT", "8012"))
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
