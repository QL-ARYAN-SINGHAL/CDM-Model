"""Public Verity and Kev tester: allow inference, not saved tracker data."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import ProxyHandler, Request, build_opener
import json
import re


ROOT = Path(__file__).resolve().parent
UPSTREAM = "http://127.0.0.1:8000"
OPENER = build_opener(ProxyHandler({}))


class Handler(BaseHTTPRequestHandler):
    def reply(self, status, body, content_type):
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/":
            page = (ROOT / "verity_ui.html").read_text(encoding="utf-8")
            page = re.sub(
                r'<div class="footer">.*?</div></main>',
                '<div class="footer">Use synthetic test prompts. Results are logged locally by the host.</div></main>',
                page,
                count=1,
                flags=re.S,
            )
            return self.reply(200, page.encode(), "text/html; charset=utf-8")
        if self.path in ("/health", "/kev/health"):
            try:
                with OPENER.open(UPSTREAM + self.path, timeout=10) as response:
                    health = json.load(response)
                public = ({"ready": health.get("ready"), "model": "Kev 0.8B"}
                          if self.path == "/kev/health" else
                          {"status": health.get("status"), "model_loaded": health.get("model_loaded"),
                           "model_name": health.get("model_name")})
                return self.reply(200, json.dumps(public).encode(), "application/json")
            except (URLError, ValueError):
                return self.reply(503, b'{"detail":"Model unavailable"}', "application/json")
        return self.reply(404, b'{"detail":"Not found"}', "application/json")

    def do_POST(self):
        if self.path not in ("/predict", "/kev/scan"):
            return self.reply(404, b'{"detail":"Not found"}', "application/json")
        try:
            size = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            size = 0
        if not 0 < size <= 65536:
            return self.reply(413, b'{"detail":"Invalid request size"}', "application/json")
        body = self.rfile.read(size)
        try:
            req = Request(UPSTREAM + self.path, data=body, headers={"Content-Type": "application/json"})
            with OPENER.open(req, timeout=180 if self.path == "/kev/scan" else 120) as response:
                return self.reply(response.status, response.read(), "application/json")
        except HTTPError as error:
            return self.reply(error.code, error.read(), "application/json")
        except URLError:
            return self.reply(503, b'{"detail":"Model unavailable"}', "application/json")


if __name__ == "__main__":
    ThreadingHTTPServer(("127.0.0.1", 8001), Handler).serve_forever()
