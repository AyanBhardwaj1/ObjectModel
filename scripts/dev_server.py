"""Local preview that mirrors the Vercel deployment: static web/ with clean URLs
plus the real /api/reconstruct function.

    python scripts/dev_server.py 8000
"""
import importlib.util
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
spec = importlib.util.spec_from_file_location("api_reconstruct", os.path.join(ROOT, "api", "reconstruct.py"))
api = importlib.util.module_from_spec(spec)
spec.loader.exec_module(api)


class Handler(SimpleHTTPRequestHandler):
    _send = api.handler._send  # the API methods below run with this handler as self

    def __init__(self, *a, **k):
        super().__init__(*a, directory=WEB, **k)

    def _api(self, method: str) -> bool:
        if self.path.split("?")[0] != "/api/reconstruct":
            return False
        getattr(api.handler, method)(self)
        return True

    def do_GET(self):
        if self._api("do_GET"):
            return
        path = self.path.split("?")[0].split("#")[0]
        if path != "/" and "." not in os.path.basename(path) and os.path.exists(os.path.join(WEB, path.lstrip("/") + ".html")):
            self.path = path + ".html"
        super().do_GET()

    def do_POST(self):
        if not self._api("do_POST"):
            self.send_error(404)


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"http://localhost:{port}")
    ThreadingHTTPServer(("127.0.0.1", port), Handler).serve_forever()
