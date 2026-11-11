"""Vercel Python function: POST /api/reconstruct

* JSON body {"object": "laptop", "noise_mm": 1, "camera_motion": 0, "seed": 0} → demo reconstruction
* application/octet-stream body (.npz tracks), optional ?hint=... → your capture
"""
import json
import os
import sys
from http.server import BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from objectmodel.service import MAX_UPLOAD_BYTES, BadRequest, health, reconstruct_demo, reconstruct_upload  # noqa: E402


class handler(BaseHTTPRequestHandler):
    def _send(self, code: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        self._send(200, health())

    def do_POST(self) -> None:
        try:
            length = int(self.headers.get("Content-Length") or 0)
            if length > MAX_UPLOAD_BYTES:
                raise BadRequest("request body too large")
            body = self.rfile.read(length) if length else b""
            params = {k: v[0] for k, v in parse_qs(urlparse(self.path).query).items()}
            ctype = (self.headers.get("Content-Type") or "").split(";")[0].strip()
            if ctype == "application/json":
                try:
                    data = json.loads(body or b"{}")
                except json.JSONDecodeError:
                    raise BadRequest("body is not valid JSON")
                if not isinstance(data, dict):
                    raise BadRequest("JSON body must be an object")
                self._send(200, reconstruct_demo(data))
            else:
                self._send(200, reconstruct_upload(body, params))
        except BadRequest as exc:
            self._send(400, {"error": str(exc)})
        except ValueError as exc:  # pipeline refusals (e.g. only one rigid part found)
            self._send(422, {"error": str(exc)})
        except Exception:  # never leak internals
            self._send(500, {"error": "internal error while reconstructing"})
