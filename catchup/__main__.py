"""HTTP sidecar: python3 -m catchup --listen 127.0.0.1:18900 --vllm http://127.0.0.1:18888/v1"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

from .service import CatchupService


def _json(handler: BaseHTTPRequestHandler, status: int, payload: dict | list) -> None:
    blob = json.dumps(payload).encode("utf-8")
    handler.send_response(status)
    handler.send_header("content-type", "application/json")
    handler.send_header("content-length", str(len(blob)))
    handler.end_headers()
    handler.wfile.write(blob)


def build_handler(service: CatchupService) -> type[BaseHTTPRequestHandler]:
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            return

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            if parsed.path == "/v1/health":
                return _json(self, 200, {"ok": True, "vllm": service.vllm_url})
            if parsed.path == "/v1/status":
                query = parse_qs(parsed.query)
                session_id = (query.get("session_id") or [""])[0]
                return _json(self, 200, service.get(session_id))
            return _json(self, 404, {"error": "not found"})

        def do_POST(self) -> None:  # noqa: N802
            if urlparse(self.path).path != "/v1/snapshot":
                return _json(self, 404, {"error": "not found"})
            length = int(self.headers.get("content-length") or 0)
            raw = self.rfile.read(length) if length else b"{}"
            try:
                body = json.loads(raw.decode("utf-8") or "{}")
                status_body = service.submit(body)
            except ValueError as exc:
                return _json(self, 400, {"error": str(exc), "state": "error", "color": "red"})
            except Exception as exc:  # noqa: BLE001
                return _json(self, 500, {"error": str(exc), "state": "error", "color": "red"})
            code = 200 if status_body.get("state") == "warm" else 202
            return _json(self, code, status_body)

    return Handler


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="KV prefill catch-up sidecar")
    parser.add_argument("--listen", default=os.environ.get("CATCHUP_LISTEN", "127.0.0.1:18900"))
    parser.add_argument("--vllm", default=os.environ.get("CATCHUP_VLLM_URL", "http://127.0.0.1:18888/v1"))
    parser.add_argument("--model", default=os.environ.get("CATCHUP_MODEL", ""))
    parser.add_argument("--max-context", type=int, default=int(os.environ.get("CATCHUP_MAX_CONTEXT", "1000000")))
    parser.add_argument("--timeout", type=float, default=float(os.environ.get("CATCHUP_TIMEOUT_S", "1800")))
    args = parser.parse_args(argv)
    host, _, port = args.listen.partition(":")
    service = CatchupService(
        vllm_url=args.vllm,
        model=args.model,
        max_context=args.max_context,
        timeout_s=args.timeout,
    )
    server = ThreadingHTTPServer((host or "127.0.0.1", int(port or 18900)), build_handler(service))
    print(f"catchup listening on {host or '127.0.0.1'}:{port or 18900} → {args.vllm}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
