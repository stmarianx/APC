"""Local graphical full-hand virtual-chip practice, not external table control."""
import argparse
import json
import secrets
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from apc.practice_session import PracticeTableSession


def create_server(port=8780):
    holder = {"session": PracticeTableSession(secrets.randbits(32))}
    recording_session = "browser-practice-" + secrets.token_hex(12)
    lock = threading.RLock()
    assets = Path(__file__).parent / "practice_assets"

    class Handler(BaseHTTPRequestHandler):
        def reply(self, value, status=200):
            payload = json.dumps(value).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            with lock:
                if self.path == "/api/state":
                    self.reply(holder["session"].snapshot())
                elif self.path == "/api/replay":
                    try:
                        self.reply(holder["session"].completed_replay(recording_session))
                    except ValueError as error:
                        self.reply({"error": str(error)}, 409)
                elif self.path == "/":
                    payload = (assets / "index.html").read_bytes()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                else:
                    self.reply({"error": "Not found"}, 404)

        def do_POST(self):
            origin = self.headers.get("Origin")
            allowed = f"http://127.0.0.1:{self.server.server_port}"
            if origin not in (None, allowed) or self.headers.get_content_type() != "application/json":
                self.reply({"error": "Only same-origin JSON requests are allowed"}, 403)
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 16384:
                    raise ValueError("Invalid request size")
                body = json.loads(self.rfile.read(size))
                if not isinstance(body, dict):
                    raise ValueError("Expected JSON object")
                with lock:
                    session = holder["session"]
                    if self.path == "/api/action":
                        result = session.act(body["action"], body["state_fingerprint"], body.get("to_amount_bb"))
                    elif self.path == "/api/new":
                        if not session.hand.terminal:
                            raise ValueError("Finish the current hand before starting another")
                        holder["session"] = PracticeTableSession(secrets.randbits(32))
                        result = holder["session"].snapshot()
                    else:
                        self.reply({"error": "Not found"}, 404)
                        return
                self.reply(result)
            except (ValueError, KeyError, TypeError, ArithmeticError) as error:
                self.reply({"error": str(error)}, 409)

        def log_message(self, *_):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    return server


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8780)
    args = parser.parse_args()
    server = create_server(args.port)
    print(f"APC practice table: http://127.0.0.1:{server.server_port}/", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
