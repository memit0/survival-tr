"""Local web app. Usage: python3 prototype/server.py   then open http://127.0.0.1:8080

Binds only to this computer, stores nothing, and doesn't log questions.
"""
import argparse
import json
import os
import sqlite3
import sys
import threading
from dataclasses import asdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from answer import Models, Request, answer  # noqa: E402
from dose import load_medicines  # noqa: E402
from llm import EMBED_MODELS, chat, query_embedder  # noqa: E402
from pack import DEFAULT_PATH  # noqa: E402
from search import unit  # noqa: E402

STATIC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
FILES = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "text/javascript; charset=utf-8"),
         "/style.css": ("style.css", "text/css; charset=utf-8")}
MAX_BODY = 4096
FIELDS = {"mode", "text", "medicine", "age", "weight_kg"}


def make_handler(conn, models, medicines, port_holder):
    # ponytail: one question at a time, because this Mac can't run several model calls at once.
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        server_version = "survival-tr"
        sys_version = ""

        def log_message(self, *args):
            pass  # questions can reveal health details

        def send(self, status, body, content_type="application/json; charset=utf-8"):
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Content-Security-Policy", "default-src 'self'; base-uri 'none'; frame-ancestors 'none'; form-action 'none'")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_json(self, status, value):
            self.send(status, json.dumps(value, ensure_ascii=False).encode())

        def host_allowed(self):
            # Blocks other websites from reaching this server through DNS tricks.
            allowed = {f"127.0.0.1:{port_holder[0]}", f"localhost:{port_holder[0]}"}
            if self.headers.get("Host") in allowed:
                return True
            self.send_json(403, {"error": "forbidden host"})
            return False

        def do_GET(self):
            if not self.host_allowed():
                return
            if self.path == "/api/medicines":
                self.send_json(200, [{"id": m["id"], "name": m["name"]} for m in medicines.values()])
            elif self.path in FILES:
                filename, content_type = FILES[self.path]
                with open(os.path.join(STATIC, filename), "rb") as f:
                    self.send(200, f.read(), content_type)
            else:
                self.send_json(404, {"error": "not found"})

        def do_POST(self):
            if not self.host_allowed():
                return
            if self.path != "/api/answer":
                return self.send_json(404, {"error": "not found"})
            if self.headers.get_content_type() != "application/json":
                return self.send_json(415, {"error": "JSON required"})
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                return self.send_json(411, {"error": "length required"})
            if length < 0 or length > MAX_BODY:
                self.close_connection = True
                return self.send_json(413, {"error": "request too large"})
            try:
                values = json.loads(self.rfile.read(length))
            except (ValueError, UnicodeDecodeError):
                return self.send_json(400, {"error": "invalid JSON"})
            if not isinstance(values, dict) or set(values) - FIELDS or not isinstance(values.get("mode"), str):
                return self.send_json(400, {"error": "invalid fields"})
            with lock:
                result = answer(Request(**values), conn, models, medicines)
            self.send_json(200, asdict(result))

    return Handler


def make_server(conn, models, medicines, port=8080):
    holder = [port]
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(conn, models, medicines, holder))
    holder[0] = server.server_address[1]
    return server


def real_models(embed_model):
    embed_query = query_embedder(embed_model)
    dimensions = EMBED_MODELS[embed_model]["dimensions"]
    return Models(embed=lambda text: unit(embed_query(text), dimensions),
                  chat=chat)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pack", default=DEFAULT_PATH)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    if not os.path.exists(args.pack):
        sys.exit(f"No pack at {args.pack}. Build one with: python3 prototype/pack.py")
    conn = sqlite3.connect(f"file:{args.pack}?mode=ro", uri=True, check_same_thread=False)
    server = make_server(conn, real_models("embeddinggemma"), load_medicines(), args.port)
    print(f"http://127.0.0.1:{server.server_address[1]}  (test only, not for real emergencies)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
