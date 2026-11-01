"""Tests the web boundary with fake models. Run with: python3 -m unittest discover -s tests"""
import http.client
import json
import os
import sqlite3
import sys
import threading
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "prototype"))
from answer import Models  # noqa: E402
from dose import load_medicines  # noqa: E402
from pack import build, load_items  # noqa: E402
from server import make_server  # noqa: E402


class Server(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.conn = sqlite3.connect(":memory:", check_same_thread=False)
        build(cls.conn, load_items())
        cls.server = make_server(cls.conn, Models(), load_medicines(), port=0)
        cls.port = cls.server.server_address[1]
        threading.Thread(target=cls.server.serve_forever, daemon=True).start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.conn.close()

    def request(self, method, path, body=None, headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=5)
        connection.request(method, path, body, {"Content-Type": "application/json", **(headers or {})})
        response = connection.getresponse()
        data = response.read()
        result = response.status, response.headers, data
        connection.close()
        return result

    def post(self, value, **headers):
        return self.request("POST", "/api/answer", json.dumps(value), headers)

    def test_binds_only_to_this_computer(self):
        self.assertEqual(self.server.server_address[0], "127.0.0.1")

    def test_page_has_security_headers_and_banner(self):
        status, headers, body = self.request("GET", "/")
        self.assertEqual(status, 200)
        self.assertIn("default-src 'self'", headers["Content-Security-Policy"])
        self.assertIn("Yalnızca test", body.decode())

    def test_other_host_is_rejected(self):
        self.assertEqual(self.request("GET", "/", headers={"Host": "evil.example"})[0], 403)
        self.assertEqual(self.post({"mode": "ask", "text": "su"}, Host="evil.example")[0], 403)

    def test_oversized_request_is_rejected(self):
        self.assertEqual(self.post({"mode": "ask", "text": "a" * 5000})[0], 413)

    def test_invalid_requests_are_rejected(self):
        self.assertEqual(self.request("POST", "/api/answer", "{not json")[0], 400)
        self.assertEqual(self.post([1])[0], 400)
        self.assertEqual(self.post({"mode": "ask", "extra": 1})[0], 400)
        self.assertEqual(self.post({"text": "su"})[0], 400)
        self.assertEqual(self.request("POST", "/api/answer", "{}", {"Content-Type": "text/plain"})[0], 415)

    def test_unknown_path_is_404_and_files_cant_be_escaped(self):
        self.assertEqual(self.request("GET", "/../answer.py")[0], 404)
        self.assertEqual(self.request("POST", "/other", "{}")[0], 404)

    def test_dose_flow(self):
        status, _, body = self.post({"mode": "dose", "medicine": "sahteril", "age": "10", "weight_kg": "20"})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["data"]["mg"], "200.0")

    def test_question_works_with_no_models(self):
        status, _, body = self.post({"mode": "ask", "text": "içme suyu"})
        self.assertEqual(status, 200)
        result = json.loads(body)
        self.assertEqual(result["kind"], "source_only")
        self.assertEqual(result["sources"][0]["id"], "icme-suyu")


if __name__ == "__main__":
    unittest.main()
