import json
import threading
import unittest
from urllib.request import Request, urlopen
from urllib.error import HTTPError
from apc.practice_web import create_server
from apc.neural.contract import validate_completed_hand_replay


class PracticeWebTests(unittest.TestCase):
    def test_full_hand_http_and_stale_action_rejection(self):
        server = create_server(0)
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        url = f"http://127.0.0.1:{server.server_port}"
        def request(path, value=None, origin=None):
            headers = {"Content-Type": "application/json"}
            if origin:
                headers["Origin"] = origin
            data = None if value is None else json.dumps(value).encode()
            with urlopen(Request(url+path,data=data,headers=headers), timeout=5) as response:
                return json.load(response)
        try:
            state = request("/api/state")["state"]
            with self.assertRaises(HTTPError) as error:
                request("/api/action", {"action":"fold","state_fingerprint":"stale"})
            self.assertEqual(error.exception.code,409)
            self.assertEqual(request("/api/state")["state"],state)
            with self.assertRaises(HTTPError) as error:
                request("/api/action", {"action":"fold","state_fingerprint":state["state_fingerprint"]},"https://example.com")
            self.assertEqual(error.exception.code,403)
            result = request("/api/action", {"action":"fold","state_fingerprint":state["state_fingerprint"]})
            self.assertTrue(result["state"]["terminal"])
            self.assertTrue(validate_completed_hand_replay(request("/api/replay"))["valid"])
            self.assertFalse(request("/api/new",{})["state"]["terminal"])
        finally:
            server.shutdown()
            server.server_close()
            worker.join(timeout=5)
