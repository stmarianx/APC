"""Loopback smoke test of real neural weights inside the practice-table API."""
import argparse
import json
import threading
from urllib.request import Request, urlopen
import torch
from apc.practice_web import create_server


def evaluate(checkpoint):
    server = create_server(0, checkpoint)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    def request(path, value=None):
        data = None if value is None else json.dumps(value).encode()
        with urlopen(Request(f"http://127.0.0.1:{server.server_port}"+path, data=data,
                             headers={"Content-Type":"application/json"}), timeout=30) as response:
            return json.load(response)
    try:
        before = request("/api/state")["state"]
        unsupported = request("/api/model-preview")
        if unsupported["status"] != "unsupported_state":
            raise ValueError("Preflop state unexpectedly accepted")
        state = request("/api/action",{"action":"raise","to_amount_bb":"2.5",
                                      "state_fingerprint":before["state_fingerprint"]})["state"]
        preview = request("/api/model-preview")
        if preview["status"] != "experimental_estimates_only" or len(preview.get("estimates",[])) != 4:
            raise ValueError("Supported lead-node preview failed")
        if preview["recommendation"] is not None or preview["activation_authorized"] is not False:
            raise ValueError("Preview incorrectly authorized play")
        if request("/api/state")["state"] != state or preview["state_fingerprint"] != state["state_fingerprint"]:
            raise ValueError("Preview mutated or mismatched table state")
        return {"passed":True,"unsupported_preflop_rejected":True,"postflop_estimates":4,
                "state_unchanged":True,"checkpoint_fingerprint":preview["checkpoint_fingerprint"],
                "elapsed_ms":preview["elapsed_ms"],"browser_verification_pending":True}
    finally:
        server.shutdown();server.server_close();worker.join(timeout=5)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    args = parser.parse_args()
    torch.set_num_threads(2)
    print(json.dumps(evaluate(args.checkpoint),indent=2))
