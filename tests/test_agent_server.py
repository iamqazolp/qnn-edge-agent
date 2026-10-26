import json
import threading
import urllib.request

from qnn_edge_agent.agent.server import EdgeConsoleService, create_http_server


def test_edge_console_http_endpoints_and_fault_recovery(tmp_path):
    svc = EdgeConsoleService(workspace_dir=tmp_path)
    server = create_http_server(host="127.0.0.1", port=0, service=svc)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    try:
        # 1. GET / serves mobile HTML console
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/") as resp:
            html = resp.read().decode("utf-8")
            assert resp.status == 200
            assert "qnn-edge-agent Live Console" in html

        # 2. GET /api/state returns presets and initial sandbox state
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/api/state") as resp:
            st = json.loads(resp.read().decode("utf-8"))
            assert len(st["presets"]) == 6

        # 3. POST /api/run with corrupted_json fault recovers via Tier 2 grammar retry
        req = urllib.request.Request(
            f"http://127.0.0.1:{port}/api/run",
            data=json.dumps({
                "prompt": "Drive GPIO pin 16 HIGH in output mode to engage the cooling relay.",
                "simulate_fault": "corrupted_json",
            }).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req) as resp:
            out = json.loads(resp.read().decode("utf-8"))
            assert out["turn"]["ok"] is True
            assert out["turn"]["path_taken"] == "primary_grammar_retry"
            assert out["state"]["gpio_pins"]["16"]["state"] == "HIGH"
    finally:
        server.shutdown()
        server.server_close()
