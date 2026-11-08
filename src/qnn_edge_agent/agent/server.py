from __future__ import annotations

import json
import socket
import tempfile
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

from qnn_edge_agent.agent.reliability import GeneratorFn, ReliableToolAgent
from qnn_edge_agent.agent.tools import EdgeToolEnvironment, load_edge_tool_schemas
from qnn_edge_agent.env import REPO_ROOT
from qnn_edge_agent.types import EvalMode

PRESET_SCENARIOS: list[dict[str, Any]] = [
    {
        "id": "mqtt_online",
        "label": "MQTT: Node12 Online",
        "category": "iot",
        "prompt": "Publish status online to topic devices/node12/status with QoS 1 and retain enabled.",
        "tool_call": {
            "name": "mqtt_publish",
            "arguments": {"topic": "devices/node12/status", "payload": "online", "qos": 1, "retain": True},
        },
    },
    {
        "id": "gpio_relay",
        "label": "GPIO: Pin 16 HIGH",
        "category": "iot",
        "prompt": "Drive GPIO pin 16 HIGH in output mode to engage the cooling relay.",
        "tool_call": {
            "name": "gpio_set",
            "arguments": {"pin": 16, "state": "HIGH", "mode": "output"},
        },
    },
    {
        "id": "thermal_guard",
        "label": "Thermal: 68°C Guard",
        "category": "iot",
        "prompt": "Switch thermal policy to conservative with max temperature 68.0C and fan speed 75%.",
        "tool_call": {
            "name": "thermal_throttle",
            "arguments": {"profile": "conservative", "max_temp_celsius": 68.0, "fan_speed_pct": 75},
        },
    },
    {
        "id": "ble_env_scan",
        "label": "BLE: Scan Sensors",
        "category": "iot",
        "prompt": "Scan BLE advertisements for 2.5 seconds filtering service UUID 181a above -80 dBm.",
        "tool_call": {
            "name": "ble_scan",
            "arguments": {"duration_seconds": 2.5, "service_uuids": ["181a"], "rssi_threshold": -80},
        },
    },
    {
        "id": "sqlite_online",
        "label": "PC: SQLite Telemetry",
        "category": "pc",
        "prompt": "Query telemetry.db for sensor_id, metric, and value where status is online.",
        "tool_call": {
            "name": "sqlite_query",
            "arguments": {
                "db_path": "telemetry.db",
                "query": "SELECT sensor_id, metric, value FROM sensors WHERE status = ?",
                "params": ["online"],
            },
        },
    },
    {
        "id": "workspace_search",
        "label": "PC: Search *.py",
        "category": "pc",
        "prompt": "Search the local workspace src directory for Python source files (*.py).",
        "tool_call": {
            "name": "local_file_search",
            "arguments": {"query": "*.py", "directory": "src", "max_results": 10},
        },
    },
]


def _match_preset_for_prompt(prompt: str) -> dict[str, Any]:
    p_low = prompt.lower()
    for preset in PRESET_SCENARIOS:
        if preset["prompt"].lower() == p_low:
            return preset
    if "gpio" in p_low or "pin" in p_low:
        return PRESET_SCENARIOS[1]
    if "thermal" in p_low or "fan" in p_low or "temp" in p_low:
        return PRESET_SCENARIOS[2]
    if "ble" in p_low or "bluetooth" in p_low:
        return PRESET_SCENARIOS[3]
    if "sqlite" in p_low or "select" in p_low or "db" in p_low:
        return PRESET_SCENARIOS[4]
    if "search" in p_low or "file" in p_low or ".py" in p_low:
        return PRESET_SCENARIOS[5]
    return PRESET_SCENARIOS[0]


MOBILE_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1, maximum-scale=1"/>
<title>qnn-edge-agent • Edge Console</title>
<style>
  :root {
    --bg: #0b1120;
    --panel: #151f32;
    --border: #263552;
    --text: #f8fafc;
    --sub: #94a3b8;
    --accent: #38bdf8;
    --green: #22c55e;
    --amber: #f59e0b;
    --red: #f43f5e;
    --purple: #a855f7;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0; padding: 14px;
    background: var(--bg); color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    max-width: 680px; margin-left: auto; margin-right: auto;
  }
  .header {
    background: linear-gradient(135deg, #162238 0%, #0f172a 100%);
    border: 1px solid var(--border); border-radius: 14px;
    padding: 14px 16px; margin-bottom: 12px;
  }
  .badge-row { display: flex; gap: 6px; flex-wrap: wrap; margin-bottom: 6px; }
  .badge {
    font-size: 11px; font-weight: 700; padding: 3px 8px;
    border-radius: 999px; background: #1e293b; color: var(--accent);
    border: 1px solid #334155;
  }
  .badge.npu { color: #86efac; border-color: #166534; background: #052e16; }
  h1 { font-size: 18px; margin: 0 0 4px 0; }
  .sub { font-size: 12px; color: var(--sub); margin: 0; }
  .card {
    background: var(--panel); border: 1px solid var(--border);
    border-radius: 12px; padding: 12px 14px; margin-bottom: 12px;
  }
  .card-title {
    font-size: 12px; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.04em; color: var(--sub); margin-bottom: 8px;
  }
  .pill-row {
    display: flex; gap: 6px; overflow-x: auto; padding-bottom: 4px;
  }
  .pill {
    flex: 0 0 auto; background: #1e293b; color: var(--text);
    border: 1px solid #334155; border-radius: 8px;
    padding: 7px 10px; font-size: 12px; font-weight: 600; cursor: pointer;
  }
  .pill:active, .pill.active { border-color: var(--accent); background: #0c4a6e; }
  .fault-row { display: grid; grid-template-columns: repeat(3, 1fr); gap: 6px; margin-top: 8px; }
  .fault-btn {
    padding: 8px 6px; border-radius: 8px; font-size: 11.5px; font-weight: 700;
    border: 1px solid #334155; background: #1e293b; color: var(--sub);
    cursor: pointer; text-align: center;
  }
  .fault-btn.active.none { background: #052e16; color: #86efac; border-color: var(--green); }
  .fault-btn.active.corrupted_json { background: #451a03; color: #fcd34d; border-color: var(--amber); }
  .fault-btn.active.wrong_schema { background: #4c0519; color: #fda4af; border-color: var(--red); }
  .prompt-box {
    width: 100%; background: #0f172a; color: var(--text);
    border: 1px solid #334155; border-radius: 8px; padding: 9px 10px;
    font-size: 13px; margin-top: 8px; resize: none;
  }
  .run-btn {
    width: 100%; margin-top: 8px; padding: 11px;
    border-radius: 9px; border: none; background: var(--accent);
    color: #082f49; font-size: 14px; font-weight: 800; cursor: pointer;
  }
  .attempt {
    background: #0f172a; border-left: 4px solid var(--green);
    border-radius: 8px; padding: 9px 10px; margin-bottom: 8px; font-size: 12px;
  }
  .attempt.fail { border-left-color: var(--red); }
  .attempt-hdr { display: flex; justify-content: space-between; font-weight: 700; margin-bottom: 4px; }
  pre {
    margin: 4px 0 0 0; padding: 6px 8px; background: #020617;
    border-radius: 6px; overflow-x: auto; font-size: 11px; color: #cbd5e1;
  }
  .err { color: #fda4af; font-size: 11px; margin-top: 4px; font-weight: 600; }
  .state-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; }
  .stat-box { background: #0f172a; border: 1px solid #1e293b; border-radius: 8px; padding: 8px 10px; }
  .stat-label { font-size: 11px; color: var(--sub); }
  .stat-val { font-size: 13px; font-weight: 700; margin-top: 3px; }
  .led-on { color: var(--green); }
</style>
</head>
<body>
  <div class="header">
    <div class="badge-row">
      <span class="badge">Qwen2.5-0.5B • E4s (W4A8 SmoothQuant)</span>
      <span class="badge npu">Hexagon v75 • 0.945ms / 44.3dB</span>
    </div>
    <h1>qnn-edge-agent Live Console</h1>
    <p class="sub">3-Tier Runtime Reliability (Unconstrained → CFG Grammar Retry → W8A8 Fallback)</p>
  </div>

  <div class="card">
    <div class="card-title">1. Tap a PC / IoT Edge Scenario</div>
    <div class="pill-row" id="presets"></div>
    <textarea id="promptInput" class="prompt-box" rows="2"></textarea>

    <div class="card-title" style="margin-top:10px;">2. Quantization Fault Injection</div>
    <div class="fault-row">
      <button class="fault-btn active none" onclick="setFault('none')">Healthy E4s<br/>(Tier 1 Pass)</button>
      <button class="fault-btn corrupted_json" onclick="setFault('corrupted_json')">Truncated JSON<br/>(Tier 2 CFG Fix)</button>
      <button class="fault-btn wrong_schema" onclick="setFault('wrong_schema')">Bad Enum/Schema<br/>(Tier 3 W8A8)</button>
    </div>

    <button class="run-btn" onclick="runAgent()">Execute Agent Turn</button>
  </div>

  <div class="card" id="traceCard" style="display:none;">
    <div class="card-title" id="traceTitle">3-Tier Recovery Trace</div>
    <div id="attemptsList"></div>
  </div>

  <div class="card">
    <div class="card-title">Live Sandboxed Environment State</div>
    <div class="state-grid" id="envState"></div>
    <div id="lastOutput"></div>
  </div>

<script>
let presets = [];
let currentFault = 'none';

function setFault(f) {
  currentFault = f;
  document.querySelectorAll('.fault-btn').forEach(b => {
    b.classList.toggle('active', b.classList.contains(f));
  });
}

async function init() {
  const res = await fetch('/api/state');
  const data = await res.json();
  presets = data.presets;
  const row = document.getElementById('presets');
  presets.forEach((p, idx) => {
    const btn = document.createElement('button');
    btn.className = 'pill' + (idx === 0 ? ' active' : '');
    btn.textContent = p.label;
    btn.onclick = () => {
      document.querySelectorAll('.pill').forEach(el => el.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('promptInput').value = p.prompt;
    };
    row.appendChild(btn);
  });
  if (presets.length > 0) {
    document.getElementById('promptInput').value = presets[0].prompt;
  }
  renderState(data.state, null);
}

async function runAgent() {
  const prompt = document.getElementById('promptInput').value;
  const res = await fetch('/api/run', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({prompt, simulate_fault: currentFault})
  });
  const data = await res.json();
  renderTrace(data.turn);
  renderState(data.state, data.turn.execution);
}

function renderTrace(turn) {
  const card = document.getElementById('traceCard');
  card.style.display = 'block';
  document.getElementById('traceTitle').textContent =
    `3-Tier Trace • Path: ${turn.path_taken} (${turn.attempts.length} attempt${turn.attempts.length>1?'s':''})`;
  const list = document.getElementById('attemptsList');
  list.innerHTML = '';
  turn.attempts.forEach((a, i) => {
    const ok = a.json_valid && a.schema_valid;
    const div = document.createElement('div');
    div.className = 'attempt' + (ok ? '' : ' fail');
    div.innerHTML = `
      <div class="attempt-hdr">
        <span>Attempt ${i+1}: ${a.stage} (${a.mode})</span>
        <span>${ok ? 'VALID' : 'FAULT'}</span>
      </div>
      <pre>${a.raw_output.replace(/</g,'&lt;')}</pre>
      ${a.validation_error ? `<div class="err">↳ ${a.validation_error}</div>` : ''}
    `;
    list.appendChild(div);
  });
}

function renderState(st, exec) {
  const grid = document.getElementById('envState');
  const pin16 = st.gpio_pins['16'] ? st.gpio_pins['16'].state : 'LOW';
  const mqttCount = st.mqtt_messages.length;
  const lastMqtt = mqttCount > 0 ? `${st.mqtt_messages[mqttCount-1].topic} (${st.mqtt_messages[mqttCount-1].payload})` : 'none';
  grid.innerHTML = `
    <div class="stat-box">
      <div class="stat-label">GPIO Pin 16 Relay</div>
      <div class="stat-val ${pin16==='HIGH'?'led-on':''}">● ${pin16}</div>
    </div>
    <div class="stat-box">
      <div class="stat-label">Thermal Policy</div>
      <div class="stat-val">${st.thermal_policy.profile} (${st.thermal_policy.max_temp_celsius}°C / ${st.thermal_policy.fan_speed_pct}%)</div>
    </div>
    <div class="stat-box" style="grid-column: span 2;">
      <div class="stat-label">MQTT Broker (${mqttCount} published)</div>
      <div class="stat-val">${lastMqtt}</div>
    </div>
  `;
  const out = document.getElementById('lastOutput');
  if (exec) {
    out.innerHTML = `<div class="card-title" style="margin-top:10px;">Last Tool Execution (${exec.tool_name} • ${exec.scenario.toUpperCase()})</div><pre>${JSON.stringify(exec.output, null, 2)}</pre>`;
  }
}

init();
</script>
</body>
</html>
"""


class EdgeConsoleService:
    """Stateful backend for the mobile/web Edge Console."""

    def __init__(
        self,
        workspace_dir: Path | None = None,
        custom_primary_gen: GeneratorFn | None = None,
        custom_fallback_gen: GeneratorFn | None = None,
    ) -> None:
        self.workspace_dir = workspace_dir or Path(tempfile.mkdtemp(prefix="qea_console_"))
        self.env = EdgeToolEnvironment(self.workspace_dir, seed_workspace=True)
        self.tools = load_edge_tool_schemas()
        self.custom_primary_gen = custom_primary_gen
        self.custom_fallback_gen = custom_fallback_gen

    def snapshot_state(self) -> dict[str, Any]:
        return {
            "workspace_dir": str(self.workspace_dir),
            "gpio_pins": {str(k): v for k, v in self.env.gpio_pins.items()},
            "mqtt_messages": self.env.mqtt_messages[-10:],
            "mqtt_retained": self.env.mqtt_retained,
            "thermal_policy": self.env.thermal_policy,
            "calendar_events": self.env.calendar_events,
        }

    def execute_prompt(self, prompt: str, simulate_fault: str = "none") -> dict[str, Any]:
        preset = _match_preset_for_prompt(prompt)
        valid_json = json.dumps(preset["tool_call"])
        tool_name = preset["tool_call"]["name"]

        def demo_primary(_m: list[dict[str, Any]], _t: list[dict[str, Any]], mode: EvalMode) -> str:
            if simulate_fault == "corrupted_json" and mode == "unconstrained":
                return f'<tool_call>\n{{"name": "{tool_name}", "arguments": {{truncated_w4a8_stream'
            if simulate_fault == "wrong_schema":
                return f'<tool_call>\n{{"name": "{tool_name}", "arguments": {{"invalid_enum_or_key": 999}}}}\n</tool_call>'
            return f"<tool_call>\n{valid_json}\n</tool_call>"

        def demo_fallback(_m: list[dict[str, Any]], _t: list[dict[str, Any]], _mode: EvalMode) -> str:
            return f"<tool_call>\n{valid_json}\n</tool_call>"

        primary_fn = self.custom_primary_gen if (self.custom_primary_gen and simulate_fault == "none") else demo_primary
        fallback_fn = self.custom_fallback_gen or demo_fallback

        agent = ReliableToolAgent(
            env=self.env,
            primary_generator=primary_fn,
            fallback_generator=fallback_fn,
            enable_grammar_retry=True,
        )
        turn = agent.run_turn(
            messages=[{"role": "user", "content": prompt}],
            tools=self.tools,
            execute_tool=True,
        )
        return {
            "turn": turn.to_dict(),
            "state": self.snapshot_state(),
        }


def get_lan_ip() -> str:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return str(s.getsockname()[0])
    except Exception:
        return "127.0.0.1"


def create_http_server(
    host: str = "0.0.0.0",
    port: int = 8080,
    service: EdgeConsoleService | None = None,
) -> ThreadingHTTPServer:
    svc = service or EdgeConsoleService()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, payload: dict[str, Any], status: int = HTTPStatus.OK) -> None:
            raw = json.dumps(payload, ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:
            if self.path in ("/", "/index.html"):
                raw = MOBILE_HTML.encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(raw)))
                self.end_headers()
                self.wfile.write(raw)
                return
            if self.path == "/api/state":
                self._send_json(
                    {
                        "presets": PRESET_SCENARIOS,
                        "state": svc.snapshot_state(),
                        "npu_summary_path": str(REPO_ROOT / "results" / "npu_probe" / "summary.json"),
                    }
                )
                return
            self.send_error(HTTPStatus.NOT_FOUND)

        def do_POST(self) -> None:
            if self.path == "/api/run":
                length = int(self.headers.get("Content-Length", "0"))
                body = json.loads(self.rfile.read(length).decode("utf-8") or "{}")
                prompt = str(body.get("prompt") or PRESET_SCENARIOS[0]["prompt"])
                fault = str(body.get("simulate_fault") or "none")
                result = svc.execute_prompt(prompt=prompt, simulate_fault=fault)
                self._send_json(result)
                return
            self.send_error(HTTPStatus.NOT_FOUND)

    return ThreadingHTTPServer((host, port), Handler)
