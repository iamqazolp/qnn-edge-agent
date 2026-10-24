from __future__ import annotations

import fnmatch
import json
import sqlite3
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qnn_edge_agent.env import REPO_ROOT
from qnn_edge_agent.eval.tool_call_eval import validate_schema_type


@dataclass(frozen=True)
class ToolExecutionResult:
    ok: bool
    tool_name: str
    scenario: str
    output: dict[str, Any]
    error: str | None = None


DEFAULT_PROCESSES: tuple[dict[str, Any], ...] = (
    {"pid": 1042, "name": "qea-OrderRouter", "cpu": 14.2, "memory": 312.5, "status": "running"},
    {"pid": 1180, "name": "mosquitto", "cpu": 2.8, "memory": 18.4, "status": "running"},
    {"pid": 1325, "name": "sqlite-wal-checkpoint", "cpu": 0.6, "memory": 12.1, "status": "sleeping"},
    {"pid": 1490, "name": "python", "cpu": 21.5, "memory": 420.0, "status": "running"},
    {"pid": 1604, "name": "ble-daemon", "cpu": 1.4, "memory": 24.0, "status": "running"},
)

DEFAULT_BLE_DEVICES: tuple[dict[str, Any], ...] = (
    {
        "address": "AA:BB:CC:11:22:01",
        "name": "EnvSensor-Node01",
        "rssi": -58,
        "service_uuids": ["181a", "180f"],
    },
    {
        "address": "AA:BB:CC:11:22:02",
        "name": "VibrationTag-A4",
        "rssi": -74,
        "service_uuids": ["181a", "fff0"],
    },
    {
        "address": "AA:BB:CC:11:22:03",
        "name": "GatewayBeacon-Far",
        "rssi": -89,
        "service_uuids": ["feaa"],
    },
)


def load_edge_tool_schemas(schema_path: Path | None = None) -> list[dict[str, Any]]:
    path = schema_path or (REPO_ROOT / "configs" / "edge_tools_schema.json")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def validate_args_against_schema(arguments: dict[str, Any], schema: dict[str, Any]) -> bool:
    if not isinstance(arguments, dict) or not isinstance(schema, dict):
        return False
    required = schema.get("required", [])
    properties = schema.get("properties", {})
    if not all(k in arguments for k in required):
        return False
    if schema.get("additionalProperties") is False and not all(k in properties for k in arguments):
        return False
    for k, v in arguments.items():
        prop_def = properties.get(k, {})
        if not validate_schema_type(v, prop_def):
            return False
        if "enum" in prop_def and v not in prop_def["enum"]:
            return False
    return True


def seed_demo_workspace(workspace_dir: Path) -> None:
    """Populate a sandbox directory with sample source files and a local SQLite DB."""
    workspace_dir.mkdir(parents=True, exist_ok=True)
    (workspace_dir / "src").mkdir(exist_ok=True)
    (workspace_dir / "configs").mkdir(exist_ok=True)

    (workspace_dir / "src" / "agent_loop.py").write_text(
        "def run_agent():\n    return 'ready'\n", encoding="utf-8"
    )
    (workspace_dir / "src" / "mqtt_client.py").write_text(
        "BROKER = 'localhost:1883'\n", encoding="utf-8"
    )
    (workspace_dir / "configs" / "gateway.yaml").write_text(
        "gateway_id: node12\nqos_default: 1\n", encoding="utf-8"
    )

    db_file = workspace_dir / "telemetry.db"
    conn = sqlite3.connect(str(db_file))
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS sensors ("
            "sensor_id TEXT PRIMARY KEY, metric TEXT, value REAL, status TEXT)"
        )
        conn.executemany(
            "INSERT OR REPLACE INTO sensors VALUES (?, ?, ?, ?)",
            [
                ("temp_01", "temperature", 23.8, "online"),
                ("hum_01", "humidity", 54.2, "online"),
                ("vib_02", "vibration", 0.12, "warning"),
            ],
        )
        conn.commit()
    finally:
        conn.close()


class EdgeToolEnvironment:
    """Sandboxed PC workspace and deterministic IoT gateway tool executor."""

    PC_TOOLS: frozenset[str] = frozenset({
        "local_file_search",
        "git_diff_status",
        "sqlite_query",
        "process_monitor",
        "calendar_schedule",
        "system_clipboard",
    })
    IOT_TOOLS: frozenset[str] = frozenset({
        "mqtt_publish",
        "gpio_set",
        "set_telemetry_threshold",
        "ble_scan",
        "thermal_throttle",
        "camera_capture",
    })

    def __init__(
        self,
        workspace_dir: Path,
        schemas: list[dict[str, Any]] | None = None,
        seed_workspace: bool = False,
    ) -> None:
        self.workspace_dir = workspace_dir.resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)
        if seed_workspace:
            seed_demo_workspace(self.workspace_dir)

        raw_schemas = schemas if schemas is not None else load_edge_tool_schemas()
        self.schemas_by_name: dict[str, dict[str, Any]] = {}
        for entry in raw_schemas:
            fn = entry.get("function", entry)
            self.schemas_by_name[fn["name"]] = fn.get("parameters", {"type": "object"})

        # In-process state for PC & IoT simulators
        self.clipboard_text: str = ""
        self.calendar_events: list[dict[str, Any]] = []
        self.mqtt_messages: list[dict[str, Any]] = []
        self.mqtt_retained: dict[str, dict[str, Any]] = {}
        self.gpio_pins: dict[int, dict[str, Any]] = {}
        self.telemetry_thresholds: dict[tuple[str, str], dict[str, Any]] = {}
        self.thermal_policy: dict[str, Any] = {
            "profile": "balanced",
            "max_temp_celsius": 75.0,
            "fan_speed_pct": 40,
        }
        self.camera_captures: list[dict[str, Any]] = []

    def _resolve_sandbox_path(self, relative_path: str = ".") -> Path:
        candidate = (self.workspace_dir / relative_path).resolve()
        try:
            candidate.relative_to(self.workspace_dir)
        except ValueError as exc:
            raise PermissionError(
                f"Path '{relative_path}' escapes sandbox root '{self.workspace_dir}'"
            ) from exc
        return candidate

    def scenario_for_tool(self, tool_name: str) -> str:
        if tool_name in self.PC_TOOLS:
            return "pc"
        if tool_name in self.IOT_TOOLS:
            return "iot"
        return "unknown"

    def validate_call(self, tool_name: str, arguments: Any) -> tuple[bool, str | None]:
        if tool_name not in self.schemas_by_name:
            return False, f"Unknown tool '{tool_name}'"
        if not isinstance(arguments, dict):
            return False, "Tool arguments must be a JSON object"
        schema = self.schemas_by_name[tool_name]
        if not validate_args_against_schema(arguments, schema):
            return False, f"Arguments for '{tool_name}' failed JSON schema validation"
        return True, None

    def execute(self, tool_name: str, arguments: dict[str, Any]) -> ToolExecutionResult:
        scenario = self.scenario_for_tool(tool_name)
        valid, err = self.validate_call(tool_name, arguments)
        if not valid:
            return ToolExecutionResult(
                ok=False,
                tool_name=tool_name,
                scenario=scenario,
                output={},
                error=err,
            )

        handler = getattr(self, f"_exec_{tool_name}", None)
        if handler is None:
            return ToolExecutionResult(
                ok=False,
                tool_name=tool_name,
                scenario=scenario,
                output={},
                error=f"No executor registered for '{tool_name}'",
            )

        try:
            out = handler(arguments)
            return ToolExecutionResult(
                ok=True,
                tool_name=tool_name,
                scenario=scenario,
                output=out,
                error=None,
            )
        except Exception as exc:
            return ToolExecutionResult(
                ok=False,
                tool_name=tool_name,
                scenario=scenario,
                output={},
                error=str(exc),
            )

    def _exec_local_file_search(self, args: dict[str, Any]) -> dict[str, Any]:
        query = str(args["query"])
        base_dir = self._resolve_sandbox_path(str(args.get("directory", ".")))
        max_results = int(args.get("max_results", 20))

        matches: list[str] = []
        if base_dir.exists():
            for p in sorted(base_dir.rglob("*")):
                if not p.is_file():
                    continue
                rel = p.relative_to(self.workspace_dir).as_posix()
                if fnmatch.fnmatch(p.name, query) or fnmatch.fnmatch(rel, query) or query.lower() in p.name.lower():
                    matches.append(rel)
                    if len(matches) >= max_results:
                        break
        return {"matches": matches, "count": len(matches)}

    def _exec_git_diff_status(self, args: dict[str, Any]) -> dict[str, Any]:
        staged = bool(args.get("staged", False))
        subpath = args.get("path")
        summary_only = bool(args.get("summary_only", False))

        if subpath is not None:
            self._resolve_sandbox_path(str(subpath))

        cmd = ["git", "diff"]
        if staged:
            cmd.append("--staged")
        if summary_only:
            cmd.append("--stat")
        if subpath:
            cmd.extend(["--", str(subpath)])

        try:
            res = subprocess.run(
                cmd,
                cwd=str(self.workspace_dir),
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            if res.returncode == 0:
                return {
                    "staged": staged,
                    "summary_only": summary_only,
                    "diff": res.stdout.strip(),
                }
        except Exception:
            pass

        files = [
            p.relative_to(self.workspace_dir).as_posix()
            for p in sorted(self.workspace_dir.rglob("*"))
            if p.is_file()
        ]
        return {
            "staged": staged,
            "summary_only": summary_only,
            "tracked_files": files,
            "diff": "",
        }

    def _exec_sqlite_query(self, args: dict[str, Any]) -> dict[str, Any]:
        db_rel = str(args["db_path"])
        query = str(args["query"]).strip()
        params = list(args.get("params", []))

        first_token = query.split(None, 1)[0].upper() if query else ""
        if first_token not in ("SELECT", "PRAGMA", "EXPLAIN", "WITH"):
            raise PermissionError("sqlite_query only permits read-only SELECT/PRAGMA/EXPLAIN/WITH queries")

        db_path = self._resolve_sandbox_path(db_rel)
        if not db_path.exists():
            raise FileNotFoundError(f"Database file '{db_rel}' not found in workspace")

        uri = f"file:{db_path.as_posix()}?mode=ro"
        conn = sqlite3.connect(uri, uri=True)
        try:
            cur = conn.execute(query, params)
            cols = [d[0] for d in cur.description] if cur.description else []
            rows = [dict(zip(cols, row)) for row in cur.fetchall()]
            return {"columns": cols, "rows": rows, "row_count": len(rows)}
        finally:
            conn.close()

    def _exec_process_monitor(self, args: dict[str, Any]) -> dict[str, Any]:
        name_filter = str(args.get("name_filter", "")).lower()
        sort_by = str(args.get("sort_by", "cpu"))
        limit = int(args.get("limit", 10))

        procs = [
            dict(p)
            for p in DEFAULT_PROCESSES
            if not name_filter or name_filter in p["name"].lower()
        ]
        reverse = sort_by in ("cpu", "memory")
        procs.sort(key=lambda p: p[sort_by], reverse=reverse)
        procs = procs[: max(0, limit)]
        return {"processes": procs, "count": len(procs)}

    def _exec_calendar_schedule(self, args: dict[str, Any]) -> dict[str, Any]:
        event = {
            "event_id": f"evt_{len(self.calendar_events) + 1:03d}",
            "title": str(args["title"]),
            "start_time": str(args["start_time"]),
            "end_time": args.get("end_time"),
            "attendees": list(args.get("attendees", [])),
        }
        self.calendar_events.append(event)
        return {"scheduled": True, "event": event}

    def _exec_system_clipboard(self, args: dict[str, Any]) -> dict[str, Any]:
        action = str(args["action"])
        if action == "write":
            self.clipboard_text = str(args.get("content", ""))
            return {"action": "write", "length": len(self.clipboard_text)}
        return {"action": "read", "content": self.clipboard_text}

    def _exec_mqtt_publish(self, args: dict[str, Any]) -> dict[str, Any]:
        msg = {
            "message_id": len(self.mqtt_messages) + 1,
            "topic": str(args["topic"]),
            "payload": str(args["payload"]),
            "qos": int(args.get("qos", 0)),
            "retain": bool(args.get("retain", False)),
        }
        self.mqtt_messages.append(msg)
        if msg["retain"]:
            self.mqtt_retained[msg["topic"]] = msg
        return {"published": True, **msg}

    def _exec_gpio_set(self, args: dict[str, Any]) -> dict[str, Any]:
        pin = int(args["pin"])
        if not (0 <= pin <= 64):
            raise ValueError(f"GPIO pin must be in [0, 64], got {pin}")
        state = str(args["state"])
        mode = str(args.get("mode", "output"))
        entry = {"pin": pin, "state": state, "mode": mode}
        self.gpio_pins[pin] = entry
        return entry

    def _exec_set_telemetry_threshold(self, args: dict[str, Any]) -> dict[str, Any]:
        sensor_id = str(args["sensor_id"])
        metric = str(args["metric"])
        rule = {
            "sensor_id": sensor_id,
            "metric": metric,
            "min_value": args.get("min_value"),
            "max_value": args.get("max_value"),
            "notify_channel": args.get("notify_channel", "cloud"),
        }
        self.telemetry_thresholds[(sensor_id, metric)] = rule
        return {"configured": True, **rule}

    def _exec_ble_scan(self, args: dict[str, Any]) -> dict[str, Any]:
        duration = float(args["duration_seconds"])
        uuid_filters = {u.lower() for u in args.get("service_uuids", [])}
        rssi_min = int(args.get("rssi_threshold", -100))
        active = bool(args.get("active", True))

        found = []
        for dev in DEFAULT_BLE_DEVICES:
            if dev["rssi"] < rssi_min:
                continue
            if uuid_filters and not (uuid_filters & {u.lower() for u in dev["service_uuids"]}):
                continue
            found.append(dict(dev))
        return {
            "duration_seconds": duration,
            "active": active,
            "devices": found,
            "count": len(found),
        }

    def _exec_thermal_throttle(self, args: dict[str, Any]) -> dict[str, Any]:
        fan = int(args.get("fan_speed_pct", self.thermal_policy["fan_speed_pct"]))
        if not (0 <= fan <= 100):
            raise ValueError(f"fan_speed_pct must be in [0, 100], got {fan}")
        self.thermal_policy = {
            "profile": str(args["profile"]),
            "max_temp_celsius": float(args["max_temp_celsius"]),
            "fan_speed_pct": fan,
        }
        return dict(self.thermal_policy)

    def _exec_camera_capture(self, args: dict[str, Any]) -> dict[str, Any]:
        res = str(args["resolution"])
        fmt = str(args["format"])
        flash = bool(args.get("flash", False))
        exposure = float(args.get("exposure_comp", 0.0))
        w, h = (int(x) for x in res.split("x"))
        bpp = 2 if fmt == "raw_bayer" else 1
        frame = {
            "frame_id": f"frame_{len(self.camera_captures) + 1:04d}",
            "resolution": res,
            "format": fmt,
            "flash": flash,
            "exposure_comp": exposure,
            "bytes_estimated": w * h * bpp,
        }
        self.camera_captures.append(frame)
        return frame
