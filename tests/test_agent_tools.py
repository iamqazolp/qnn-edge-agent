import json

from qnn_edge_agent.agent.tools import EdgeToolEnvironment
from qnn_edge_agent.env import REPO_ROOT


def test_pc_workspace_tools_and_sandbox_isolation(tmp_path):
    env = EdgeToolEnvironment(tmp_path, seed_workspace=True)

    # 1. local_file_search
    r_search = env.execute("local_file_search", {"query": "*.py", "directory": "src"})
    assert r_search.ok is True
    assert r_search.scenario == "pc"
    assert "src/agent_loop.py" in r_search.output["matches"]

    # Path traversal outside sandbox is rejected
    r_escape = env.execute("local_file_search", {"query": "*", "directory": "../.."})
    assert r_escape.ok is False
    assert "escapes sandbox" in (r_escape.error or "")

    # 2. git_diff_status
    r_git = env.execute("git_diff_status", {"staged": False, "summary_only": True})
    assert r_git.ok is True

    # 3. sqlite_query (read-only allowed, write rejected)
    r_sql = env.execute(
        "sqlite_query",
        {
            "db_path": "telemetry.db",
            "query": "SELECT sensor_id, value FROM sensors WHERE status = ?",
            "params": ["online"],
        },
    )
    assert r_sql.ok is True
    assert r_sql.output["row_count"] == 2

    r_drop = env.execute(
        "sqlite_query",
        {"db_path": "telemetry.db", "query": "DROP TABLE sensors"},
    )
    assert r_drop.ok is False
    assert "read-only" in (r_drop.error or "")

    # 4. process_monitor
    r_proc = env.execute("process_monitor", {"name_filter": "python", "sort_by": "cpu", "limit": 3})
    assert r_proc.ok is True
    assert r_proc.output["count"] == 1

    # 5. calendar_schedule
    r_cal = env.execute(
        "calendar_schedule",
        {"title": "Edge Gate Review", "start_time": "2026-10-26T10:00:00Z"},
    )
    assert r_cal.ok is True
    assert len(env.calendar_events) == 1

    # 6. system_clipboard
    assert env.execute("system_clipboard", {"action": "write", "content": "pin=16"}).ok is True
    r_clip = env.execute("system_clipboard", {"action": "read"})
    assert r_clip.ok is True
    assert r_clip.output["content"] == "pin=16"


def test_iot_gateway_simulators_and_schema_validation(tmp_path):
    env = EdgeToolEnvironment(tmp_path)

    # 7. mqtt_publish
    r_mqtt = env.execute(
        "mqtt_publish",
        {"topic": "telemetry/env", "payload": '{"temp": 23.8}', "qos": 2, "retain": True},
    )
    assert r_mqtt.ok is True
    assert r_mqtt.scenario == "iot"
    assert "telemetry/env" in env.mqtt_retained

    # Invalid enum for qos is rejected by schema validation before execution
    r_bad_qos = env.execute(
        "mqtt_publish",
        {"topic": "telemetry/env", "payload": "x", "qos": 9},
    )
    assert r_bad_qos.ok is False

    # 8. gpio_set
    r_gpio = env.execute("gpio_set", {"pin": 16, "state": "HIGH", "mode": "output"})
    assert r_gpio.ok is True
    assert env.gpio_pins[16]["state"] == "HIGH"

    # 9. set_telemetry_threshold
    r_thr = env.execute(
        "set_telemetry_threshold",
        {
            "sensor_id": "temp_01",
            "metric": "temperature",
            "min_value": 5.0,
            "max_value": 45.0,
            "notify_channel": "local_buzzer",
        },
    )
    assert r_thr.ok is True
    assert ("temp_01", "temperature") in env.telemetry_thresholds

    # 10. ble_scan
    r_ble = env.execute(
        "ble_scan",
        {"duration_seconds": 2.5, "service_uuids": ["181a"], "rssi_threshold": -80},
    )
    assert r_ble.ok is True
    assert r_ble.output["count"] == 2

    # 11. thermal_throttle
    r_therm = env.execute(
        "thermal_throttle",
        {"profile": "conservative", "max_temp_celsius": 68.0, "fan_speed_pct": 70},
    )
    assert r_therm.ok is True
    assert env.thermal_policy["profile"] == "conservative"

    # 12. camera_capture
    r_cam = env.execute(
        "camera_capture",
        {"resolution": "1280x720", "format": "jpeg", "flash": True, "exposure_comp": -0.5},
    )
    assert r_cam.ok is True
    assert r_cam.output["bytes_estimated"] == 1280 * 720


def test_all_40_custom_eval_ground_truths_validate_cleanly(tmp_path):
    env = EdgeToolEnvironment(tmp_path, seed_workspace=True)
    iot_data = json.loads((REPO_ROOT / "data" / "iot_items.json").read_text(encoding="utf-8"))
    eval_items = iot_data["eval"]
    assert len(eval_items) == 40

    for item in eval_items:
        gt = item["ground_truth"]
        valid, err = env.validate_call(gt["name"], gt["args"])
        assert valid is True, f"{item['id']} failed validation: {err}"
