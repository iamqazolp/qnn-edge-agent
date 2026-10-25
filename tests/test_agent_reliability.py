from qnn_edge_agent.agent.reliability import ReliableToolAgent
from qnn_edge_agent.agent.tools import EdgeToolEnvironment
from qnn_edge_agent.cli import main
from qnn_edge_agent.types import EvalMode


def test_reliable_agent_primary_unconstrained_pass(tmp_path):
    env = EdgeToolEnvironment(tmp_path)

    def primary_gen(_messages, _tools, _mode: EvalMode) -> str:
        return '<tool_call>\n{"name": "gpio_set", "arguments": {"pin": 16, "state": "HIGH"}}\n</tool_call>'

    agent = ReliableToolAgent(env=env, primary_generator=primary_gen)
    res = agent.run_turn(messages=[{"role": "user", "content": "Set GPIO 16 high."}])

    assert res.ok is True
    assert res.path_taken == "primary_unconstrained"
    assert len(res.attempts) == 1
    assert res.execution is not None and res.execution.ok is True
    assert env.gpio_pins[16]["state"] == "HIGH"


def test_reliable_agent_recovers_via_grammar_retry_on_corrupted_json(tmp_path):
    env = EdgeToolEnvironment(tmp_path)

    def primary_gen(_messages, _tools, mode: EvalMode) -> str:
        if mode == "unconstrained":
            return '<tool_call>\n{"name": "mqtt_publish", "arguments": {"topic": "sensors/temp", "payload": '
        return '<tool_call>\n{"name": "mqtt_publish", "arguments": {"topic": "sensors/temp", "payload": "24.1"}}\n</tool_call>'

    agent = ReliableToolAgent(env=env, primary_generator=primary_gen)
    res = agent.run_turn(messages=[{"role": "user", "content": "Publish 24.1 to sensors/temp."}])

    assert res.ok is True
    assert res.path_taken == "primary_grammar_retry"
    assert len(res.attempts) == 2
    assert res.attempts[0].json_valid is False
    assert res.attempts[1].json_valid is True
    assert res.attempts[1].schema_valid is True
    assert len(env.mqtt_messages) == 1


def test_reliable_agent_falls_back_to_w8a8_on_persistent_schema_fault(tmp_path):
    env = EdgeToolEnvironment(tmp_path, seed_workspace=True)

    def primary_w4a8(_messages, _tools, _mode: EvalMode) -> str:
        # Valid JSON syntax, but qos=99 violates the integer enum [0, 1, 2]
        return '<tool_call>\n{"name": "mqtt_publish", "arguments": {"topic": "t", "payload": "p", "qos": 99}}\n</tool_call>'

    def fallback_w8a8(_messages, _tools, _mode: EvalMode) -> str:
        return '<tool_call>\n{"name": "mqtt_publish", "arguments": {"topic": "t", "payload": "p", "qos": 1}}\n</tool_call>'

    agent = ReliableToolAgent(
        env=env,
        primary_generator=primary_w4a8,
        fallback_generator=fallback_w8a8,
    )
    res = agent.run_turn(messages=[{"role": "user", "content": "Publish p to t."}])

    assert res.ok is True
    assert res.path_taken == "fallback_w8a8"
    assert len(res.attempts) == 3
    assert res.attempts[0].schema_valid is False
    assert res.attempts[1].schema_valid is False
    assert res.attempts[2].schema_valid is True


def test_reliable_agent_reports_failed_when_all_stages_invalid(tmp_path):
    env = EdgeToolEnvironment(tmp_path)

    def broken_gen(_messages, _tools, _mode: EvalMode) -> str:
        return "not json at all"

    agent = ReliableToolAgent(env=env, primary_generator=broken_gen, fallback_generator=broken_gen)
    res = agent.run_turn(messages=[{"role": "user", "content": "Do something."}])

    assert res.ok is False
    assert res.path_taken == "failed"
    assert len(res.attempts) == 3
    assert res.execution is None


def test_cli_agent_offline_demo_recovery_paths(tmp_path, capsys):
    for fault, expected_path in [
        ("none", "primary_unconstrained"),
        ("corrupted_json", "primary_grammar_retry"),
        ("wrong_schema", "fallback_w8a8"),
    ]:
        rc = main(
            [
                "agent",
                "--item-id",
                "iot_eval_00",
                "--offline-demo",
                "--simulate-fault",
                fault,
                "--workspace-dir",
                str(tmp_path / fault),
            ]
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert f'"path_taken": "{expected_path}"' in out
