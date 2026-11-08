import torch
import torch.nn as nn

from qnn_edge_agent.agent.reliability import (
    InPlaceW4A16FallbackGenerator,
    ReliableToolAgent,
)
from qnn_edge_agent.agent.tools import EdgeToolEnvironment
from qnn_edge_agent.cli import load_recipe_registry, main
from qnn_edge_agent.quant.calibrate import CalibStats
from qnn_edge_agent.quant.fake_quant import apply_recipe
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


def test_inplace_w4a16_fallback_promotes_hooks_with_zero_ram_overhead():
    class _TinyLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = nn.Module()
            self.self_attn.q_proj = nn.Linear(8, 8, bias=False)
            self.mlp = nn.Module()
            self.mlp.down_proj = nn.Linear(8, 8, bias=False)

    class _TinyModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.model = nn.Module()
            self.model.layers = nn.ModuleList([_TinyLayer()])

    model = _TinyModel()
    calib = CalibStats(
        act_ranges={
            "model.layers.0.self_attn.q_proj": (-1.0, 1.0),
            "model.layers.0.mlp.down_proj": (-2.0, 2.0),
        },
        act_scales={},
        group_ranges={},
        act_minmax={
            "model.layers.0.self_attn.q_proj": (-3.0, 3.0),
            "model.layers.0.mlp.down_proj": (-10.0, 10.0),
        },
    )
    registry = load_recipe_registry()
    info = apply_recipe(model, registry["E2"], calib_stats=calib)
    assert len(info["hook_handles"]) == 2

    x = torch.tensor([[5.0] * 8])
    out_a8 = model.model.layers[0].mlp.down_proj(x)

    class _DummyPrimaryGen:
        def __init__(self, m: nn.Module) -> None:
            self.model = m
            self.out_during_fallback: torch.Tensor | None = None

        def __call__(self, _messages, _tools, _mode: EvalMode) -> str:
            self.out_during_fallback = self.model.model.layers[0].mlp.down_proj(x)
            return '<tool_call>\n{"name": "gpio_set", "arguments": {"pin": 16, "state": "HIGH"}}\n</tool_call>'

    primary_gen = _DummyPrimaryGen(model)
    fb_gen = InPlaceW4A16FallbackGenerator(
        primary_generator=primary_gen,  # type: ignore[arg-type]
        hook_handles=info["hook_handles"],
        active_calib=calib,
    )
    assert fb_gen.memory_overhead_mb == 0.0

    raw = fb_gen([], [], "constrained")
    assert "gpio_set" in raw
    assert primary_gen.out_during_fallback is not None
    # During fallback, A16 uses unclipped [-10, 10] instead of A8 clipped [-2, 2]
    assert not torch.allclose(out_a8, primary_gen.out_during_fallback)

    # After fallback completes, A8 hooks are restored automatically
    out_restored = model.model.layers[0].mlp.down_proj(x)
    assert torch.allclose(out_a8, out_restored)
