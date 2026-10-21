import copy
import pytest
import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from qnn_edge_agent.quant.calibrate import calibrate_model
from qnn_edge_agent.quant.fake_quant import apply_recipe
from qnn_edge_agent.types import LayerQuant, QuantFormat, Recipe


@pytest.fixture
def tiny_qwen_config():
    return Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=1000,
    )


@pytest.fixture
def tiny_model(tiny_qwen_config):
    torch.manual_seed(42)
    return Qwen2ForCausalLM(tiny_qwen_config)


@pytest.fixture
def dummy_tokenizer():
    class SimpleTokenizer:
        def __init__(self):
            self.pad_token = "<|pad|>"
            self.eos_token = "<|endoftext|>"

        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
            return "\n".join(m.get("content", "") for m in messages)

        def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
            if isinstance(texts, str):
                texts = [texts]
            seq_len = 16
            input_ids = torch.randint(0, 1000, (len(texts), seq_len))
            attention_mask = torch.ones_like(input_ids)
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    return SimpleTokenizer()


def make_recipe(
    recipe_id: str,
    weight_bits: int | None,
    act_bits: int | None,
) -> Recipe:
    w_fmt = (
        QuantFormat(bits=weight_bits, symmetric=True, granularity="per_channel")
        if weight_bits is not None
        else None
    )
    a_fmt = (
        QuantFormat(bits=act_bits, symmetric=False, granularity="per_tensor")
        if act_bits is not None
        else None
    )
    return Recipe(
        id=recipe_id,
        smooth="none",
        alpha=None,
        default=LayerQuant(weight=w_fmt, act=a_fmt, adaround=False),
        allocator="uniform",
        bit_budget=None,
        clip_search=False,
        kv_bits=16,
    )


def test_calibrate_tiny_model(tiny_model, dummy_tokenizer):
    calib_items = [
        {"messages": [{"role": "user", "content": "Hello edge agent"}]},
        {"messages": [{"role": "user", "content": "Set GPIO pin 12 HIGH"}]},
    ]
    stats = calibrate_model(tiny_model, dummy_tokenizer, calib_items, max_length=32, batch_size=2)

    assert len(stats.act_scales) == 14
    for name, scale in stats.act_scales.items():
        assert scale.ndim == 1
        assert (scale > 0).all()

    assert len(stats.group_ranges) == 8
    assert len(stats.act_ranges) == 14
    for name, (rmin, rmax) in stats.act_ranges.items():
        assert rmin <= 0.0 <= rmax
        assert rmax > rmin


def test_recipes_on_tiny_model(tiny_qwen_config, dummy_tokenizer):
    torch.manual_seed(42)
    base_model = Qwen2ForCausalLM(tiny_qwen_config)
    calib_items = [
        {"messages": [{"role": "user", "content": "Test prompt"}]},
    ]
    stats = calibrate_model(base_model, dummy_tokenizer, calib_items, max_length=16, batch_size=1)

    recipes = {
        "E0": make_recipe("E0", None, None),
        "E1": make_recipe("E1", 8, 8),
        "E2": make_recipe("E2", 4, 8),
        "E2a": make_recipe("E2a", 4, 16),
        "E2b": make_recipe("E2b", None, 8),
    }

    dummy_input = torch.randint(0, 1000, (1, 8))

    for r_id, recipe in recipes.items():
        model_copy = copy.deepcopy(base_model)
        orig_w = model_copy.model.layers[0].self_attn.q_proj.weight.clone()

        res = apply_recipe(model_copy, recipe, calib_stats=stats)

        # Forward pass runs cleanly
        with torch.no_grad():
            out = model_copy(dummy_input)
            assert out.logits.shape == (1, 8, 1000)
            assert not torch.isnan(out.logits).any()

        # Generate runs cleanly
        gen_tokens = model_copy.generate(dummy_input, max_new_tokens=4, do_sample=False)
        assert gen_tokens.shape == (1, 12)

        new_w = model_copy.model.layers[0].self_attn.q_proj.weight

        if r_id == "E0":
            assert len(res["hook_handles"]) == 0
            assert torch.equal(orig_w, new_w)
            assert res["avg_bits_linear"] == 32.0
        elif r_id == "E1":
            assert len(res["hook_handles"]) == 14
            assert not torch.equal(orig_w, new_w)
            assert res["avg_bits_linear"] == 8.0
        elif r_id == "E2":
            assert len(res["hook_handles"]) == 14
            assert not torch.equal(orig_w, new_w)
            assert res["avg_bits_linear"] == 4.0
        elif r_id == "E2a":
            assert len(res["hook_handles"]) == 14
            assert not torch.equal(orig_w, new_w)
            assert res["avg_bits_linear"] == 4.0
        elif r_id == "E2b":
            assert len(res["hook_handles"]) == 14
            # Linear weights are float (unchanged), but activation hooks are attached
            assert torch.equal(orig_w, new_w)

        # Clean up hooks
        for h in res["hook_handles"]:
            h.remove()

        # Ensure model still functions after hook removal
        with torch.no_grad():
            out_after = model_copy(dummy_input)
            assert out_after.logits.shape == (1, 8, 1000)


PAD_ID = 0


class LeftPaddingTokenizer:
    """Deterministic char-level tokenizer that left-pads with PAD_ID like HF does for decoding."""

    pad_token = "<|pad|>"
    eos_token = "<|endoftext|>"
    padding_side = "left"

    def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
        return "\n".join(m.get("content", "") for m in messages)

    def __call__(self, texts, padding=True, truncation=True, max_length=256, return_tensors="pt"):
        if isinstance(texts, str):
            texts = [texts]
        seqs = [[1 + (ord(c) % 997) for c in t][:max_length] for t in texts]
        width = max(len(s) for s in seqs)
        input_ids = [[PAD_ID] * (width - len(s)) + s for s in seqs]
        attention_mask = [[0] * (width - len(s)) + [1] * len(s) for s in seqs]
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
        }


def test_calibration_excludes_left_padding(tiny_model):
    # Make the pad embedding a single-channel spike: after RMSNorm it becomes
    # sqrt(hidden_size) = 8.0 on channel 0, far above any real-token activation.
    with torch.no_grad():
        tiny_model.model.embed_tokens.weight[PAD_ID].zero_()
        tiny_model.model.embed_tokens.weight[PAD_ID, 0] = 1.0

    calib_items = [
        {"messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"role": "user", "content": "turn on gpio pin twelve and report status"}]},
    ]
    tokenizer = LeftPaddingTokenizer()

    padded = calibrate_model(tiny_model, tokenizer, calib_items, batch_size=2)
    unpadded = calibrate_model(tiny_model, tokenizer, calib_items, batch_size=1)

    # Layer-0 q/k/v inputs are RMSNorm(embedding): position-independent, so the
    # padded batch must reproduce the per-prompt (pad-free) statistics exactly.
    name = "model.layers.0.self_attn.q_proj"
    assert padded.act_scales[name][0] < 8.0 - 1e-3
    assert torch.allclose(padded.act_scales[name], unpadded.act_scales[name], atol=1e-5)
    assert padded.group_ranges["model.layers.0.qkv"] == pytest.approx(
        unpadded.group_ranges["model.layers.0.qkv"], abs=1e-5
    )


def test_calibration_minmax_is_unclipped_superset(tiny_model):
    calib_items = [
        {"messages": [{"role": "user", "content": "read modbus register 40001"}]},
        {"messages": [{"role": "user", "content": "publish telemetry to mqtt topic sensors/temp"}]},
        {"messages": [{"role": "user", "content": "scan ble beacons"}]},
    ]
    stats = calibrate_model(tiny_model, LeftPaddingTokenizer(), calib_items, batch_size=2)

    assert set(stats.group_minmax) == set(stats.group_ranges)
    assert set(stats.act_minmax) == set(stats.act_ranges)
    for key, (mse_min, mse_max) in stats.group_ranges.items():
        full_min, full_max = stats.group_minmax[key]
        assert full_min <= 0.0 <= full_max
        assert full_min <= mse_min + 1e-6
        assert full_max >= mse_max - 1e-6

    # min/max must see every valid token, not just the sampled subset: its
    # magnitude has to match the running per-channel abs-max of the same input.
    for i in range(2):
        full_min, full_max = stats.group_minmax[f"model.layers.{i}.qkv"]
        abs_max = stats.act_scales[f"model.layers.{i}.self_attn.q_proj"].max().item()
        assert max(-full_min, full_max) == pytest.approx(abs_max, rel=1e-6)

    assert stats.ranges_for_bits(8) is stats.act_ranges
    assert stats.ranges_for_bits(16) is stats.act_minmax


@pytest.mark.parametrize("as_dict", [False, True])
def test_apply_recipe_selects_ranges_by_activation_bits(tiny_qwen_config, as_dict):
    from qnn_edge_agent.quant.calibrate import CalibStats

    torch.manual_seed(0)
    base_model = Qwen2ForCausalLM(tiny_qwen_config)
    names = [
        f"model.layers.{i}.{blk}.{proj}"
        for i in range(tiny_qwen_config.num_hidden_layers)
        for blk, projs in (("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
                           ("mlp", ("gate_proj", "up_proj", "down_proj")))
        for proj in projs
    ]
    stats = CalibStats(
        act_scales={n: torch.ones(1) for n in names},
        group_ranges={},
        act_ranges={n: (-1.0, 1.0) for n in names},  # clipped 8-bit range
        act_minmax={n: (-50.0, 50.0) for n in names},  # full observed range
    )
    calib = stats.to_dict() if as_dict else stats

    x = torch.zeros(1, 1, tiny_qwen_config.hidden_size)
    x[0, 0, 0] = 40.0

    seen: dict[str, float] = {}
    for r_id, act_bits in (("E2", 8), ("E2a", 16)):
        model = copy.deepcopy(base_model)
        res = apply_recipe(model, make_recipe(r_id, 4, act_bits), calib_stats=calib)
        q_proj = model.model.layers[0].self_attn.q_proj
        captured = {}
        # Registered after apply_recipe, so it observes the fake-quantized input.
        cap = q_proj.register_forward_pre_hook(lambda m, a: captured.setdefault("x", a[0]))
        with torch.no_grad():
            q_proj(x)
        cap.remove()
        for h in res["hook_handles"]:
            h.remove()
        seen[r_id] = captured["x"][0, 0, 0].item()

    assert seen["E2"] == pytest.approx(1.0, abs=0.01)  # 8-bit clips the outlier
    assert seen["E2a"] == pytest.approx(40.0, abs=0.01)  # 16-bit preserves it
