import copy
import pytest
import torch
from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

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
