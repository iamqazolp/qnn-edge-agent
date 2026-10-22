import copy

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Qwen2Config, Qwen2ForCausalLM

from qnn_edge_agent.quant.calibrate import CalibStats, calibrate_model
from qnn_edge_agent.quant.fake_quant import apply_recipe
from qnn_edge_agent.quant.grids import (
    fake_quant_act_asymmetric,
    fake_quant_weight_symmetric,
    quantize_asymmetric_per_tensor,
)
from qnn_edge_agent.quant.smooth import (
    apply_smoothquant,
    compute_smooth_scale,
    compute_weight_column_max,
    search_best_alpha,
    smooth_ln_fcs,
    smooth_up_down,
)
from qnn_edge_agent.types import LayerQuant, QuantFormat, Recipe


class SimpleRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        var = x.pow(2).mean(dim=-1, keepdim=True)
        return x * torch.rsqrt(var + self.eps) * self.weight


def test_smooth_ln_fcs_and_up_down_fp64_exact():
    torch.manual_seed(7)
    dim = 32
    inter = 64

    norm = SimpleRMSNorm(dim).to(torch.float64)
    norm.weight.data.uniform_(0.5, 2.0)
    fc1 = nn.Linear(dim, inter, bias=False).to(torch.float64)
    fc2 = nn.Linear(dim, inter, bias=False).to(torch.float64)

    x = torch.randn(4, 10, dim, dtype=torch.float64)
    scale_ab = torch.linspace(0.25, 4.0, dim, dtype=torch.float64)

    with torch.no_grad():
        h_before = norm(x)
        y1_before = fc1(h_before)
        y2_before = fc2(h_before)

        smooth_ln_fcs(norm, [fc1, fc2], scale_ab)

        h_after = norm(x)
        y1_after = fc1(h_after)
        y2_after = fc2(h_after)

    assert torch.max(torch.abs(y1_before - y1_after)).item() < 1e-12
    assert torch.max(torch.abs(y2_before - y2_after)).item() < 1e-12

    # Boundary C across SwiGLU: down_proj(silu(gate_proj(x)) * up_proj(x))
    gate_proj = nn.Linear(dim, inter, bias=False).to(torch.float64)
    up_proj = nn.Linear(dim, inter, bias=False).to(torch.float64)
    down_proj = nn.Linear(inter, dim, bias=False).to(torch.float64)
    scale_c = torch.linspace(0.5, 8.0, inter, dtype=torch.float64)

    with torch.no_grad():
        swiglu_before = F.silu(gate_proj(x)) * up_proj(x)
        out_before = down_proj(swiglu_before)

        smooth_up_down(up_proj, down_proj, scale_c)

        swiglu_after = F.silu(gate_proj(x)) * up_proj(x)
        out_after = down_proj(swiglu_after)

    assert torch.max(torch.abs(swiglu_before / scale_c - swiglu_after)).item() < 1e-12
    assert torch.max(torch.abs(out_before - out_after)).item() < 1e-12


@pytest.mark.parametrize(
    "mode,alpha",
    [
        ("AB", 0.5),
        ("ABC", 0.5),
        ("ABC", "search"),
    ],
)
def test_apply_smoothquant_full_model_equivalence(mode, alpha):
    torch.manual_seed(42)
    cfg = Qwen2Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        vocab_size=500,
    )
    model_orig = Qwen2ForCausalLM(cfg).eval()
    model_smooth = copy.deepcopy(model_orig).eval()

    class DummyTok:
        pad_token = "<pad>"
        eos_token = "<eos>"
        padding_side = "left"

        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
            return "test"

        def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="pt"):
            torch.manual_seed(99)
            ids = torch.randint(1, 500, (len(texts), 12))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    calib_items = [{"messages": [{"role": "user", "content": "calibrate"}]}]
    stats = calibrate_model(model_orig, DummyTok(), calib_items, batch_size=1)

    new_stats, chosen = apply_smoothquant(model_smooth, stats, mode=mode, alpha=alpha)

    expected_boundaries = 2 * 2 if mode == "AB" else 2 * 3
    assert len(chosen) == expected_boundaries
    for val in chosen.values():
        assert 0.1 <= val <= 0.9

    torch.manual_seed(123)
    test_ids = torch.randint(1, 500, (2, 10))
    with torch.no_grad():
        logits_orig = model_orig(test_ids).logits
        logits_smooth = model_smooth(test_ids).logits

    max_diff = torch.max(torch.abs(logits_orig - logits_smooth)).item()
    assert max_diff < 1e-4
    assert set(new_stats.act_ranges) == set(stats.act_ranges)


def test_boundary_c_reduces_outlier_and_w4a8_error():
    torch.manual_seed(11)
    in_features = 128
    out_features = 64
    down = nn.Linear(in_features, out_features, bias=False)

    # Create activations where channel 0 has a 50x outlier spike (like Qwen2 down_proj).
    x = torch.randn(64, in_features) * 0.5
    x[:, 0] *= 100.0

    act_scale = x.abs().amax(dim=0)
    ch_min = x.amin(dim=0)
    ch_max = x.amax(dim=0)
    w_scale = compute_weight_column_max([down])

    with torch.no_grad():
        y_ref = x @ down.weight.t()

        # Naive W4A8 (unsmoothed)
        sc_naive, zp_naive = quantize_asymmetric_per_tensor(
            rmin=float(ch_min.min().item()),
            rmax=float(ch_max.max().item()),
            bits=8,
        )
        x_q_naive = fake_quant_act_asymmetric(x, scale=sc_naive, zp=zp_naive, bits=8)
        w_q_naive = fake_quant_weight_symmetric(down.weight, bits=4, axis=0)
        y_naive = x_q_naive @ w_q_naive.t()
        mse_naive = float(torch.mean((y_ref - y_naive) ** 2).item())

        # Smoothed W4A8 with searched alpha*
        best_alpha = search_best_alpha(
            [down], act_scale, ch_min=ch_min, ch_max=ch_max, weight_bits=4, act_bits=8
        )
        s = compute_smooth_scale(act_scale, w_scale, alpha=best_alpha)
        x_smooth = x / s.unsqueeze(0)
        w_smooth = down.weight * s.unsqueeze(0)

        assert x_smooth.abs().max().item() < x.abs().max().item() * 0.25

        sc_sm, zp_sm = quantize_asymmetric_per_tensor(
            rmin=min(0.0, float(x_smooth.min().item())),
            rmax=max(0.0, float(x_smooth.max().item())),
            bits=8,
        )
        x_q_sm = fake_quant_act_asymmetric(x_smooth, scale=sc_sm, zp=zp_sm, bits=8)
        w_q_sm = fake_quant_weight_symmetric(w_smooth, bits=4, axis=0)
        y_sm = x_q_sm @ w_q_sm.t()
        mse_sm = float(torch.mean((y_ref - y_sm) ** 2).item())

    assert mse_sm < mse_naive * 0.5


def test_apply_recipe_config_driven_and_guards():
    torch.manual_seed(0)
    cfg = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=100,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    # Non-"E0" ID with FP configuration still takes the unmodified FP path.
    fp_recipe = Recipe(
        id="custom_fp_baseline",
        smooth="none",
        alpha=None,
        default=LayerQuant(weight=None, act=None, adaround=False),
    )
    info = apply_recipe(model, fp_recipe)
    assert info["avg_bits_linear"] == 32.0
    assert len(info["hook_handles"]) == 0

    # E4s recipe runs SmoothQuant + W4A8 hooks and returns chosen_alphas.
    names = [
        f"model.layers.0.{blk}.{proj}"
        for blk, projs in (
            ("self_attn", ("q_proj", "k_proj", "v_proj", "o_proj")),
            ("mlp", ("gate_proj", "up_proj", "down_proj")),
        )
        for proj in projs
    ]
    stats = CalibStats(
        act_scales={
            n: torch.ones(64 if "down_proj" in n else 32) * 2.0 for n in names
        },
        group_ranges={},
        act_ranges={n: (-2.0, 2.0) for n in names},
        act_minmax={n: (-2.0, 2.0) for n in names},
    )
    e4s = Recipe(
        id="E4s",
        smooth="ABC",
        alpha="search",
        default=LayerQuant(
            weight=QuantFormat(bits=4, symmetric=True, granularity="per_channel"),
            act=QuantFormat(bits=8, symmetric=False, granularity="per_tensor"),
            adaround=False,
        ),
    )
    res = apply_recipe(copy.deepcopy(model), e4s, calib_stats=stats)
    assert res["avg_bits_linear"] == 4.0
    assert set(res["chosen_alphas"]) == {
        "model.layers.0.A",
        "model.layers.0.B",
        "model.layers.0.C",
    }
    for h in res["hook_handles"]:
        h.remove()

    # Unimplemented features fail loudly instead of running naive quantization.
    with pytest.raises(NotImplementedError, match="allocator"):
        apply_recipe(
            copy.deepcopy(model),
            Recipe(
                id="E5",
                smooth="none",
                alpha=None,
                default=LayerQuant(weight=None, act=None),
                allocator="sqnr_knapsack",
            ),
        )
    with pytest.raises(NotImplementedError, match="adaround"):
        apply_recipe(
            copy.deepcopy(model),
            Recipe(
                id="E5h",
                smooth="none",
                alpha=None,
                default=LayerQuant(weight=None, act=None, adaround=True),
            ),
        )
