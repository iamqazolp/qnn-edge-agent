from __future__ import annotations

import copy
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from qnn_edge_agent.quant.calibrate import CalibStats
from qnn_edge_agent.quant.grids import (
    fake_quant_act_asymmetric,
    fake_quant_weight_symmetric,
    quantize_asymmetric_per_tensor,
)

DEFAULT_ALPHA_GRID: tuple[float, ...] = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


def compute_weight_column_max(linears: list[nn.Linear]) -> torch.Tensor:
    """Per-input-channel maximum absolute weight across one or more Linear layers."""
    if not linears:
        raise ValueError("linears must be non-empty")
    col_max = linears[0].weight.detach().to(torch.float32).abs().amax(dim=0)
    for lin in linears[1:]:
        w_max = lin.weight.detach().to(torch.float32).abs().amax(dim=0)
        col_max = torch.maximum(col_max, w_max)
    return col_max


def compute_smooth_scale(
    act_scale: torch.Tensor,
    weight_scale: torch.Tensor,
    alpha: float = 0.5,
    eps: float = 1e-5,
) -> torch.Tensor:
    """Compute per-channel SmoothQuant migration scale s = act^alpha / weight^(1-alpha)."""
    if not (0.0 <= alpha <= 1.0):
        raise ValueError(f"alpha must be in [0, 1], got {alpha}")
    a = act_scale.detach().to(torch.float64).clamp(min=eps)
    w = weight_scale.detach().to(torch.float64).clamp(min=eps)
    s = (a.pow(alpha) / w.pow(1.0 - alpha)).clamp(min=eps)
    return s.to(torch.float32)


def smooth_ln_fcs(
    norm: nn.Module,
    linears: list[nn.Linear],
    scale: torch.Tensor,
) -> None:
    """Absorb s^-1 into RMSNorm/LayerNorm weights and s into downstream Linear columns."""
    with torch.no_grad():
        dev = norm.weight.device
        s64 = scale.detach().to(device=dev, dtype=torch.float64)
        norm_w = norm.weight.data.to(torch.float64) / s64
        norm.weight.data.copy_(norm_w.to(norm.weight.dtype))
        if getattr(norm, "bias", None) is not None and norm.bias is not None:
            norm_b = norm.bias.data.to(torch.float64) / s64
            norm.bias.data.copy_(norm_b.to(norm.bias.dtype))

        for lin in linears:
            s_lin = s64.to(device=lin.weight.device)
            w64 = lin.weight.data.to(torch.float64) * s_lin.unsqueeze(0)
            lin.weight.data.copy_(w64.to(lin.weight.dtype))


def smooth_up_down(
    up_proj: nn.Linear,
    down_proj: nn.Linear,
    scale: torch.Tensor,
) -> None:
    """Absorb s^-1 into SwiGLU up_proj output rows and s into down_proj input columns."""
    with torch.no_grad():
        s_up = scale.detach().to(device=up_proj.weight.device, dtype=torch.float64)
        up_w64 = up_proj.weight.data.to(torch.float64) / s_up.unsqueeze(1)
        up_proj.weight.data.copy_(up_w64.to(up_proj.weight.dtype))
        if up_proj.bias is not None:
            up_b64 = up_proj.bias.data.to(torch.float64) / s_up
            up_proj.bias.data.copy_(up_b64.to(up_proj.bias.dtype))

        s_down = scale.detach().to(device=down_proj.weight.device, dtype=torch.float64)
        down_w64 = down_proj.weight.data.to(torch.float64) * s_down.unsqueeze(0)
        down_proj.weight.data.copy_(down_w64.to(down_proj.weight.dtype))


def _build_representative_activations(
    act_scale: torch.Tensor,
    ch_min: torch.Tensor | None = None,
    ch_max: torch.Tensor | None = None,
) -> torch.Tensor:
    """Build a deterministic multi-quantile activation batch (T, in_features) for alpha search."""
    device = act_scale.device
    s_x = act_scale.to(torch.float32)
    c_max = ch_max.to(device=device, dtype=torch.float32) if ch_max is not None else s_x
    c_min = ch_min.to(device=device, dtype=torch.float32) if ch_min is not None else -s_x

    # Ladder of typical bulk token magnitudes (0.05..0.5 of peak) plus the observed extrema.
    levels = torch.tensor([0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1.0], device=device, dtype=torch.float32)
    pos_rows = levels.unsqueeze(1) * c_max.unsqueeze(0)
    neg_rows = levels.unsqueeze(1) * c_min.unsqueeze(0)
    return torch.cat([pos_rows, neg_rows], dim=0)


def search_best_alpha(
    linears: list[nn.Linear],
    act_scale: torch.Tensor,
    ch_min: torch.Tensor | None = None,
    ch_max: torch.Tensor | None = None,
    candidates: tuple[float, ...] | list[float] = DEFAULT_ALPHA_GRID,
    weight_bits: int = 4,
    act_bits: int = 8,
) -> float:
    """Grid-search alpha in candidates to minimize post-quantization L2 output error."""
    with torch.no_grad():
        device = linears[0].weight.device
        a_scale = act_scale.detach().to(device=device, dtype=torch.float32)
        w_scale = compute_weight_column_max(linears).to(device=device, dtype=torch.float32)
        x_rep = _build_representative_activations(a_scale, ch_min=ch_min, ch_max=ch_max)

        weights_fp = [lin.weight.detach().to(device=device, dtype=torch.float32) for lin in linears]
        y_refs = [x_rep @ w.t() for w in weights_fp]

        best_alpha = float(candidates[0])
        best_mse = float("inf")

        for alpha in candidates:
            s = compute_smooth_scale(a_scale, w_scale, alpha=float(alpha)).to(device=device)
            x_smooth = x_rep / s.unsqueeze(0)
            rmin = min(0.0, float(x_smooth.min().item()))
            rmax = max(0.0, float(x_smooth.max().item()))
            a_sc, a_zp = quantize_asymmetric_per_tensor(rmin=rmin, rmax=rmax, bits=act_bits)
            x_q = fake_quant_act_asymmetric(x_smooth, scale=a_sc, zp=a_zp, bits=act_bits)

            total_mse = 0.0
            for w, y_ref in zip(weights_fp, y_refs):
                w_smooth = w * s.unsqueeze(0)
                w_q = fake_quant_weight_symmetric(w_smooth, bits=weight_bits, axis=0)
                y_q = x_q @ w_q.t()
                total_mse += float(torch.mean((y_ref - y_q) ** 2).item())

            if total_mse < best_mse:
                best_mse = total_mse
                best_alpha = float(alpha)

        return best_alpha


def _update_smoothed_stats(
    stats: CalibStats,
    proj_names: list[str],
    group_key: str,
    scale: torch.Tensor,
) -> None:
    """Update per-channel scales and tensor activation ranges after dividing input by scale."""
    s_cpu = scale.detach().cpu().to(torch.float32)
    ref_name = proj_names[0]

    old_scale = stats.act_scales.get(ref_name)
    old_peak = float(old_scale.max().item()) if old_scale is not None and old_scale.numel() > 0 else 1.0

    for name in proj_names:
        if name in stats.act_scales:
            stats.act_scales[name] = stats.act_scales[name].to(torch.float32) / s_cpu
        if name in stats.act_ch_min:
            stats.act_ch_min[name] = stats.act_ch_min[name].to(torch.float32) / s_cpu
        if name in stats.act_ch_max:
            stats.act_ch_max[name] = stats.act_ch_max[name].to(torch.float32) / s_cpu

    if ref_name in stats.act_ch_min and ref_name in stats.act_ch_max:
        cmin = min(0.0, float(stats.act_ch_min[ref_name].min().item()))
        cmax = max(0.0, float(stats.act_ch_max[ref_name].max().item()))
        if cmax <= cmin:
            cmin, cmax = -1.0, 1.0
        new_minmax = (cmin, cmax)
        new_range = (cmin, cmax)
    else:
        new_scale = stats.act_scales.get(ref_name)
        new_peak = float(new_scale.max().item()) if new_scale is not None and new_scale.numel() > 0 else 1.0
        ratio = (new_peak / old_peak) if old_peak > 1e-12 else 1.0

        if ref_name in stats.act_ranges:
            r0, r1 = stats.act_ranges[ref_name]
            new_range = (r0 * ratio, r1 * ratio)
        else:
            new_range = (-new_peak, new_peak)

        if ref_name in stats.act_minmax:
            m0, m1 = stats.act_minmax[ref_name]
            new_minmax = (m0 * ratio, m1 * ratio)
        else:
            new_minmax = new_range

    stats.group_ranges[group_key] = new_range
    stats.group_minmax[group_key] = new_minmax
    for name in proj_names:
        stats.act_ranges[name] = new_range
        stats.act_minmax[name] = new_minmax


def apply_smoothquant(
    model: PreTrainedModel | nn.Module,
    calib_stats: CalibStats | dict[str, Any],
    mode: str = "ABC",
    alpha: float | str | None = 0.5,
    weight_bits: int = 4,
    act_bits: int = 8,
) -> tuple[CalibStats, dict[str, float]]:
    """Apply 2-boundary (AB) or 3-boundary (ABC) SmoothQuant in-place to a Qwen2 model.

    Returns a new CalibStats with post-smoothing activation ranges and a dict of
    per-boundary alpha values used.
    """
    if isinstance(calib_stats, dict):
        stats = CalibStats.from_dict(calib_stats)
    else:
        stats = copy.deepcopy(calib_stats)

    if mode == "none" or mode is None:
        return stats, {}
    if mode not in ("AB", "ABC", "ABCV"):
        raise ValueError(f"Unsupported SmoothQuant mode: {mode!r}")

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise AttributeError("Could not find decoder layers in model.model.layers")

    do_search = alpha == "search"
    fixed_alpha = 0.5 if (alpha is None or do_search) else float(alpha)
    chosen_alphas: dict[str, float] = {}

    for i, layer in enumerate(layers):
        # Boundary A: input_layernorm -> q_proj, k_proj, v_proj
        q_name = f"model.layers.{i}.self_attn.q_proj"
        k_name = f"model.layers.{i}.self_attn.k_proj"
        v_name = f"model.layers.{i}.self_attn.v_proj"
        qkv_linears = [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj]

        if q_name in stats.act_scales and hasattr(layer, "input_layernorm"):
            act_a = stats.act_scales[q_name]
            ch_min_a = stats.act_ch_min.get(q_name)
            ch_max_a = stats.act_ch_max.get(q_name)
            alpha_a = (
                search_best_alpha(
                    qkv_linears,
                    act_a,
                    ch_min=ch_min_a,
                    ch_max=ch_max_a,
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                )
                if do_search
                else fixed_alpha
            )
            chosen_alphas[f"model.layers.{i}.A"] = alpha_a
            w_a = compute_weight_column_max(qkv_linears)
            s_a = compute_smooth_scale(act_a.to(w_a.device), w_a, alpha=alpha_a)
            smooth_ln_fcs(layer.input_layernorm, qkv_linears, s_a)
            _update_smoothed_stats(stats, [q_name, k_name, v_name], f"model.layers.{i}.qkv", s_a)

        # Boundary B: post_attention_layernorm -> gate_proj, up_proj
        gate_name = f"model.layers.{i}.mlp.gate_proj"
        up_name = f"model.layers.{i}.mlp.up_proj"
        gu_linears = [layer.mlp.gate_proj, layer.mlp.up_proj]

        if gate_name in stats.act_scales and hasattr(layer, "post_attention_layernorm"):
            act_b = stats.act_scales[gate_name]
            ch_min_b = stats.act_ch_min.get(gate_name)
            ch_max_b = stats.act_ch_max.get(gate_name)
            alpha_b = (
                search_best_alpha(
                    gu_linears,
                    act_b,
                    ch_min=ch_min_b,
                    ch_max=ch_max_b,
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                )
                if do_search
                else fixed_alpha
            )
            chosen_alphas[f"model.layers.{i}.B"] = alpha_b
            w_b = compute_weight_column_max(gu_linears)
            s_b = compute_smooth_scale(act_b.to(w_b.device), w_b, alpha=alpha_b)
            smooth_ln_fcs(layer.post_attention_layernorm, gu_linears, s_b)
            _update_smoothed_stats(stats, [gate_name, up_name], f"model.layers.{i}.gate_up", s_b)

        # Boundary C: mlp.up_proj -> mlp.down_proj (across SwiGLU)
        down_name = f"model.layers.{i}.mlp.down_proj"
        if mode in ("ABC", "ABCV") and down_name in stats.act_scales:
            down_lin = [layer.mlp.down_proj]
            act_c = stats.act_scales[down_name]
            ch_min_c = stats.act_ch_min.get(down_name)
            ch_max_c = stats.act_ch_max.get(down_name)
            alpha_c = (
                search_best_alpha(
                    down_lin,
                    act_c,
                    ch_min=ch_min_c,
                    ch_max=ch_max_c,
                    weight_bits=weight_bits,
                    act_bits=act_bits,
                )
                if do_search
                else fixed_alpha
            )
            chosen_alphas[f"model.layers.{i}.C"] = alpha_c
            w_c = compute_weight_column_max(down_lin)
            s_c = compute_smooth_scale(act_c.to(w_c.device), w_c, alpha=alpha_c)
            smooth_up_down(layer.mlp.up_proj, layer.mlp.down_proj, s_c)
            _update_smoothed_stats(stats, [down_name], f"model.layers.{i}.down", s_c)

    return stats, chosen_alphas
