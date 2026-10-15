from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

QWEN_05B_LINEAR_PARAMS = 357_854_208
QWEN_05B_EMBED_PARAMS = 136_134_656
QWEN_05B_OTHER_PARAMS = 43_904


def quantize_symmetric_per_channel(
    weight: torch.Tensor | np.ndarray,
    bits: int = 8,
    axis: int = 0,
) -> tuple[torch.Tensor | np.ndarray, torch.Tensor | np.ndarray, torch.Tensor | np.ndarray]:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1

    if isinstance(weight, torch.Tensor):
        orig_device = weight.device
        orig_dtype = weight.dtype
        w = weight.to(torch.float32)

        reduce_dims = [d for d in range(w.ndim) if d != axis]
        max_val = w.abs()
        for d in sorted(reduce_dims, reverse=True):
            max_val = max_val.amax(dim=d, keepdim=True)
        scale = torch.clamp(max_val / float(qmax), min=1e-8)
        zp = torch.zeros_like(scale, dtype=torch.int32)
        w_q = torch.clamp(torch.round(w / scale), qmin, qmax).to(torch.int32)
        return w_q, scale.to(orig_device, dtype=orig_dtype), zp.to(orig_device)

    elif isinstance(weight, np.ndarray):
        w = weight.astype(np.float32)
        reduce_dims = tuple(d for d in range(w.ndim) if d != axis)
        max_val = np.max(np.abs(w), axis=reduce_dims, keepdims=True)
        scale = np.maximum(max_val / float(qmax), 1e-8).astype(weight.dtype)
        zp = np.zeros_like(scale, dtype=np.int32)
        w_q = np.clip(np.round(w / scale), qmin, qmax).astype(np.int32)
        return w_q, scale, zp

    else:
        raise TypeError(f"Unsupported weight type: {type(weight)}")


def dequantize_symmetric_per_channel(
    w_q: torch.Tensor | np.ndarray,
    scale: torch.Tensor | np.ndarray,
    axis: int = 0,
) -> torch.Tensor | np.ndarray:
    if isinstance(w_q, torch.Tensor):
        return (w_q.to(torch.float32) * scale.to(torch.float32)).to(scale.dtype)
    elif isinstance(w_q, np.ndarray):
        return (w_q.astype(np.float32) * scale.astype(np.float32)).astype(scale.dtype)
    else:
        raise TypeError(f"Unsupported quantized weight type: {type(w_q)}")


def fake_quant_weight_symmetric(
    weight: torch.Tensor | np.ndarray,
    bits: int = 8,
    axis: int = 0,
) -> torch.Tensor | np.ndarray:
    w_q, scale, _ = quantize_symmetric_per_channel(weight, bits=bits, axis=axis)
    return dequantize_symmetric_per_channel(w_q, scale, axis=axis)


def quantize_asymmetric_per_tensor(
    act: torch.Tensor | np.ndarray | None = None,
    bits: int = 8,
    rmin: float | None = None,
    rmax: float | None = None,
) -> tuple[float, int]:
    qmin = 0
    qmax = (1 << bits) - 1

    if rmin is None:
        if act is None:
            raise ValueError("Must provide either 'act' or 'rmin'")
        rmin = float(act.min().item() if isinstance(act, torch.Tensor) else act.min())

    if rmax is None:
        if act is None:
            raise ValueError("Must provide either 'act' or 'rmax'")
        rmax = float(act.max().item() if isinstance(act, torch.Tensor) else act.max())

    rmin = min(float(rmin), 0.0)
    rmax = max(float(rmax), 0.0)

    if rmax == rmin:
        return 1.0, 0

    scale = max((rmax - rmin) / float(qmax - qmin), 1e-8)
    zp = int(round(-rmin / scale))
    zp = max(qmin, min(qmax, zp))
    return float(scale), zp


def fake_quant_act_asymmetric(
    act: torch.Tensor | np.ndarray,
    scale: float,
    zp: int,
    bits: int = 8,
) -> torch.Tensor | np.ndarray:
    qmin = 0
    qmax = (1 << bits) - 1

    if isinstance(act, torch.Tensor):
        orig_dtype = act.dtype
        x = act.to(torch.float32)
        x_q = torch.clamp(torch.round(x / scale) + zp, qmin, qmax)
        x_dq = (x_q - zp) * scale
        return x_dq.to(orig_dtype)

    elif isinstance(act, np.ndarray):
        orig_dtype = act.dtype
        x = act.astype(np.float32)
        x_q = np.clip(np.round(x / scale) + zp, qmin, qmax)
        x_dq = (x_q - zp) * scale
        return x_dq.astype(orig_dtype)

    else:
        raise TypeError(f"Unsupported act type: {type(act)}")


def compute_model_footprint(
    model: Any = None,
    layer_bits_map: dict[str, int] | int | float | None = None,
    embed_bits: int = 8,
) -> tuple[float, float]:
    if model is None or not isinstance(model, nn.Module):
        linear_params = QWEN_05B_LINEAR_PARAMS
        embed_params = QWEN_05B_EMBED_PARAMS
        other_params = QWEN_05B_OTHER_PARAMS

        if layer_bits_map is None:
            avg_bits_linear = 32.0
            footprint_mb = (linear_params * 2 + embed_params * 2 + other_params * 2) / 1e6
            return avg_bits_linear, round(footprint_mb, 2)

        if isinstance(layer_bits_map, (int, float)):
            avg_bits_linear = float(layer_bits_map)
        elif isinstance(layer_bits_map, dict):
            avg_bits_linear = float(np.mean(list(layer_bits_map.values())))
        else:
            avg_bits_linear = 8.0

        if avg_bits_linear >= 16.0:
            footprint_mb = (linear_params * 2 + embed_params * 2 + other_params * 2) / 1e6
            return avg_bits_linear, round(footprint_mb, 2)

        linear_bytes = linear_params * (avg_bits_linear / 8.0)
        embed_bytes = embed_params * (embed_bits / 8.0)
        other_bytes = other_params * 2.0
        footprint_mb = (linear_bytes + embed_bytes + other_bytes) / 1e6
        return avg_bits_linear, round(footprint_mb, 2)

    decoder_linears: list[tuple[str, nn.Linear]] = []
    seen_params: set[int] = set()
    embed_params = 0
    other_params = 0

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and "layers." in name:
            decoder_linears.append((name, module))

    for name, param in model.named_parameters():
        ptr = param.data_ptr()
        if ptr in seen_params:
            continue
        seen_params.add(ptr)

        is_linear = any(
            param is mod.weight or (mod.bias is not None and param is mod.bias)
            for _, mod in decoder_linears
        )
        if is_linear:
            continue

        if "embed_tokens" in name or "lm_head" in name:
            embed_params += param.numel()
        else:
            other_params += param.numel()

    total_linear_numel = sum(mod.weight.numel() for _, mod in decoder_linears)

    if layer_bits_map is None:
        avg_bits_linear = 32.0
        total_unique = total_linear_numel + embed_params + other_params
        footprint_mb = (total_unique * 2) / 1e6
        return avg_bits_linear, round(footprint_mb, 2)

    if isinstance(layer_bits_map, (int, float)):
        avg_bits_linear = float(layer_bits_map)
        linear_bytes = total_linear_numel * (avg_bits_linear / 8.0)
    elif isinstance(layer_bits_map, dict):
        weighted_bits = sum(
            layer_bits_map.get(name, 8) * mod.weight.numel()
            for name, mod in decoder_linears
        )
        avg_bits_linear = weighted_bits / total_linear_numel if total_linear_numel > 0 else 8.0
        linear_bytes = total_linear_numel * (avg_bits_linear / 8.0)
    else:
        avg_bits_linear = 8.0
        linear_bytes = total_linear_numel * (avg_bits_linear / 8.0)

    if avg_bits_linear >= 16.0:
        total_unique = total_linear_numel + embed_params + other_params
        footprint_mb = (total_unique * 2) / 1e6
        return avg_bits_linear, round(footprint_mb, 2)

    embed_bytes = embed_params * (embed_bits / 8.0)
    other_bytes = other_params * 2.0
    footprint_mb = (linear_bytes + embed_bytes + other_bytes) / 1e6
    return avg_bits_linear, round(footprint_mb, 2)
