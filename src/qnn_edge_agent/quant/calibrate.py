from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel, PreTrainedTokenizerBase

from qnn_edge_agent.quant.grids import (
    fake_quant_act_asymmetric,
    quantize_asymmetric_per_tensor,
)


@dataclass
class CalibStats:
    act_scales: dict[str, torch.Tensor]
    # 8-bit MSE-searched (possibly clipped) ranges.
    group_ranges: dict[str, tuple[float, float]]
    act_ranges: dict[str, tuple[float, float]]
    # Unclipped min/max over every valid calibration token (always includes 0).
    group_minmax: dict[str, tuple[float, float]] = field(default_factory=dict)
    act_minmax: dict[str, tuple[float, float]] = field(default_factory=dict)
    # Per-channel signed extrema over all valid tokens, used to recompute exact
    # post-SmoothQuant activation ranges after per-channel scaling.
    act_ch_min: dict[str, torch.Tensor] = field(default_factory=dict)
    act_ch_max: dict[str, torch.Tensor] = field(default_factory=dict)

    def ranges_for_bits(self, bits: int) -> dict[str, tuple[float, float]]:
        """Per-layer activation ranges appropriate for a given activation bit width.

        8-bit grids are coarse, so MSE-optimal clipping of rare outliers pays off.
        16-bit grids have 256x finer steps, so clipping only destroys outliers;
        use the full observed range instead.
        """
        if bits >= 16 and self.act_minmax:
            return self.act_minmax
        return self.act_ranges

    def to_dict(self) -> dict[str, Any]:
        return {
            "act_scales": {k: v.cpu().tolist() for k, v in self.act_scales.items()},
            "group_ranges": self.group_ranges,
            "act_ranges": self.act_ranges,
            "group_minmax": self.group_minmax,
            "act_minmax": self.act_minmax,
            "act_ch_min": {k: v.cpu().tolist() for k, v in self.act_ch_min.items()},
            "act_ch_max": {k: v.cpu().tolist() for k, v in self.act_ch_max.items()},
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CalibStats:
        return cls(
            act_scales={k: torch.tensor(v, dtype=torch.float32) for k, v in data["act_scales"].items()},
            group_ranges={k: tuple(v) for k, v in data["group_ranges"].items()},
            act_ranges={k: tuple(v) for k, v in data["act_ranges"].items()},
            group_minmax={k: tuple(v) for k, v in data.get("group_minmax", {}).items()},
            act_minmax={k: tuple(v) for k, v in data.get("act_minmax", {}).items()},
            act_ch_min={k: torch.tensor(v, dtype=torch.float32) for k, v in data.get("act_ch_min", {}).items()},
            act_ch_max={k: torch.tensor(v, dtype=torch.float32) for k, v in data.get("act_ch_max", {}).items()},
        )


def find_best_mse_range(
    samples: torch.Tensor,
    percentiles: list[float] | None = None,
    bits: int = 8,
) -> tuple[float, float]:
    if percentiles is None:
        percentiles = [0.99, 0.999, 0.9999, 0.99999, 1.0]

    flat = samples.flatten().to(torch.float32)
    if flat.numel() == 0:
        return 0.0, 1.0

    global_min = min(float(flat.min().item()), 0.0)
    global_max = max(float(flat.max().item()), 0.0)

    if samples.ndim >= 2 and flat.numel() > 50_000:
        width = samples.shape[-1]
        rows = samples.reshape(-1, width)
        max_rows = max(1, 50_000 // width)
        row_step = max(1, rows.shape[0] // max_rows)
        sub_flat = rows[::row_step][:max_rows].reshape(-1).to(torch.float32)
    elif flat.numel() > 50_000:
        step = max(1, flat.numel() // 50_000)
        sub_flat = flat[::step][:50_000]
    else:
        sub_flat = flat

    best_mse = float("inf")
    best_range = (global_min, global_max)

    for p in percentiles:
        if p < 1.0:
            cmin = min(float(torch.quantile(sub_flat, 1.0 - p).item()), 0.0)
            cmax = max(float(torch.quantile(sub_flat, p).item()), 0.0)
        else:
            cmin = global_min
            cmax = global_max

        if cmax == cmin:
            continue

        scale, zp = quantize_asymmetric_per_tensor(rmin=cmin, rmax=cmax, bits=bits)
        dq = fake_quant_act_asymmetric(sub_flat, scale=scale, zp=zp, bits=bits)
        mse = float(torch.mean((sub_flat - dq) ** 2).item())

        if mse < best_mse:
            best_mse = mse
            best_range = (cmin, cmax)

    return best_range


def calibrate_model(
    model: PreTrainedModel | nn.Module,
    tokenizer: PreTrainedTokenizerBase,
    calib_items: list[dict[str, Any]] | Path | str,
    max_length: int = 256,
    batch_size: int = 8,
    device: str | torch.device | None = None,
    max_samples_per_group: int = 500,
) -> CalibStats:
    if isinstance(calib_items, (str, Path)):
        p = Path(calib_items)
        with open(p, "r", encoding="utf-8") as f:
            items = [json.loads(line) for line in f if line.strip()]
    else:
        items = list(calib_items)

    if device is None:
        device = next(model.parameters()).device

    if hasattr(tokenizer, "pad_token") and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if hasattr(tokenizer, "padding_side"):
        tokenizer.padding_side = "left"

    layers = getattr(getattr(model, "model", model), "layers", None)
    if layers is None:
        raise AttributeError("Could not find decoder layers in model.model.layers")

    num_layers = len(layers)
    act_scales: dict[str, torch.Tensor] = {}
    act_ch_min: dict[str, torch.Tensor] = {}
    act_ch_max: dict[str, torch.Tensor] = {}
    group_samples: dict[str, list[torch.Tensor]] = {}
    group_counts: dict[str, int] = {}
    # Running min/max over all valid tokens (not just the sampled subset), seeded
    # at 0 so the asymmetric grid can always represent zero exactly.
    group_minmax: dict[str, tuple[float, float]] = {}

    for i, layer in enumerate(layers):
        for gname in ("qkv", "o", "gate_up", "down"):
            gkey = f"model.layers.{i}.{gname}"
            group_samples[gkey] = []
            group_counts[gkey] = 0
            group_minmax[gkey] = (0.0, 0.0)

        for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            mod = getattr(layer.self_attn, proj_name)
            name = f"model.layers.{i}.self_attn.{proj_name}"
            act_scales[name] = torch.zeros(mod.in_features, dtype=torch.float32, device=device)
            act_ch_min[name] = torch.zeros(mod.in_features, dtype=torch.float32, device=device)
            act_ch_max[name] = torch.zeros(mod.in_features, dtype=torch.float32, device=device)

        for proj_name in ("gate_proj", "up_proj", "down_proj"):
            mod = getattr(layer.mlp, proj_name)
            name = f"model.layers.{i}.mlp.{proj_name}"
            act_scales[name] = torch.zeros(mod.in_features, dtype=torch.float32, device=device)
            act_ch_min[name] = torch.zeros(mod.in_features, dtype=torch.float32, device=device)
            act_ch_max[name] = torch.zeros(mod.in_features, dtype=torch.float32, device=device)

    hooks = []
    # Boolean attention mask of the batch currently in flight. Hooks use it to
    # drop left-padding positions so pad-token activations never enter the stats.
    batch_state: dict[str, torch.Tensor | None] = {"valid": None}

    def valid_rows(x: torch.Tensor, in_features: int) -> torch.Tensor:
        flat = x.reshape(-1, in_features).to(torch.float32)
        valid = batch_state["valid"]
        if valid is not None and x.dim() == 3 and tuple(x.shape[:2]) == tuple(valid.shape):
            flat = flat[valid.reshape(-1)]
        return flat

    def make_pre_hook(layer_name: str, group_key: str | None = None):
        def pre_hook(module: nn.Module, args: tuple[torch.Tensor, ...]):
            if len(args) == 0:
                return args
            x = args[0].detach()
            in_features = module.in_features
            flat = valid_rows(x, in_features)
            if flat.shape[0] == 0:
                return args
            curr_max = flat.abs().max(dim=0).values
            act_scales[layer_name] = torch.maximum(act_scales[layer_name], curr_max)
            act_ch_min[layer_name] = torch.minimum(act_ch_min[layer_name], flat.min(dim=0).values)
            act_ch_max[layer_name] = torch.maximum(act_ch_max[layer_name], flat.max(dim=0).values)

            if group_key is not None:
                gmin, gmax = group_minmax[group_key]
                group_minmax[group_key] = (
                    min(gmin, float(flat.min().item())),
                    max(gmax, float(flat.max().item())),
                )

                current_count = group_counts[group_key]
                if current_count < max_samples_per_group:
                    needed = max_samples_per_group - current_count
                    to_take = flat[:needed]
                    group_samples[group_key].append(to_take.cpu())
                    group_counts[group_key] += to_take.shape[0]
            return args

        return pre_hook

    for i, layer in enumerate(layers):
        hooks.append(
            layer.self_attn.q_proj.register_forward_pre_hook(
                make_pre_hook(f"model.layers.{i}.self_attn.q_proj", f"model.layers.{i}.qkv")
            )
        )
        hooks.append(
            layer.self_attn.k_proj.register_forward_pre_hook(
                make_pre_hook(f"model.layers.{i}.self_attn.k_proj", None)
            )
        )
        hooks.append(
            layer.self_attn.v_proj.register_forward_pre_hook(
                make_pre_hook(f"model.layers.{i}.self_attn.v_proj", None)
            )
        )
        hooks.append(
            layer.self_attn.o_proj.register_forward_pre_hook(
                make_pre_hook(f"model.layers.{i}.self_attn.o_proj", f"model.layers.{i}.o")
            )
        )
        hooks.append(
            layer.mlp.gate_proj.register_forward_pre_hook(
                make_pre_hook(f"model.layers.{i}.mlp.gate_proj", f"model.layers.{i}.gate_up")
            )
        )
        hooks.append(
            layer.mlp.up_proj.register_forward_pre_hook(
                make_pre_hook(f"model.layers.{i}.mlp.up_proj", None)
            )
        )
        hooks.append(
            layer.mlp.down_proj.register_forward_pre_hook(
                make_pre_hook(f"model.layers.{i}.mlp.down_proj", f"model.layers.{i}.down")
            )
        )

    was_training = model.training
    model.eval()

    try:
        formatted_prompts: list[str] = []
        for it in items:
            msgs = it.get("messages", [])
            tools = it.get("tools")
            if hasattr(tokenizer, "apply_chat_template"):
                try:
                    text = tokenizer.apply_chat_template(
                        msgs, tools=tools, add_generation_prompt=True, tokenize=False
                    )
                except Exception:
                    text = "\n".join(m.get("content", "") for m in msgs)
            else:
                text = "\n".join(m.get("content", "") for m in msgs)
            formatted_prompts.append(text)

        with torch.no_grad():
            for start_idx in range(0, len(formatted_prompts), batch_size):
                batch_texts = formatted_prompts[start_idx : start_idx + batch_size]
                encoded = tokenizer(
                    batch_texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                if hasattr(encoded, "to"):
                    encoded = encoded.to(device)
                else:
                    encoded = {k: v.to(device) if hasattr(v, "to") else v for k, v in encoded.items()}
                attention_mask = encoded.get("attention_mask")
                batch_state["valid"] = attention_mask.bool() if attention_mask is not None else None
                model(**encoded)

    finally:
        batch_state["valid"] = None
        for h in hooks:
            h.remove()
        model.train(was_training)

    group_ranges: dict[str, tuple[float, float]] = {}
    act_ranges: dict[str, tuple[float, float]] = {}
    act_minmax: dict[str, tuple[float, float]] = {}

    group_ref_proj = {
        "qkv": "self_attn.q_proj",
        "o": "self_attn.o_proj",
        "gate_up": "mlp.gate_proj",
        "down": "mlp.down_proj",
    }

    for i in range(num_layers):
        for gname in ("qkv", "o", "gate_up", "down"):
            gkey = f"model.layers.{i}.{gname}"
            samples = group_samples[gkey]
            if samples:
                ref_proj = f"model.layers.{i}.{group_ref_proj[gname]}"
                extra_rows = []
                if ref_proj in act_ch_min and ref_proj in act_ch_max:
                    extra_rows = [
                        act_ch_min[ref_proj].cpu().unsqueeze(0),
                        act_ch_max[ref_proj].cpu().unsqueeze(0),
                    ]
                merged = torch.cat(samples + extra_rows, dim=0)
                best_range = find_best_mse_range(merged, bits=8)
            else:
                best_range = (-1.0, 1.0)
            group_ranges[gkey] = best_range
            if group_minmax[gkey][1] <= group_minmax[gkey][0]:
                group_minmax[gkey] = (-1.0, 1.0)

        for ranges, out in ((group_ranges, act_ranges), (group_minmax, act_minmax)):
            qkv_range = ranges[f"model.layers.{i}.qkv"]
            o_range = ranges[f"model.layers.{i}.o"]
            gate_up_range = ranges[f"model.layers.{i}.gate_up"]
            down_range = ranges[f"model.layers.{i}.down"]

            out[f"model.layers.{i}.self_attn.q_proj"] = qkv_range
            out[f"model.layers.{i}.self_attn.k_proj"] = qkv_range
            out[f"model.layers.{i}.self_attn.v_proj"] = qkv_range
            out[f"model.layers.{i}.self_attn.o_proj"] = o_range
            out[f"model.layers.{i}.mlp.gate_proj"] = gate_up_range
            out[f"model.layers.{i}.mlp.up_proj"] = gate_up_range
            out[f"model.layers.{i}.mlp.down_proj"] = down_range

    return CalibStats(
        act_scales={k: v.cpu() for k, v in act_scales.items()},
        group_ranges=group_ranges,
        act_ranges=act_ranges,
        group_minmax=group_minmax,
        act_minmax=act_minmax,
        act_ch_min={k: v.cpu() for k, v in act_ch_min.items()},
        act_ch_max={k: v.cpu() for k, v in act_ch_max.items()},
    )
