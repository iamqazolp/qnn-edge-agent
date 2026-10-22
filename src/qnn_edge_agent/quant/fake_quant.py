from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
from transformers import PreTrainedModel

from qnn_edge_agent.quant.calibrate import CalibStats
from qnn_edge_agent.quant.grids import (
    compute_model_footprint,
    fake_quant_act_asymmetric,
    fake_quant_weight_symmetric,
    quantize_asymmetric_per_tensor,
)
from qnn_edge_agent.quant.smooth import apply_smoothquant
from qnn_edge_agent.types import Recipe


def apply_recipe(
    model: PreTrainedModel | nn.Module,
    recipe: Recipe,
    calib_stats: CalibStats | dict[str, Any] | None = None,
    layer_bits_map: dict[str, int] | None = None,
) -> dict[str, Any]:
    hook_handles: list[torch.utils.hooks.RemovableHandle] = []

    if recipe.allocator not in ("uniform", None) and layer_bits_map is None:
        raise NotImplementedError(
            f"Recipe '{recipe.id}' requests allocator='{recipe.allocator}', "
            "which requires a resolved layer_bits_map."
        )
    if recipe.default.adaround:
        raise NotImplementedError(
            f"Recipe '{recipe.id}' requests adaround=True, which is not yet implemented."
        )
    if recipe.clip_search:
        raise NotImplementedError(
            f"Recipe '{recipe.id}' requests clip_search=True, which is not yet implemented."
        )

    is_unmodified_fp = (
        recipe.smooth in ("none", None)
        and recipe.default.weight is None
        and recipe.default.act is None
        and layer_bits_map is None
    )
    if is_unmodified_fp:
        avg_bits, footprint = compute_model_footprint(model, layer_bits_map=None)
        return {
            "avg_bits_linear": avg_bits,
            "footprint_mb": footprint,
            "hook_handles": hook_handles,
            "chosen_alphas": {},
        }

    chosen_alphas: dict[str, float] = {}
    active_calib: CalibStats | dict[str, Any] | None = calib_stats

    if recipe.smooth not in ("none", None):
        if calib_stats is None:
            raise ValueError(
                f"Recipe '{recipe.id}' enables SmoothQuant (smooth={recipe.smooth!r}) "
                "and requires calib_stats."
            )
        w_bits_hint = recipe.default.weight.bits if recipe.default.weight is not None else 4
        a_bits_hint = recipe.default.act.bits if recipe.default.act is not None else 8
        active_calib, chosen_alphas = apply_smoothquant(
            model=model,
            calib_stats=calib_stats,
            mode=recipe.smooth,
            alpha=recipe.alpha,
            weight_bits=w_bits_hint,
            act_bits=a_bits_hint,
        )

    layers = getattr(getattr(model, "model", model), "layers", None)
    decoder_linears: list[tuple[str, nn.Linear]] = []
    if layers is not None:
        for i, layer in enumerate(layers):
            for proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
                mod = getattr(layer.self_attn, proj_name, None)
                if isinstance(mod, nn.Linear):
                    decoder_linears.append((f"model.layers.{i}.self_attn.{proj_name}", mod))
            for proj_name in ("gate_proj", "up_proj", "down_proj"):
                mod = getattr(layer.mlp, proj_name, None)
                if isinstance(mod, nn.Linear):
                    decoder_linears.append((f"model.layers.{i}.mlp.{proj_name}", mod))
    else:
        for name, mod in model.named_modules():
            if isinstance(mod, nn.Linear) and "layers." in name:
                decoder_linears.append((name, mod))

    for name, module in decoder_linears:
        if layer_bits_map is not None and name in layer_bits_map:
            w_bits = layer_bits_map[name]
        elif recipe.default.weight is not None:
            w_bits = recipe.default.weight.bits
        else:
            w_bits = None

        if w_bits is not None and w_bits in (4, 8):
            with torch.no_grad():
                w_dq = fake_quant_weight_symmetric(module.weight.data, bits=w_bits, axis=0)
                module.weight.data.copy_(w_dq)

    embed_module = None
    if hasattr(model, "get_input_embeddings"):
        embed_module = model.get_input_embeddings()
    elif hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
        embed_module = model.model.embed_tokens

    if embed_module is not None and hasattr(embed_module, "weight"):
        with torch.no_grad():
            w_dq = fake_quant_weight_symmetric(embed_module.weight.data, bits=8, axis=0)
            embed_module.weight.data.copy_(w_dq)

    act_format = recipe.default.act
    if act_format is not None and act_format.bits in (8, 16):
        act_bits = act_format.bits

        for name, module in decoder_linears:
            scale: float | None = None
            zp: int | None = None

            if active_calib is not None:
                if isinstance(active_calib, CalibStats):
                    act_ranges = active_calib.ranges_for_bits(act_bits)
                elif isinstance(active_calib, dict):
                    minmax = active_calib.get("act_minmax") if act_bits >= 16 else None
                    act_ranges = minmax or active_calib.get("act_ranges", {})
                else:
                    act_ranges = {}

                if name in act_ranges:
                    rmin, rmax = act_ranges[name]
                    scale, zp = quantize_asymmetric_per_tensor(
                        rmin=rmin, rmax=rmax, bits=act_bits
                    )

            def make_act_hook(s: float | None, z: int | None, b: int):
                def hook(mod: nn.Module, args: tuple[torch.Tensor, ...]):
                    if len(args) == 0:
                        return args
                    x = args[0]
                    if s is not None and z is not None:
                        x_dq = fake_quant_act_asymmetric(x, scale=s, zp=z, bits=b)
                    else:
                        s_dyn, z_dyn = quantize_asymmetric_per_tensor(x, bits=b)
                        x_dq = fake_quant_act_asymmetric(x, scale=s_dyn, zp=z_dyn, bits=b)
                    return (x_dq,) + args[1:]

                return hook

            h = module.register_forward_pre_hook(make_act_hook(scale, zp, act_bits))
            hook_handles.append(h)

    eff_layer_bits: Any = None
    if layer_bits_map is not None:
        eff_layer_bits = layer_bits_map
    elif recipe.default.weight is not None:
        eff_layer_bits = recipe.default.weight.bits
    else:
        eff_layer_bits = 32.0

    avg_bits, footprint = compute_model_footprint(model, layer_bits_map=eff_layer_bits)

    return {
        "avg_bits_linear": avg_bits,
        "footprint_mb": footprint,
        "hook_handles": hook_handles,
        "chosen_alphas": chosen_alphas,
    }
