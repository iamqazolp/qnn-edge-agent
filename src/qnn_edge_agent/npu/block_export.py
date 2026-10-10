import io
from pathlib import Path

import onnx
import onnx.version_converter
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    apply_rotary_pos_emb,
    repeat_kv,
)

from qnn_edge_agent.env import MODEL_ID


def get_real_block(layer_idx: int = 0, model_id: str = MODEL_ID) -> Qwen2DecoderLayer:
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    return model.model.layers[layer_idx].eval()


class Qwen2BlockPrefillWrapper(nn.Module):
    def __init__(
        self,
        layer: Qwen2DecoderLayer,
        mask_as_constant: bool = True,
        seq_len: int = 128,
    ) -> None:
        super().__init__()
        self.layer = layer
        self.mask_as_constant = mask_as_constant
        if mask_as_constant:
            mask = torch.triu(
                torch.full((1, 1, seq_len, seq_len), -1e4, dtype=torch.float32),
                diagonal=1,
            )
            self.register_buffer("causal_mask", mask)
        else:
            self.causal_mask = None

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = self.causal_mask if self.mask_as_constant else attention_mask
        return self.layer(
            hidden_states=hidden_states,
            attention_mask=mask,
            position_embeddings=(cos, sin),
            use_cache=False,
        )


class Qwen2BlockDecodeWrapper(nn.Module):
    def __init__(self, layer: Qwen2DecoderLayer, config: AutoConfig | None = None) -> None:
        super().__init__()
        self.layer = layer
        cfg = config or getattr(layer, "config", getattr(getattr(layer, "self_attn", None), "config", None))
        self.hidden_size = getattr(layer, "hidden_size", cfg.hidden_size if cfg else 896)
        self.num_heads = cfg.num_attention_heads if cfg else 14
        self.num_kv_heads = cfg.num_key_value_heads if cfg else 2
        self.head_dim = getattr(layer.self_attn, "head_dim", getattr(cfg, "head_dim", self.hidden_size // self.num_heads))
        self.num_groups = self.num_heads // self.num_kv_heads
        self.scaling = self.head_dim**-0.5

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_key: torch.Tensor,
        past_value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        residual = hidden_states
        normed = self.layer.input_layernorm(hidden_states)

        q = self.layer.self_attn.q_proj(normed).view(1, 1, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.layer.self_attn.k_proj(normed).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.layer.self_attn.v_proj(normed).view(1, 1, self.num_kv_heads, self.head_dim).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        k_all = torch.cat([past_key, k], dim=2)
        v_all = torch.cat([past_value, v], dim=2)

        k_rep = repeat_kv(k_all, self.num_groups)
        v_rep = repeat_kv(v_all, self.num_groups)

        attn_weights = torch.matmul(q, k_rep.transpose(2, 3)) * self.scaling
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32)

        attn_out = torch.matmul(attn_weights, v_rep).transpose(1, 2).reshape(1, 1, -1)
        attn_out = self.layer.self_attn.o_proj(attn_out)

        h = residual + attn_out
        h = h + self.layer.mlp(self.layer.post_attention_layernorm(h))
        return h, k_all, v_all


def export_prefill_block(
    layer: Qwen2DecoderLayer | None = None,
    layer_idx: int = 0,
    output_path: str | Path | None = None,
    batch_size: int = 1,
    seq_len: int = 128,
    opset_version: int = 21,
    mask_as_constant: bool = True,
) -> onnx.ModelProto:
    if layer is None:
        layer = get_real_block(layer_idx=layer_idx)

    layer = layer.eval()
    cfg = getattr(layer, "config", getattr(getattr(layer, "self_attn", None), "config", None))
    hidden_size = getattr(layer, "hidden_size", cfg.hidden_size if cfg else 896)
    num_heads = cfg.num_attention_heads if cfg else 14
    head_dim = getattr(layer.self_attn, "head_dim", getattr(cfg, "head_dim", hidden_size // num_heads))

    wrapper = Qwen2BlockPrefillWrapper(layer, mask_as_constant=mask_as_constant, seq_len=seq_len).eval()

    dummy_x = torch.zeros(batch_size, seq_len, hidden_size, dtype=torch.float32)
    dummy_cos = torch.zeros(batch_size, seq_len, head_dim, dtype=torch.float32)
    dummy_sin = torch.zeros(batch_size, seq_len, head_dim, dtype=torch.float32)

    input_names = ["hidden_states", "cos", "sin"]
    dummy_inputs = (dummy_x, dummy_cos, dummy_sin)

    if not mask_as_constant:
        input_names.append("attention_mask")
        dummy_mask = torch.triu(
            torch.full((batch_size, 1, seq_len, seq_len), -1e4, dtype=torch.float32),
            diagonal=1,
        )
        dummy_inputs = (dummy_x, dummy_cos, dummy_sin, dummy_mask)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        dummy_inputs,
        buf,
        input_names=input_names,
        output_names=["output"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    model = onnx.load_model_from_string(buf.getvalue())
    if opset_version > 17:
        model = onnx.version_converter.convert_version(model, opset_version)
    model.ir_version = 10

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, str(output_path))

    return model


def export_decode_block(
    layer: Qwen2DecoderLayer | None = None,
    layer_idx: int = 0,
    output_path: str | Path | None = None,
    batch_size: int = 1,
    kv_seq_len: int = 512,
    opset_version: int = 21,
) -> onnx.ModelProto:
    if layer is None:
        layer = get_real_block(layer_idx=layer_idx)

    layer = layer.eval()
    cfg = getattr(layer, "config", getattr(getattr(layer, "self_attn", None), "config", None))
    hidden_size = getattr(layer, "hidden_size", cfg.hidden_size if cfg else 896)
    num_heads = cfg.num_attention_heads if cfg else 14
    num_kv_heads = cfg.num_key_value_heads if cfg else 2
    head_dim = getattr(layer.self_attn, "head_dim", getattr(cfg, "head_dim", hidden_size // num_heads))

    wrapper = Qwen2BlockDecodeWrapper(layer, config=cfg).eval()

    dummy_x = torch.zeros(batch_size, 1, hidden_size, dtype=torch.float32)
    dummy_cos = torch.zeros(batch_size, 1, head_dim, dtype=torch.float32)
    dummy_sin = torch.zeros(batch_size, 1, head_dim, dtype=torch.float32)
    dummy_k = torch.zeros(batch_size, num_kv_heads, kv_seq_len, head_dim, dtype=torch.float32)
    dummy_v = torch.zeros(batch_size, num_kv_heads, kv_seq_len, head_dim, dtype=torch.float32)

    buf = io.BytesIO()
    torch.onnx.export(
        wrapper,
        (dummy_x, dummy_cos, dummy_sin, dummy_k, dummy_v),
        buf,
        input_names=["hidden_states", "cos", "sin", "past_key", "past_value"],
        output_names=["next_hidden_states", "next_key", "next_value"],
        opset_version=17,
        do_constant_folding=True,
        dynamo=False,
    )

    model = onnx.load_model_from_string(buf.getvalue())
    if opset_version > 17:
        model = onnx.version_converter.convert_version(model, opset_version)
    model.ir_version = 10

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, str(output_path))

    return model
