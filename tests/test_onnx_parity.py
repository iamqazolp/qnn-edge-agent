import copy

import numpy as np
import onnxruntime as ort
import pytest
import torch
from transformers.models.qwen2.configuration_qwen2 import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from qnn_edge_agent.npu.block_export import (
    Qwen2BlockPrefillWrapper,
    export_decode_block,
    export_prefill_block,
)
from qnn_edge_agent.npu.qdq_insert import (
    insert_qdq,
    quantize_asymmetric_per_tensor,
    quantize_symmetric_per_channel,
)


@pytest.fixture
def tiny_qwen2_layer() -> Qwen2DecoderLayer:
    torch.manual_seed(42)
    cfg = Qwen2Config(
        vocab_size=100,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=512,
    )
    cfg._attn_implementation = "eager"
    return Qwen2DecoderLayer(cfg, layer_idx=0).eval()


def test_fp_onnx_parity(tiny_qwen2_layer: Qwen2DecoderLayer):
    seq_len = 16
    hidden_size = 64
    head_dim = 16

    torch.manual_seed(123)
    x = torch.randn(1, seq_len, hidden_size)
    cos = torch.randn(1, seq_len, head_dim)
    sin = torch.randn(1, seq_len, head_dim)

    wrapper = Qwen2BlockPrefillWrapper(tiny_qwen2_layer, mask_as_constant=True, seq_len=seq_len).eval()
    with torch.no_grad():
        y_torch = wrapper(x, cos, sin).detach().numpy()

    onnx_model = export_prefill_block(
        layer=tiny_qwen2_layer,
        seq_len=seq_len,
        opset_version=21,
        mask_as_constant=True,
    )

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    y_ort = sess.run(
        None,
        {
            "hidden_states": x.numpy(),
            "cos": cos.numpy(),
            "sin": sin.numpy(),
        },
    )[0]

    max_diff = float(np.max(np.abs(y_torch - y_ort)))
    assert max_diff < 1e-5, f"FP parity diff too large: {max_diff}"


def test_w8a16_qdq_parity(tiny_qwen2_layer: Qwen2DecoderLayer):
    seq_len = 16
    hidden_size = 64
    head_dim = 16

    torch.manual_seed(123)
    x = torch.randn(1, seq_len, hidden_size)
    cos = torch.randn(1, seq_len, head_dim)
    sin = torch.randn(1, seq_len, head_dim)

    onnx_model = export_prefill_block(
        layer=tiny_qwen2_layer,
        seq_len=seq_len,
        opset_version=21,
        mask_as_constant=True,
    )

    feed = {"hidden_states": x.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
    qdq_model = insert_qdq(
        onnx_model,
        weight_bits=8,
        act_bits=16,
        calibration_inputs=[feed],
    )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(qdq_model.SerializeToString(), opts)
    y_ort = sess.run(None, feed)[0]

    layer_fq = copy.deepcopy(tiny_qwen2_layer)
    named_linears = [
        layer_fq.self_attn.q_proj,
        layer_fq.self_attn.k_proj,
        layer_fq.self_attn.v_proj,
        layer_fq.self_attn.o_proj,
        layer_fq.mlp.gate_proj,
        layer_fq.mlp.up_proj,
        layer_fq.mlp.down_proj,
    ]

    for lin in named_linears:
        w_pt = lin.weight.data.numpy()
        w_q, scale, _ = quantize_symmetric_per_channel(w_pt.T, bits=8, axis=1)
        w_dq = (w_q.astype(np.float32) * scale[None, :]).T
        lin.weight.data = torch.from_numpy(w_dq)

    wrapper_fq = Qwen2BlockPrefillWrapper(layer_fq, mask_as_constant=True, seq_len=seq_len).eval()

    def make_act_hook(scale: float, zp: int):
        def pre_hook(module, args):
            inp = args[0]
            arr = inp.detach().numpy()
            q = np.clip(np.round(arr / scale) + zp, 0, 65535)
            dq = ((q - zp) * scale).astype(np.float32)
            return (torch.from_numpy(dq),) + args[1:]

        return pre_hook

    inits = {init.name: init for init in onnx_model.graph.initializer}
    act_names = set()
    for node in onnx_model.graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2 and node.input[1] in inits:
            act_names.add(node.input[0])

    from onnx import TensorProto, helper

    temp_model = copy.deepcopy(onnx_model)
    for act_name in act_names:
        temp_model.graph.output.append(helper.make_tensor_value_info(act_name, TensorProto.FLOAT, None))

    sess_temp = ort.InferenceSession(temp_model.SerializeToString())
    outs = sess_temp.run(list(act_names), feed)

    for act_name, val in zip(act_names, outs):
        scale, zp = quantize_asymmetric_per_tensor(val, bits=16)
        hook_fn = make_act_hook(scale, zp)
        if "input_layernorm" in act_name:
            layer_fq.self_attn.q_proj.register_forward_pre_hook(hook_fn)
            layer_fq.self_attn.k_proj.register_forward_pre_hook(hook_fn)
            layer_fq.self_attn.v_proj.register_forward_pre_hook(hook_fn)
        elif "Reshape" in act_name or "o_proj" in act_name:
            layer_fq.self_attn.o_proj.register_forward_pre_hook(hook_fn)
        elif "post_attention_layernorm" in act_name:
            layer_fq.mlp.gate_proj.register_forward_pre_hook(hook_fn)
            layer_fq.mlp.up_proj.register_forward_pre_hook(hook_fn)
        elif "Mul" in act_name:
            layer_fq.mlp.down_proj.register_forward_pre_hook(hook_fn)

    with torch.no_grad():
        y_torch_fq = wrapper_fq(x, cos, sin).detach().numpy()

    max_diff = float(np.max(np.abs(y_torch_fq - y_ort)))
    output_dyn_range = float(y_torch_fq.max() - y_torch_fq.min())
    one_quant_step = output_dyn_range / 255.0

    assert max_diff <= one_quant_step, (
        f"W8A16 diff ({max_diff}) exceeds one output quantization step ({one_quant_step})"
    )


def test_w4a16_qdq_parity(tiny_qwen2_layer: Qwen2DecoderLayer):
    seq_len = 16
    hidden_size = 64
    head_dim = 16

    torch.manual_seed(123)
    x = torch.randn(1, seq_len, hidden_size)
    cos = torch.randn(1, seq_len, head_dim)
    sin = torch.randn(1, seq_len, head_dim)

    onnx_model = export_prefill_block(
        layer=tiny_qwen2_layer,
        seq_len=seq_len,
        opset_version=21,
        mask_as_constant=True,
    )

    feed = {"hidden_states": x.numpy(), "cos": cos.numpy(), "sin": sin.numpy()}
    qdq_model = insert_qdq(
        onnx_model,
        weight_bits=4,
        act_bits=16,
        calibration_inputs=[feed],
    )

    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(qdq_model.SerializeToString(), opts)
    y_ort = sess.run(None, feed)[0]

    layer_fq = copy.deepcopy(tiny_qwen2_layer)
    named_linears = [
        layer_fq.self_attn.q_proj,
        layer_fq.self_attn.k_proj,
        layer_fq.self_attn.v_proj,
        layer_fq.self_attn.o_proj,
        layer_fq.mlp.gate_proj,
        layer_fq.mlp.up_proj,
        layer_fq.mlp.down_proj,
    ]

    for lin in named_linears:
        w_pt = lin.weight.data.numpy()
        w_q, scale, _ = quantize_symmetric_per_channel(w_pt.T, bits=4, axis=1)
        w_dq = (w_q.astype(np.float32) * scale[None, :]).T
        lin.weight.data = torch.from_numpy(w_dq)

    wrapper_fq = Qwen2BlockPrefillWrapper(layer_fq, mask_as_constant=True, seq_len=seq_len).eval()

    def make_act_hook(scale: float, zp: int):
        def pre_hook(module, args):
            inp = args[0]
            arr = inp.detach().numpy()
            q = np.clip(np.round(arr / scale) + zp, 0, 65535)
            dq = ((q - zp) * scale).astype(np.float32)
            return (torch.from_numpy(dq),) + args[1:]

        return pre_hook

    from onnx import helper, TensorProto

    inits = {init.name: init for init in onnx_model.graph.initializer}
    act_names = set()
    for node in onnx_model.graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2 and node.input[1] in inits:
            act_names.add(node.input[0])

    temp_model = copy.deepcopy(onnx_model)
    for act_name in act_names:
        temp_model.graph.output.append(helper.make_tensor_value_info(act_name, TensorProto.FLOAT, None))

    sess_temp = ort.InferenceSession(temp_model.SerializeToString())
    outs = sess_temp.run(list(act_names), feed)

    for act_name, val in zip(act_names, outs):
        scale, zp = quantize_asymmetric_per_tensor(val, bits=16)
        hook_fn = make_act_hook(scale, zp)
        if "input_layernorm" in act_name:
            layer_fq.self_attn.q_proj.register_forward_pre_hook(hook_fn)
            layer_fq.self_attn.k_proj.register_forward_pre_hook(hook_fn)
            layer_fq.self_attn.v_proj.register_forward_pre_hook(hook_fn)
        elif "Reshape" in act_name or "o_proj" in act_name:
            layer_fq.self_attn.o_proj.register_forward_pre_hook(hook_fn)
        elif "post_attention_layernorm" in act_name:
            layer_fq.mlp.gate_proj.register_forward_pre_hook(hook_fn)
            layer_fq.mlp.up_proj.register_forward_pre_hook(hook_fn)
        elif "Mul" in act_name:
            layer_fq.mlp.down_proj.register_forward_pre_hook(hook_fn)

    with torch.no_grad():
        y_torch_fq = wrapper_fq(x, cos, sin).detach().numpy()

    max_diff = float(np.max(np.abs(y_torch_fq - y_ort)))
    output_dyn_range = float(y_torch_fq.max() - y_torch_fq.min())
    one_quant_step = output_dyn_range / 255.0

    assert max_diff <= one_quant_step, (
        f"W4A16 diff ({max_diff}) exceeds one output quantization step ({one_quant_step})"
    )


def test_decode_block_export(tiny_qwen2_layer: Qwen2DecoderLayer):
    kv_seq_len = 64
    hidden_size = 64
    head_dim = 16
    num_kv_heads = 2

    onnx_model = export_decode_block(
        layer=tiny_qwen2_layer,
        kv_seq_len=kv_seq_len,
        opset_version=21,
    )

    x = np.random.randn(1, 1, hidden_size).astype(np.float32)
    cos = np.random.randn(1, 1, head_dim).astype(np.float32)
    sin = np.random.randn(1, 1, head_dim).astype(np.float32)
    pk = np.random.randn(1, num_kv_heads, kv_seq_len, head_dim).astype(np.float32)
    pv = np.random.randn(1, num_kv_heads, kv_seq_len, head_dim).astype(np.float32)

    sess = ort.InferenceSession(onnx_model.SerializeToString())
    outs = sess.run(
        None,
        {
            "hidden_states": x,
            "cos": cos,
            "sin": sin,
            "past_key": pk,
            "past_value": pv,
        },
    )

    assert len(outs) == 3
    assert outs[0].shape == (1, 1, hidden_size)
    assert outs[1].shape == (1, num_kv_heads, kv_seq_len + 1, head_dim)
    assert outs[2].shape == (1, num_kv_heads, kv_seq_len + 1, head_dim)


def test_weight_requantization_is_exact(tiny_qwen2_layer: Qwen2DecoderLayer):
    onnx_model = export_prefill_block(
        layer=tiny_qwen2_layer,
        seq_len=16,
        opset_version=21,
        mask_as_constant=True,
    )

    for bits in (8, 4):
        qdq_model = insert_qdq(
            onnx_model,
            weight_bits=bits,
            act_bits=16,
        )

        inits = {init.name: init for init in qdq_model.graph.initializer}
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))

        from onnx import numpy_helper

        checked = 0
        for name, init in inits.items():
            if name.endswith("_grid"):
                base_name = name[:-5]
                w_on_grid = numpy_helper.to_array(init)
                scale = numpy_helper.to_array(inits[f"{base_name}_scale"])

                w_q_re = np.clip(np.round(w_on_grid / scale[None, :]), qmin, qmax).astype(np.int8)
                w_reconstructed = (w_q_re.astype(np.float32) * scale[None, :]).astype(np.float32)

                assert np.array_equal(w_on_grid, w_reconstructed), (
                    f"Re-quantization not exact for {name} with bits={bits}"
                )
                checked += 1

        assert checked == 7, f"Expected 7 linear weight grids, found {checked}"

