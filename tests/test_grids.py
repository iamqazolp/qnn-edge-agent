import numpy as np
import torch
import torch.nn as nn

from qnn_edge_agent.quant.grids import (
    compute_model_footprint,
    dequantize_symmetric_per_channel,
    fake_quant_act_asymmetric,
    fake_quant_weight_symmetric,
    quantize_asymmetric_per_tensor,
    quantize_symmetric_per_channel,
)


def test_symmetric_weight_roundtrip_and_bound_torch():
    torch.manual_seed(42)
    w = torch.randn(32, 64)
    w_q, scale, zp = quantize_symmetric_per_channel(w, bits=8, axis=0)

    assert w_q.dtype == torch.int32
    assert w_q.min().item() >= -128
    assert w_q.max().item() <= 127
    assert (zp == 0).all()

    w_dq = dequantize_symmetric_per_channel(w_q, scale, axis=0)
    diff = (w - w_dq).abs()
    assert (diff <= scale / 2.0 + 1e-6).all()


def test_symmetric_weight_roundtrip_and_bound_numpy():
    np.random.seed(42)
    w = np.random.randn(32, 64).astype(np.float32)
    w_q, scale, zp = quantize_symmetric_per_channel(w, bits=8, axis=0)

    assert w_q.dtype == np.int32
    assert w_q.min() >= -128
    assert w_q.max() <= 127
    assert np.all(zp == 0)

    w_dq = dequantize_symmetric_per_channel(w_q, scale, axis=0)
    diff = np.abs(w - w_dq)
    assert np.all(diff <= scale / 2.0 + 1e-6)


def test_symmetric_weight_int4_range():
    torch.manual_seed(42)
    w = torch.randn(16, 32)
    w_q, scale, zp = quantize_symmetric_per_channel(w, bits=4, axis=0)

    assert w_q.min().item() >= -8
    assert w_q.max().item() <= 7
    assert (zp == 0).all()

    w_dq = dequantize_symmetric_per_channel(w_q, scale, axis=0)
    diff = (w - w_dq).abs()
    assert (diff <= scale / 2.0 + 1e-6).all()


def test_symmetric_weight_idempotence():
    torch.manual_seed(42)
    w = torch.randn(20, 40)
    w_dq1 = fake_quant_weight_symmetric(w, bits=8, axis=0)
    w_dq2 = fake_quant_weight_symmetric(w_dq1, bits=8, axis=0)
    assert torch.allclose(w_dq1, w_dq2, atol=1e-7)

    w_dq4_1 = fake_quant_weight_symmetric(w, bits=4, axis=0)
    w_dq4_2 = fake_quant_weight_symmetric(w_dq4_1, bits=4, axis=0)
    assert torch.allclose(w_dq4_1, w_dq4_2, atol=1e-7)


def test_asymmetric_act_exact_zero():
    for bits in (8, 16):
        data = torch.linspace(-4.0, 12.0, 100)
        scale, zp = quantize_asymmetric_per_tensor(data, bits=bits)

        zero_tensor = torch.tensor([0.0])
        zero_dq = fake_quant_act_asymmetric(zero_tensor, scale, zp, bits=bits)
        assert zero_dq.item() == 0.0

        pos_only = torch.linspace(0.0, 10.0, 50)
        scale_p, zp_p = quantize_asymmetric_per_tensor(pos_only, bits=bits)
        assert zp_p == 0
        zero_dq_p = fake_quant_act_asymmetric(zero_tensor, scale_p, zp_p, bits=bits)
        assert zero_dq_p.item() == 0.0

        neg_only = torch.linspace(-10.0, 0.0, 50)
        scale_n, zp_n = quantize_asymmetric_per_tensor(neg_only, bits=bits)
        assert zp_n == (1 << bits) - 1
        zero_dq_n = fake_quant_act_asymmetric(zero_tensor, scale_n, zp_n, bits=bits)
        assert zero_dq_n.item() == 0.0


def test_asymmetric_act_idempotence_and_bound():
    for bits in (8, 16):
        data = np.random.randn(100).astype(np.float32)
        scale, zp = quantize_asymmetric_per_tensor(data, bits=bits)

        dq1 = fake_quant_act_asymmetric(data, scale, zp, bits=bits)
        dq2 = fake_quant_act_asymmetric(dq1, scale, zp, bits=bits)
        assert np.allclose(dq1, dq2, atol=1e-7)

        diff = np.abs(data - dq1)
        assert np.all(diff <= scale / 2.0 + 1e-6)


def test_model_footprint_constants():
    avg_bits, fp_e0 = compute_model_footprint(None, layer_bits_map=None)
    assert avg_bits == 32.0
    assert abs(fp_e0 - 988.07) < 0.1

    avg_bits, fp_w8 = compute_model_footprint(None, layer_bits_map=8)
    assert avg_bits == 8.0
    assert abs(fp_w8 - 494.08) < 0.1

    avg_bits, fp_w4 = compute_model_footprint(None, layer_bits_map=4)
    assert avg_bits == 4.0
    assert abs(fp_w4 - 315.15) < 0.1

    avg_bits, fp_w46 = compute_model_footprint(None, layer_bits_map=4.6)
    assert avg_bits == 4.6
    assert abs(fp_w46 - 341.99) < 0.1


def test_model_footprint_with_dummy_module():
    class DummyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(1000, 128)
            self.model.layers = nn.ModuleList([
                nn.ModuleDict({
                    "self_attn": nn.ModuleDict({
                        "q_proj": nn.Linear(128, 128, bias=False),
                        "k_proj": nn.Linear(128, 128, bias=False),
                    })
                })
            ])
            self.lm_head = nn.Linear(128, 1000, bias=False)
            self.lm_head.weight = self.model.embed_tokens.weight

    dummy = DummyModel()
    layer_map = {
        "model.layers.0.self_attn.q_proj": 4,
        "model.layers.0.self_attn.k_proj": 8,
    }
    avg_bits, footprint = compute_model_footprint(dummy, layer_bits_map=layer_map)
    assert avg_bits == 6.0
    assert footprint > 0.0
