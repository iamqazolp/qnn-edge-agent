import pytest
import torch
import torch.nn as nn

from qnn_edge_agent.introspect.outliers import (
    PROJECTION_NAMES,
    ActivationMomentAccumulator,
    OutlierProfiler,
    ProjectionOutlierTracker,
    compute_activation_metrics,
    compute_kurtosis,
)


def test_analytical_synthetic_tensor() -> None:
    x = torch.zeros(1, 3, 5, dtype=torch.float32)
    x[0, 0, 0] = 100.0
    x[0, 0, 1:] = 0.0
    x[0, 1, :4] = 1.0
    x[0, 1, 4] = 25.0
    x[0, 2, :4] = 1.0
    x[0, 2, 4] = 25.0

    metrics_all = compute_activation_metrics(x)
    assert metrics_all.max_abs == pytest.approx(100.0, rel=1e-6)
    assert metrics_all.median_channel_max == pytest.approx(1.0, rel=1e-6)
    assert metrics_all.outlier_ratio == pytest.approx(100.0, rel=1e-6)
    assert metrics_all.kurtosis == pytest.approx(10.4745250835, rel=1e-5)

    x_nobos = x[:, 1:, :]
    metrics_nobos = compute_activation_metrics(x_nobos)
    assert metrics_nobos.max_abs == pytest.approx(25.0, rel=1e-6)
    assert metrics_nobos.median_channel_max == pytest.approx(1.0, rel=1e-6)
    assert metrics_nobos.outlier_ratio == pytest.approx(25.0, rel=1e-6)
    assert metrics_nobos.kurtosis == pytest.approx(3.25, rel=1e-6)

    tracker = ProjectionOutlierTracker()
    tracker.update(x)
    finalized = tracker.finalize()

    all_t = finalized["all_tokens"]
    no_b = finalized["no_bos"]

    assert all_t.max_abs == pytest.approx(100.0, rel=1e-6)
    assert all_t.median_channel_max == pytest.approx(1.0, rel=1e-6)
    assert all_t.outlier_ratio == pytest.approx(100.0, rel=1e-6)
    assert all_t.kurtosis == pytest.approx(10.4745250835, rel=1e-5)

    assert no_b.max_abs == pytest.approx(25.0, rel=1e-6)
    assert no_b.median_channel_max == pytest.approx(1.0, rel=1e-6)
    assert no_b.outlier_ratio == pytest.approx(25.0, rel=1e-6)
    assert no_b.kurtosis == pytest.approx(3.25, rel=1e-6)


def test_kurtosis_distributions() -> None:
    torch.manual_seed(0)
    normal_samples = torch.randn(200000, dtype=torch.float32)
    kurt_normal = compute_kurtosis(normal_samples)
    assert kurt_normal == pytest.approx(3.0, abs=0.1)

    uniform_samples = (torch.rand(200000, dtype=torch.float32) - 0.5) * 2.0
    kurt_uniform = compute_kurtosis(uniform_samples)
    assert kurt_uniform == pytest.approx(1.8, abs=0.05)


def test_chunked_moment_accumulation_exactness() -> None:
    torch.manual_seed(42)
    x1 = torch.randn(20, 16, dtype=torch.float32) * 3.0 + 1.0
    x2 = torch.randn(30, 16, dtype=torch.float32) * 2.0 - 0.5
    x3 = torch.randn(15, 16, dtype=torch.float32) * 5.0

    x_full = torch.cat([x1, x2, x3], dim=0)

    direct_metrics = compute_activation_metrics(x_full)

    accum = ActivationMomentAccumulator()
    accum.update(x1)
    accum.update(x2)
    accum.update(x3)
    accum_metrics = accum.finalize()

    assert accum_metrics.max_abs == pytest.approx(direct_metrics.max_abs, rel=1e-6)
    assert accum_metrics.median_channel_max == pytest.approx(
        direct_metrics.median_channel_max, rel=1e-6
    )
    assert accum_metrics.outlier_ratio == pytest.approx(
        direct_metrics.outlier_ratio, rel=1e-6
    )
    assert accum_metrics.kurtosis == pytest.approx(direct_metrics.kurtosis, rel=1e-6)


class ToyAttention(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        attn = (q + k + v) / 3.0
        return self.o_proj(attn)


class ToyMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class ToyDecoderLayer(nn.Module):
    def __init__(self, dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.self_attn = ToyAttention(dim)
        self.mlp = ToyMLP(dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.self_attn(x)
        x = x + attn_out
        mlp_out = self.mlp(x)
        return x + mlp_out


class ToyModel(nn.Module):
    def __init__(self, num_layers: int = 2, dim: int = 16, hidden_dim: int = 32) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [ToyDecoderLayer(dim, hidden_dim) for _ in range(num_layers)]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def test_outlier_profiler_with_toy_model() -> None:
    model = ToyModel(num_layers=2, dim=8, hidden_dim=16)

    with OutlierProfiler(model) as profiler:
        x1 = torch.randn(1, 10, 8)
        x1[0, 0, 0] = 50.0
        x2 = torch.randn(1, 12, 8)
        x2[0, 0, 0] = 80.0

        with torch.no_grad():
            model(x1)
            model(x2)

        telemetry = profiler.get_telemetry()

    assert telemetry["num_layers"] == 2
    assert set(telemetry["projections"]) == set(PROJECTION_NAMES)

    for l_idx in ("0", "1"):
        assert l_idx in telemetry["layers"]
        for p in PROJECTION_NAMES:
            assert p in telemetry["layers"][l_idx]
            proj_data = telemetry["layers"][l_idx][p]
            assert "all_tokens" in proj_data
            assert "no_bos" in proj_data
            assert proj_data["all_tokens"]["outlier_ratio"] > 0.0
            assert proj_data["all_tokens"]["kurtosis"] > 0.0
            assert proj_data["no_bos"]["outlier_ratio"] > 0.0
            assert proj_data["no_bos"]["kurtosis"] > 0.0

    assert len(profiler.hooks) == 0
