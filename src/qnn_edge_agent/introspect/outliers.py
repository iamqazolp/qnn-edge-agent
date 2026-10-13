from dataclasses import asdict, dataclass
from typing import Any, Sequence
import numpy as np
import torch
import torch.nn as nn


PROJECTION_NAMES: tuple[str, ...] = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)


@dataclass(frozen=True)
class OutlierMetrics:
    channel_max: list[float]
    max_abs: float
    median_channel_max: float
    outlier_ratio: float
    kurtosis: float


def compute_kurtosis(tensor: torch.Tensor | np.ndarray) -> float:
    if isinstance(tensor, torch.Tensor):
        x = tensor.detach().to(torch.float64).flatten()
    else:
        x = torch.from_numpy(tensor).to(torch.float64).flatten()

    n = x.numel()
    if n == 0:
        return 0.0

    mu = torch.mean(x)
    diff = x - mu
    var = torch.mean(diff**2)
    if var.item() == 0.0:
        return 0.0

    m4 = torch.mean(diff**4)
    kurt = (m4 / (var**2)).item()
    return float(kurt)


def compute_channel_max(tensor: torch.Tensor | np.ndarray) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.from_numpy(tensor)
    t = tensor.detach().to(torch.float32)
    if t.ndim == 3:
        t = t.reshape(-1, t.shape[-1])
    elif t.ndim == 1:
        t = t.unsqueeze(0)
    abs_t = torch.abs(t)
    return torch.max(abs_t, dim=0).values


def compute_outlier_ratio(
    channel_max: torch.Tensor | np.ndarray | Sequence[float],
) -> tuple[float, float, float]:
    if isinstance(channel_max, torch.Tensor):
        c = channel_max.detach().to(torch.float32).cpu().numpy()
    elif isinstance(channel_max, np.ndarray):
        c = channel_max.astype(np.float32)
    else:
        c = np.array(channel_max, dtype=np.float32)

    if c.size == 0:
        return 0.0, 0.0, 0.0

    max_abs = float(np.max(c))
    median_channel_max = float(np.median(c))
    if median_channel_max > 0.0:
        outlier_ratio = float(max_abs / median_channel_max)
    else:
        outlier_ratio = 1.0 if max_abs == 0.0 else float("inf")

    return max_abs, median_channel_max, outlier_ratio


def compute_activation_metrics(
    tensor: torch.Tensor | np.ndarray,
) -> OutlierMetrics:
    c_j = compute_channel_max(tensor)
    max_abs, median_max, ratio = compute_outlier_ratio(c_j)
    kurt = compute_kurtosis(tensor)
    return OutlierMetrics(
        channel_max=[float(v) for v in c_j.cpu().tolist()],
        max_abs=max_abs,
        median_channel_max=median_max,
        outlier_ratio=ratio,
        kurtosis=kurt,
    )


class ActivationMomentAccumulator:
    def __init__(self) -> None:
        self.channel_max: torch.Tensor | None = None
        self.n: int = 0
        self.s1: float = 0.0
        self.s2: float = 0.0
        self.s3: float = 0.0
        self.s4: float = 0.0

    def update(self, x_flat_2d: torch.Tensor) -> None:
        x_f32 = x_flat_2d.detach().to(torch.float32)
        local_c = torch.max(torch.abs(x_f32), dim=0).values.cpu()
        if self.channel_max is None:
            self.channel_max = local_c
        else:
            self.channel_max = torch.maximum(self.channel_max, local_c)

        x_f64 = x_f32.cpu().to(torch.float64).flatten()
        count = x_f64.numel()
        if count == 0:
            return

        self.n += count
        self.s1 += float(x_f64.sum().item())
        x2 = x_f64 * x_f64
        self.s2 += float(x2.sum().item())
        self.s3 += float((x2 * x_f64).sum().item())
        self.s4 += float((x2 * x2).sum().item())

    def finalize(self) -> OutlierMetrics:
        if self.n == 0 or self.channel_max is None:
            return OutlierMetrics(
                channel_max=[],
                max_abs=0.0,
                median_channel_max=0.0,
                outlier_ratio=0.0,
                kurtosis=0.0,
            )

        c = self.channel_max.cpu().numpy()
        max_abs, median_max, ratio = compute_outlier_ratio(c)

        mu = self.s1 / self.n
        var = (self.s2 - self.n * (mu**2)) / self.n
        if var <= 0.0:
            kurtosis = 0.0
        else:
            m4 = (
                self.s4
                - 4.0 * mu * self.s3
                + 6.0 * (mu**2) * self.s2
                - 3.0 * self.n * (mu**4)
            ) / self.n
            kurtosis = float(m4 / (var**2))

        return OutlierMetrics(
            channel_max=[float(v) for v in c.tolist()],
            max_abs=max_abs,
            median_channel_max=median_max,
            outlier_ratio=ratio,
            kurtosis=kurtosis,
        )


class ProjectionOutlierTracker:
    def __init__(self) -> None:
        self.channel_max_all: torch.Tensor | None = None
        self.channel_max_nobos: torch.Tensor | None = None
        self.n_all: int = 0
        self.s1_all: float = 0.0
        self.s2_all: float = 0.0
        self.s3_all: float = 0.0
        self.s4_all: float = 0.0
        self.n_nobos: int = 0
        self.s1_nobos: float = 0.0
        self.s2_nobos: float = 0.0
        self.s3_nobos: float = 0.0
        self.s4_nobos: float = 0.0

    def update(self, activation: torch.Tensor) -> None:
        abs_act = torch.abs(activation.detach().to(torch.float32))

        if activation.ndim == 3:
            b, t, d = activation.shape
            c_all = torch.amax(abs_act, dim=(0, 1)).cpu()
            if self.channel_max_all is None:
                self.channel_max_all = c_all
            else:
                self.channel_max_all = torch.maximum(self.channel_max_all, c_all)

            x_f64 = activation.detach().cpu().to(torch.float64)
            n_all_batch = x_f64.numel()
            s1_all_batch = float(x_f64.sum().item())
            x2_all = x_f64 * x_f64
            s2_all_batch = float(x2_all.sum().item())
            s3_all_batch = float((x2_all * x_f64).sum().item())
            s4_all_batch = float((x2_all * x2_all).sum().item())

            self.n_all += n_all_batch
            self.s1_all += s1_all_batch
            self.s2_all += s2_all_batch
            self.s3_all += s3_all_batch
            self.s4_all += s4_all_batch

            if t > 1:
                c_nobos = torch.amax(abs_act[:, 1:, :], dim=(0, 1)).cpu()
                if self.channel_max_nobos is None:
                    self.channel_max_nobos = c_nobos
                else:
                    self.channel_max_nobos = torch.maximum(self.channel_max_nobos, c_nobos)

                x0_f64 = x_f64[:, 0, :]
                n_0 = x0_f64.numel()
                s1_0 = float(x0_f64.sum().item())
                x2_0 = x0_f64 * x0_f64
                s2_0 = float(x2_0.sum().item())
                s3_0 = float((x2_0 * x0_f64).sum().item())
                s4_0 = float((x2_0 * x2_0).sum().item())

                self.n_nobos += (n_all_batch - n_0)
                self.s1_nobos += (s1_all_batch - s1_0)
                self.s2_nobos += (s2_all_batch - s2_0)
                self.s3_nobos += (s3_all_batch - s3_0)
                self.s4_nobos += (s4_all_batch - s4_0)

        elif activation.ndim == 2:
            t, d = activation.shape
            c_all = torch.amax(abs_act, dim=0).cpu()
            if self.channel_max_all is None:
                self.channel_max_all = c_all
            else:
                self.channel_max_all = torch.maximum(self.channel_max_all, c_all)

            x_f64 = activation.detach().cpu().to(torch.float64)
            n_all_batch = x_f64.numel()
            s1_all_batch = float(x_f64.sum().item())
            x2_all = x_f64 * x_f64
            s2_all_batch = float(x2_all.sum().item())
            s3_all_batch = float((x2_all * x_f64).sum().item())
            s4_all_batch = float((x2_all * x2_all).sum().item())

            self.n_all += n_all_batch
            self.s1_all += s1_all_batch
            self.s2_all += s2_all_batch
            self.s3_all += s3_all_batch
            self.s4_all += s4_all_batch

            if t > 1:
                c_nobos = torch.amax(abs_act[1:, :], dim=0).cpu()
                if self.channel_max_nobos is None:
                    self.channel_max_nobos = c_nobos
                else:
                    self.channel_max_nobos = torch.maximum(self.channel_max_nobos, c_nobos)

                x0_f64 = x_f64[0, :]
                n_0 = x0_f64.numel()
                s1_0 = float(x0_f64.sum().item())
                x2_0 = x0_f64 * x0_f64
                s2_0 = float(x2_0.sum().item())
                s3_0 = float((x2_0 * x0_f64).sum().item())
                s4_0 = float((x2_0 * x2_0).sum().item())

                self.n_nobos += (n_all_batch - n_0)
                self.s1_nobos += (s1_all_batch - s1_0)
                self.s2_nobos += (s2_all_batch - s2_0)
                self.s3_nobos += (s3_all_batch - s3_0)
                self.s4_nobos += (s4_all_batch - s4_0)

        elif activation.ndim == 1:
            c_all = abs_act.cpu()
            if self.channel_max_all is None:
                self.channel_max_all = c_all
            else:
                self.channel_max_all = torch.maximum(self.channel_max_all, c_all)

            x_f64 = activation.detach().cpu().to(torch.float64)
            n_all_batch = x_f64.numel()
            s1_all_batch = float(x_f64.sum().item())
            x2_all = x_f64 * x_f64
            s2_all_batch = float(x2_all.sum().item())
            s3_all_batch = float((x2_all * x_f64).sum().item())
            s4_all_batch = float((x2_all * x2_all).sum().item())

            self.n_all += n_all_batch
            self.s1_all += s1_all_batch
            self.s2_all += s2_all_batch
            self.s3_all += s3_all_batch
            self.s4_all += s4_all_batch

    def _finalize_moments(
        self,
        c_tensor: torch.Tensor | None,
        n: int,
        s1: float,
        s2: float,
        s3: float,
        s4: float,
    ) -> OutlierMetrics:
        if n == 0 or c_tensor is None:
            return OutlierMetrics(
                channel_max=[],
                max_abs=0.0,
                median_channel_max=0.0,
                outlier_ratio=0.0,
                kurtosis=0.0,
            )

        c = c_tensor.cpu().numpy()
        max_abs, median_max, ratio = compute_outlier_ratio(c)

        mu = s1 / n
        var = (s2 - n * (mu**2)) / n
        if var <= 0.0:
            kurtosis = 0.0
        else:
            m4 = (
                s4
                - 4.0 * mu * s3
                + 6.0 * (mu**2) * s2
                - 3.0 * n * (mu**4)
            ) / n
            kurtosis = float(m4 / (var**2))

        return OutlierMetrics(
            channel_max=[float(v) for v in c.tolist()],
            max_abs=max_abs,
            median_channel_max=median_max,
            outlier_ratio=ratio,
            kurtosis=kurtosis,
        )

    def finalize(self) -> dict[str, OutlierMetrics]:
        return {
            "all_tokens": self._finalize_moments(
                self.channel_max_all,
                self.n_all,
                self.s1_all,
                self.s2_all,
                self.s3_all,
                self.s4_all,
            ),
            "no_bos": self._finalize_moments(
                self.channel_max_nobos,
                self.n_nobos,
                self.s1_nobos,
                self.s2_nobos,
                self.s3_nobos,
                self.s4_nobos,
            ),
        }


class OutlierProfiler:
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.trackers: dict[tuple[int, str], ProjectionOutlierTracker] = {}
        self.hooks: list[Any] = []
        self._register_hooks()

    def _find_layers(self) -> list[nn.Module]:
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            return list(self.model.model.layers)
        if hasattr(self.model, "layers"):
            return list(self.model.layers)
        for _, module in self.model.named_modules():
            if isinstance(module, (nn.ModuleList, list)) and len(module) > 0:
                first = module[0]
                if hasattr(first, "self_attn") or hasattr(first, "mlp"):
                    return list(module)
        return []

    def _resolve_projection(self, layer: nn.Module, proj_name: str) -> nn.Module | None:
        if proj_name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            attn = getattr(layer, "self_attn", None)
            return getattr(attn, proj_name, None) if attn is not None else None
        if proj_name in ("gate_proj", "up_proj", "down_proj"):
            mlp = getattr(layer, "mlp", None)
            return getattr(mlp, proj_name, None) if mlp is not None else None
        return None

    def _register_hooks(self) -> None:
        layers = self._find_layers()
        for layer_idx, layer in enumerate(layers):
            for proj_name in PROJECTION_NAMES:
                proj_mod = self._resolve_projection(layer, proj_name)
                if proj_mod is None:
                    continue
                tracker = ProjectionOutlierTracker()
                self.trackers[(layer_idx, proj_name)] = tracker

                def _make_hook(tr: ProjectionOutlierTracker):
                    def _hook(_module: nn.Module, inputs: tuple[Any, ...]) -> None:
                        if inputs and isinstance(inputs[0], torch.Tensor):
                            tr.update(inputs[0])

                    return _hook

                hook_handle = proj_mod.register_forward_pre_hook(_make_hook(tracker))
                self.hooks.append(hook_handle)

    def remove_hooks(self) -> None:
        for h in self.hooks:
            h.remove()
        self.hooks.clear()

    def __enter__(self) -> "OutlierProfiler":
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.remove_hooks()

    def get_telemetry(self) -> dict[str, Any]:
        layer_indices = sorted(list(set(l for l, _ in self.trackers.keys())))
        telemetry: dict[str, Any] = {
            "num_layers": len(layer_indices),
            "projections": list(PROJECTION_NAMES),
            "layers": {},
        }
        for l in layer_indices:
            layer_dict: dict[str, Any] = {}
            for p in PROJECTION_NAMES:
                if (l, p) in self.trackers:
                    fin = self.trackers[(l, p)].finalize()
                    layer_dict[p] = {
                        "all_tokens": asdict(fin["all_tokens"]),
                        "no_bos": asdict(fin["no_bos"]),
                    }
            telemetry["layers"][str(l)] = layer_dict
        return telemetry
