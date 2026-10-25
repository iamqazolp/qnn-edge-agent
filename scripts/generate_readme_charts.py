"""Generate publication-grade SVG charts for README.md from measured telemetry in results/."""

from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
ASSETS_DIR = REPO_ROOT / "docs" / "assets"
PREVIEW_DIR = Path(os.environ.get("QEA_CHART_PREVIEW_DIR", "")) if os.environ.get("QEA_CHART_PREVIEW_DIR") else None

BG_DARK = "#0f172a"
PANEL_DARK = "#1e293b"
GRID_COLOR = "#334155"
TEXT_MAIN = "#f8fafc"


def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(PANEL_DARK)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#475569")
    ax.tick_params(colors="#cbd5e1", labelsize=10)
    ax.grid(True, linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.7, axis="y")
    ax.set_axisbelow(True)


def _save(fig: plt.Figure, name: str) -> None:
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(ASSETS_DIR / f"{name}.svg", format="svg", facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.22)
    if PREVIEW_DIR is not None:
        PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
        fig.savefig(PREVIEW_DIR / f"{name}.png", format="png", dpi=160, facecolor=fig.get_facecolor(), bbox_inches="tight", pad_inches=0.22)
    plt.close(fig)


def plot_layer_kurtosis(telemetry: dict) -> None:
    layers = list(range(24))
    down_nb = [telemetry["layers"][str(i)]["down_proj"]["no_bos"]["kurtosis"] for i in layers]
    gate_nb = [telemetry["layers"][str(i)]["gate_proj"]["no_bos"]["kurtosis"] for i in layers]
    up_nb = [telemetry["layers"][str(i)]["up_proj"]["no_bos"]["kurtosis"] for i in layers]
    q_nb = [telemetry["layers"][str(i)]["q_proj"]["no_bos"]["kurtosis"] for i in layers]

    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    x = np.array(layers)
    ax.fill_between(x, down_nb, color="#f97316", alpha=0.18)
    ax.plot(x, down_nb, color="#f97316", linewidth=2.5, marker="o", markersize=5.2, label="mlp.down_proj (SwiGLU output)")
    ax.plot(x, gate_nb, color="#38bdf8", linewidth=1.9, linestyle="--", label="mlp.gate_proj")
    ax.plot(x, up_nb, color="#a855f7", linewidth=1.9, marker="s", markersize=3.8, label="mlp.up_proj")
    ax.plot(x, q_nb, color="#22c55e", linewidth=1.8, label="self_attn.q_proj")

    for idx, offset in [(0, (24, -4)), (1, (26, 8)), (18, (0, 14)), (23, (-26, 14))]:
        val = down_nb[idx]
        ax.annotate(
            f"L{idx}: {val:.1f}",
            xy=(idx, val),
            xytext=offset,
            textcoords="offset points",
            ha="center",
            fontsize=9.2,
            fontweight="bold",
            color="#fdba74",
            bbox=dict(boxstyle="round,pad=0.28", fc="#0f172a", ec="#f97316", lw=1.0),
        )

    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{i}" for i in layers], fontsize=9.5)
    ax.set_xlim(-0.6, 23.6)
    ax.set_ylim(0, 690)
    ax.set_xlabel("Transformer Block Index (Layer 0 → Layer 23)", color="#e2e8f0", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_ylabel("Excess Kurtosis κ (non-BOS tokens)", color="#e2e8f0", fontsize=11, fontweight="bold", labelpad=8)

    ax.set_title(
        "Layer-Wise Activation Excess Kurtosis Across Qwen2.5-0.5B-Instruct (Non-BOS Tokens)\n"
        "Post-SwiGLU inputs to mlp.down_proj exhibit 5x–15x higher kurtosis than attention or pre-SwiGLU projections",
        color=TEXT_MAIN,
        fontsize=12,
        fontweight="bold",
        loc="center",
        pad=38,
    )

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.115),
        ncol=4,
        frameon=True,
        facecolor=BG_DARK,
        edgecolor="#334155",
        fontsize=9.5,
        labelcolor="#e2e8f0",
    )
    leg.get_frame().set_linewidth(1.0)

    _save(fig, "layer_kurtosis_profile")


def plot_bos_sink_vs_swiglu(telemetry: dict) -> None:
    cases = [
        ("L0\ndown_proj", telemetry["layers"]["0"]["down_proj"]),
        ("L6\nup_proj\n(BOS Sink)", telemetry["layers"]["6"]["up_proj"]),
        ("L18\ndown_proj", telemetry["layers"]["18"]["down_proj"]),
        ("L21\ndown_proj", telemetry["layers"]["21"]["down_proj"]),
        ("L22\ndown_proj", telemetry["layers"]["22"]["down_proj"]),
        ("L23\ndown_proj\n(SwiGLU)", telemetry["layers"]["23"]["down_proj"]),
    ]

    labels = [c[0] for c in cases]
    v_all = [c[1]["all_tokens"]["max_abs"] for c in cases]
    v_nb = [c[1]["no_bos"]["max_abs"] for c in cases]
    r_all = [c[1]["all_tokens"]["outlier_ratio"] for c in cases]
    r_nb = [c[1]["no_bos"]["outlier_ratio"] for c in cases]

    fig, ax = plt.subplots(figsize=(11.2, 5.0))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)

    x = np.arange(len(cases))
    width = 0.36

    b1 = ax.bar(x - width / 2, v_all, width, color="#f43f5e", edgecolor="#0f172a", label="all_tokens (includes token 0 / BOS attention sink)")
    b2 = ax.bar(x + width / 2, v_nb, width, color="#38bdf8", edgecolor="#0f172a", label="no_bos (real sequence tokens 1..T only)")

    ax.set_yscale("log")
    ax.set_ylim(2.0, 5500.0)

    for bar, ratio in zip(b1, r_all):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.14,
            f"{h:.1f}\n({ratio:.1f}x)",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#fda4af",
        )

    for bar, ratio in zip(b2, r_nb):
        h = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h * 1.14,
            f"{h:.1f}\n({ratio:.1f}x)",
            ha="center",
            va="bottom",
            fontsize=8.6,
            fontweight="bold",
            color="#7dd3fc",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9.8, fontweight="bold")
    ax.set_ylabel("Peak Activation max(|X|) [Log Scale]", color="#e2e8f0", fontsize=11, fontweight="bold", labelpad=8)

    ax.set_title(
        "Initial-Token Attention Sink (all_tokens) vs. Persistent SwiGLU Spikes (no_bos)\n"
        "Excluding token 0 eliminates BOS sinks (L6 up_proj, L21 spike), leaving structural SwiGLU outliers up to 119.6 (12.6x)",
        color=TEXT_MAIN,
        fontsize=12,
        fontweight="bold",
        loc="center",
        pad=38,
    )

    leg = ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.115),
        ncol=2,
        frameon=True,
        facecolor=BG_DARK,
        edgecolor="#334155",
        fontsize=9.5,
        labelcolor="#e2e8f0",
    )
    leg.get_frame().set_linewidth(1.0)

    _save(fig, "bos_sink_vs_swiglu")


def plot_recipe_footprint() -> None:
    recipes = [
        ("E0  (FP32 Baseline)", 1404.5, "1.0x  |  FP32 weights & activations", "#64748b"),
        ("E1  (W8A8 Fallback)", 351.1, "4.0x  |  8-bit sym W / 8-bit asym A", "#38bdf8"),
        ("E2b (W8A16 Reference)", 351.1, "4.0x  |  8-bit sym W / 16-bit asym A", "#0ea5e9"),
        ("E2  (W4A8 Naive)", 175.6, "8.0x  |  4-bit sym W / 8-bit asym A", "#f97316"),
        ("E2a (W4A16 Diagnostic)", 175.6, "8.0x  |  4-bit sym W / 16-bit asym A", "#f59e0b"),
        ("E4  (W4A8 + Smooth α=0.5)", 175.6, "8.0x  |  3-Boundary SmoothQuant (fixed α=0.5)", "#10b981"),
        ("E4s (W4A8 + Smooth α* Search)", 175.6, "8.0x  |  3-Boundary SmoothQuant (per-layer MSE α*)", "#22c55e"),
    ][::-1]

    names = [r[0] for r in recipes]
    mbs = [r[1] for r in recipes]
    notes = [r[2] for r in recipes]
    colors = [r[3] for r in recipes]

    fig, ax = plt.subplots(figsize=(11.2, 4.5))
    fig.patch.set_facecolor(BG_DARK)
    _style_ax(ax)
    ax.grid(False, axis="y")
    ax.grid(True, linestyle="--", linewidth=0.7, color=GRID_COLOR, alpha=0.7, axis="x")

    y = np.arange(len(recipes))
    bars = ax.barh(y, mbs, height=0.58, color=colors, edgecolor="#0f172a")

    for bar, mb, note in zip(bars, mbs, notes):
        w = bar.get_width()
        ax.text(
            w + 24,
            bar.get_y() + bar.get_height() / 2,
            f"{mb:.1f} MB   ({note})",
            va="center",
            ha="left",
            fontsize=9.5,
            fontweight="bold",
            color="#e2e8f0",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=10, fontweight="bold", color="#f1f5f9")
    ax.set_xlim(0, 2500)
    ax.set_xlabel("Linear Weight Footprint (MB)", color="#e2e8f0", fontsize=11, fontweight="bold", labelpad=8)
    ax.set_title(
        "Quantization Recipe Registry: Linear Weight Footprint & Compression Ratio (E0–E4s)\n"
        "3-boundary SmoothQuant (E4, E4s) folds channel scales into adjacent weights offline—zero runtime memory overhead",
        color=TEXT_MAIN,
        fontsize=11.8,
        fontweight="bold",
        loc="center",
        pad=14,
    )

    _save(fig, "recipe_footprint")


def plot_silicon_validation(npu_summary: list[dict]) -> None:
    by_var = {d["variant"]: d for d in npu_summary}
    variants = ["fp16", "w8a16", "w4a16"]
    latencies = [by_var[v]["latency_median_ms"] for v in variants]
    rams = [by_var[v]["peak_memory_mb"] for v in variants]
    sqnrs = [by_var[v]["sqnr_sim_vs_silicon_db"] for v in variants]

    fig, axes = plt.subplots(1, 3, figsize=(11.4, 4.4))
    fig.patch.set_facecolor(BG_DARK)
    fig.subplots_adjust(wspace=0.30, top=0.74)

    fig.suptitle(
        "Silicon Validation on Samsung Galaxy S24 (Snapdragon 8 Gen 3 Hexagon v75 NPU)\n"
        "100% NPU operator offload across all variants; ~44 dB simulation-vs-silicon SQNR confirms PyTorch/ORT fidelity",
        color=TEXT_MAIN,
        fontsize=12,
        fontweight="bold",
        x=0.5,
        ha="center",
        y=0.96,
    )

    bar_colors = ["#64748b", "#38bdf8", "#22c55e"]

    # Panel 1: Latency
    ax0 = axes[0]
    _style_ax(ax0)
    b0 = ax0.bar(variants, latencies, width=0.52, color=bar_colors, edgecolor="#0f172a")
    for bar, v in zip(b0, latencies):
        ax0.text(bar.get_x() + bar.get_width() / 2, v + 0.04, f"{v:.3f} ms", ha="center", va="bottom", fontsize=9.6, fontweight="bold", color="#f8fafc")
    ax0.set_ylim(0, 1.38)
    ax0.set_title("Median Latency (ms) ↓\n(w4a16 is 15.1% faster)", color="#f8fafc", fontsize=10.2, fontweight="bold", pad=10)
    ax0.set_ylabel("Milliseconds (ms)", color="#cbd5e1", fontsize=10)

    # Panel 2: Peak RAM
    ax1 = axes[1]
    _style_ax(ax1)
    b1 = ax1.bar(variants, rams, width=0.52, color=bar_colors, edgecolor="#0f172a")
    for bar, v in zip(b1, rams):
        ax1.text(bar.get_x() + bar.get_width() / 2, v + 6.0, f"{v:.1f} MB", ha="center", va="bottom", fontsize=9.6, fontweight="bold", color="#f8fafc")
    ax1.set_ylim(0, 275)
    ax1.set_title("Peak On-Device RAM (MB) ↓\n(-33.5% memory vs fp16)", color="#f8fafc", fontsize=10.2, fontweight="bold", pad=10)
    ax1.set_ylabel("Megabytes (MB)", color="#cbd5e1", fontsize=10)

    # Panel 3: Sim-vs-Silicon SQNR
    ax2 = axes[2]
    _style_ax(ax2)
    b2 = ax2.bar(variants, sqnrs, width=0.52, color=["#a855f7", "#a855f7", "#a855f7"], edgecolor="#0f172a")
    for bar, v in zip(b2, sqnrs):
        ax2.text(bar.get_x() + bar.get_width() / 2, v + 1.2, f"{v:.2f} dB", ha="center", va="bottom", fontsize=9.6, fontweight="bold", color="#e9d5ff")
    ax2.set_ylim(0, 56)
    ax2.set_title("Sim-vs-Silicon SQNR (dB) ↑\n(ORT QDQ vs Physical NPU)", color="#f8fafc", fontsize=10.2, fontweight="bold", pad=10)
    ax2.set_ylabel("SQNR (dB)", color="#cbd5e1", fontsize=10)

    _save(fig, "silicon_validation_s24")


def main() -> None:
    telemetry = json.loads((REPO_ROOT / "results" / "outlier_telemetry.json").read_text(encoding="utf-8"))
    npu_summary = json.loads((REPO_ROOT / "results" / "npu_probe" / "summary.json").read_text(encoding="utf-8"))

    plot_layer_kurtosis(telemetry)
    plot_bos_sink_vs_swiglu(telemetry)
    plot_recipe_footprint()
    plot_silicon_validation(npu_summary)
    print("Generated 4 matplotlib SVG charts in docs/assets/")


if __name__ == "__main__":
    main()
