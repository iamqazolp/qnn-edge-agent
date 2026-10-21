import argparse
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qnn_edge_agent.env import MODEL_ID, REPO_ROOT
from qnn_edge_agent.introspect.outliers import OutlierProfiler


PLOT_PROJECTIONS = ["q_proj", "o_proj", "up_proj", "down_proj"]
PROJ_STYLES = {
    "q_proj": {"color": "#1f77b4", "marker": "o", "label": "q_proj (attn in)"},
    "o_proj": {"color": "#ff7f0e", "marker": "s", "label": "o_proj (context)"},
    "up_proj": {"color": "#2ca02c", "marker": "^", "label": "up_proj (mlp in)"},
    "down_proj": {"color": "#d62728", "marker": "D", "label": "down_proj (swiglu act)"},
}


def plot_outlier_profile(telemetry: dict[str, Any], output_path: Path) -> None:
    num_layers = telemetry["num_layers"]
    layers_data = telemetry["layers"]
    layer_indices = sorted([int(k) for k in layers_data.keys()])

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax_or_all, ax_or_nobos = axes[0, 0], axes[0, 1]
    ax_kt_all, ax_kt_nobos = axes[1, 0], axes[1, 1]

    for p in PLOT_PROJECTIONS:
        style = PROJ_STYLES.get(p, {"color": "black", "marker": ".", "label": p})
        or_all_vals = [
            layers_data[str(l)][p]["all_tokens"]["outlier_ratio"]
            for l in layer_indices
        ]
        or_nobos_vals = [
            layers_data[str(l)][p]["no_bos"]["outlier_ratio"]
            for l in layer_indices
        ]
        kt_all_vals = [
            layers_data[str(l)][p]["all_tokens"]["kurtosis"]
            for l in layer_indices
        ]
        kt_nobos_vals = [
            layers_data[str(l)][p]["no_bos"]["kurtosis"]
            for l in layer_indices
        ]

        ax_or_all.plot(
            layer_indices,
            or_all_vals,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5,
            label=style["label"],
        )
        ax_or_nobos.plot(
            layer_indices,
            or_nobos_vals,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5,
            label=style["label"],
        )
        ax_kt_all.plot(
            layer_indices,
            kt_all_vals,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5,
            label=style["label"],
        )
        ax_kt_nobos.plot(
            layer_indices,
            kt_nobos_vals,
            color=style["color"],
            marker=style["marker"],
            linewidth=1.8,
            markersize=5,
            label=style["label"],
        )

    ax_or_all.set_title(r"Outlier Ratio across Layers (All Tokens, $t \geq 0$)", fontsize=11, fontweight="bold")
    ax_or_all.set_ylabel("Outlier Ratio (max / median)", fontsize=10)
    ax_or_all.set_xlabel("Transformer Layer Index", fontsize=10)
    ax_or_all.grid(True, linestyle="--", alpha=0.5)
    ax_or_all.legend(frameon=True, fontsize=9)

    ax_or_nobos.set_title(r"Outlier Ratio across Layers (Excluding Token 0, $t \geq 1$)", fontsize=11, fontweight="bold")
    ax_or_nobos.set_ylabel("Outlier Ratio (max / median)", fontsize=10)
    ax_or_nobos.set_xlabel("Transformer Layer Index", fontsize=10)
    ax_or_nobos.grid(True, linestyle="--", alpha=0.5)
    ax_or_nobos.legend(frameon=True, fontsize=9)

    ax_kt_all.set_title(r"Activation Kurtosis across Layers (All Tokens, $t \geq 0$)", fontsize=11, fontweight="bold")
    ax_kt_all.set_ylabel("Pearson Kurtosis", fontsize=10)
    ax_kt_all.set_xlabel("Transformer Layer Index", fontsize=10)
    ax_kt_all.grid(True, linestyle="--", alpha=0.5)
    ax_kt_all.legend(frameon=True, fontsize=9)
    if any(
        max(layers_data[str(l)][p]["all_tokens"]["kurtosis"] for l in layer_indices) > 50
        for p in PLOT_PROJECTIONS
    ):
        ax_kt_all.set_yscale("log")

    ax_kt_nobos.set_title(r"Activation Kurtosis across Layers (Excluding Token 0, $t \geq 1$)", fontsize=11, fontweight="bold")
    ax_kt_nobos.set_ylabel("Pearson Kurtosis", fontsize=10)
    ax_kt_nobos.set_xlabel("Transformer Layer Index", fontsize=10)
    ax_kt_nobos.grid(True, linestyle="--", alpha=0.5)
    ax_kt_nobos.legend(frameon=True, fontsize=9)
    if any(
        max(layers_data[str(l)][p]["no_bos"]["kurtosis"] for l in layer_indices) > 50
        for p in PLOT_PROJECTIONS
    ):
        ax_kt_nobos.set_yscale("log")

    step = 2 if num_layers > 12 else 1
    for ax in axes.flat:
        ax.set_xticks(range(0, num_layers, step))

    fig.suptitle(
        f"{MODEL_ID} Activation Outlier Topology (Calibration Pass)",
        fontsize=13,
        fontweight="bold",
    )
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved plot to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Profile activation outliers on Qwen2.5.")
    parser.add_argument("--model-id", type=str, default=MODEL_ID)
    parser.add_argument("--calib-file", type=str, default=str(REPO_ROOT / "data" / "calibration_128.jsonl"))
    parser.add_argument("--output-json", type=str, default=str(REPO_ROOT / "results" / "outlier_telemetry.json"))
    parser.add_argument("--output-plot", type=str, default=str(REPO_ROOT / "assets" / "outlier_profile.png"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    calib_path = Path(args.calib_file)
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calibration data file: {calib_path}")

    with open(calib_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]

    print(f"Loading {args.model_id} in FP32 on {device}...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    ).to(device)
    model.eval()
    print(f"Model loaded in {time.time() - t0:.2f}s")

    print(f"Profiling outliers across {len(items)} calibration prompts...")
    t0_prof = time.time()

    with OutlierProfiler(model) as profiler:
        for idx, it in enumerate(items):
            prompt = tokenizer.apply_chat_template(
                it["messages"],
                tools=it.get("tools"),
                add_generation_prompt=True,
                tokenize=False,
            )
            tokens = tokenizer(prompt, return_tensors="pt")
            input_ids = tokens["input_ids"].to(device)

            with torch.no_grad():
                model(input_ids)

            if (idx + 1) % 16 == 0 or (idx + 1) == len(items):
                print(f"  Processed [{idx + 1}/{len(items)}] prompts ({time.time() - t0_prof:.1f}s)")

        telemetry = profiler.get_telemetry()

    telemetry["model_id"] = args.model_id
    telemetry["num_prompts"] = len(items)

    json_path = Path(args.output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(telemetry, f, indent=2)
    print(f"Saved outlier telemetry to {json_path}")

    plot_path = Path(args.output_plot)
    plot_outlier_profile(telemetry, plot_path)


if __name__ == "__main__":
    main()
