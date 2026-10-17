# `qnn-edge-agent` Project Overview

**Target Role:** Qualcomm AI Residency Program (Engineering Track, `Job Req ID: 3097473`)  
**Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (494.0M parameters, 24 decoder blocks, `hidden_size=896`, `intermediate_size=4864`)  
**Hardware Backend:** Qualcomm AI Hub (`qai-hub 0.56.0`) targeting `--target_runtime qnn_dlc` on physical Hexagon NPUs

---

## 1. What This Project Is Aiming At

### Core Research Question
> *"When a function-calling SLM (0.5B to 1.5B) is compressed for local execution on AI PCs, IoT gateways, and phones, how do failures split between syntactic collapse (broken JSON) and semantic corruption (wrong tool or hallucinated arguments), and how much accuracy can runtime grammar constraints recover versus weight-level mathematical repairs (3-boundary SmoothQuant, SQNR Knapsack Mixed-Precision, and AdaRound)?"*

### Three Core Engineering Pillars

1. **On-Device Agentic AI Reliability**
   - Standard perplexity (`WikiText-2 PPL`) masks structural failures in tool-calling agents. A model can lose only ~1 PPL point while flipping a quote, brace, or parameter key that crashes downstream tool execution.
   - We grade every generation through a deterministic 4-tier funnel in [`tool_call_eval.py`](src/qnn_edge_agent/eval/tool_call_eval.py) (`json_valid` -> `tool_correct` -> `args_schema_ok` -> `values_exact`) and compare **Unconstrained** decoding against **Grammar-Constrained** decoding (`lm-format-enforcer`).
   - This isolates how much of low-bit tool-calling degradation is pure syntax breakage (recoverable via logit masking) versus semantic corruption (wrong tool selection or corrupted argument values that only weight/activation repairs can fix).

2. **PyTorch Quantization & GenAI Model Optimization**
   - In `Qwen2.5` SwiGLU MLPs, element-wise multiplication of `gate_proj` and `up_proj` creates extreme activation outliers entering `down_proj` (kurtosis up to `542.6`, max magnitude `119.57`, `measured`). Uniform 8-bit activation quantization stretches the step size $\Delta$ and crushes normal channels toward zero.
   - We implement and ablate three mathematical repairs in PyTorch:
     - **3-Boundary SmoothQuant (`E3`, `E4`, `E4s`):** Exact function-preserving scaling across `input_layernorm -> q/k/v`, `post_attention_layernorm -> gate/up`, and SwiGLU `up_proj -> down_proj` (with per-boundary MSE-searched $\alpha \in [0.1, 0.9]$).
     - **SQNR Knapsack Mixed-Precision (`E5`):** Layer-wise Signal-to-Quantization-Noise Ratio (SQNR) profiling across all 168 decoder linear projections, solved as a parameter-weighted 0-1 knapsack allocation (`W4` vs `W8`) at a `4.6` average bits/weight budget (`341.99 MB`).
     - **Clip Search + Gram-Matrix AdaRound (`E5h`):** Per-channel weight clipping and local L2 output reconstruction (`||WX - \tilde{W}X||_F^2`) on `W4` layers.

3. **Cross-Form-Factor Edge Deployment (AI PC, Industrial IoT, Mobile)**
   - Rather than testing only phones, we compile and profile a representative static-shape Qwen2 decoder block (`(1, 128, 896)`) with explicit ONNX `QuantizeLinear -> DequantizeLinear` (`QDQ`) nodes across three Qualcomm computing form factors via [`ai_hub.py`](src/qnn_edge_agent/npu/ai_hub.py):
     - **AI PC / Laptop:** `Snapdragon X Elite CRD` (Hexagon v73) for local developer workspace tools (`git_status_diff`, `sqlite_local_query`, `workspace_file_search`, `system_process_inspect`).
     - **Industrial / Embedded IoT:** `Dragonwing IQ-9075 EVK` (Hexagon v73) / `Dragonwing RB3 Gen 2 Vision Kit` (Hexagon v68) for edge sensor and actuator tools (`gpio_pin_control`, `mqtt_publish_telemetry`, `modbus_read_register`, `ble_beacon_scan`).
     - **Mobile:** `Samsung Galaxy S24 (Family)` (`SM8650`, Snapdragon 8 Gen 3, Hexagon v75, matching the Xiaomi 14 Pro SoC) for on-device assistant tools (`camera_capture_frame`, `network_interface_config`, `schedule_local_alarm`, `thermal_throttle_policy`).

---

## 2. What This Project Does (System Architecture)

```text
qnn-edge-agent/
├── configs/
│   ├── edge_tools_schema.json         # 12 cross-form-factor tools (4 AI PC, 4 IoT, 4 Mobile/System)
│   └── experiments.yaml               # Declarative recipe registry (E0 through E5h)
├── data/
│   ├── calibration_128.jsonl          # 128 prompt-only calibration items (disjoint from eval)
│   ├── eval_tool_calls.jsonl          # 460 ground-truth tool-calling items (BFCL + AI PC/IoT/Mobile)
│   └── iot_items.json                 # 40 deterministic edge/workspace/IoT evaluation cases
├── src/qnn_edge_agent/
│   ├── env.py                         # Repo root resolution + QUALCOMM_AIHUB_KEY loader (.env)
│   ├── types.py                       # Frozen dataclasses (Recipe, EvalScores, NpuMetrics, etc.)
│   ├── data/
│   │   └── build_datasets.py          # Deterministic dataset builder with SHA-256 disjointness gate
│   ├── eval/
│   │   ├── tool_call_eval.py          # 4-tier JSON/schema/value grader + lm-format-enforcer decoding
│   │   ├── perplexity.py              # Stride-512 WikiText-2 PPL evaluator
│   │   └── stats.py                   # Wilson 95% confidence intervals + McNemar paired exact p-values
│   ├── introspect/
│   │   └── outliers.py                # Per-layer max_abs, outlier ratio, excess kurtosis (all vs no-BOS)
│   ├── quant/
│   │   ├── grids.py                   # Symmetric INT4/INT8 per-channel weight & UINT8/UINT16 act grids
│   │   ├── calibrate.py               # 128-prompt activation range & channel scale collector
│   │   └── fake_quant.py              # PyTorch in-place weight fake-quant + activation pre-hooks
│   └── npu/
│       ├── block_export.py            # Static-shape Qwen2 decoder block ONNX exporter (opset 21)
│       ├── qdq_insert.py              # AIMET-aligned FP32 -> Q(axis=1) -> DQ(axis=1) -> MatMul rewriter
│       └── ai_hub.py                  # Qualcomm AI Hub compile, profile, and on-device inference client
├── scripts/
│   ├── run_e0.py                      # Baseline FP32 evaluation script
│   ├── probe_npu.py                   # Snapdragon 8 Gen 3 Hexagon NPU hardware probe
│   ├── profile_outliers.py            # 128-prompt outlier profiler & 3-panel heatmap generator
│   └── run_ablation_suite.py          # Multi-recipe ablation runner + W4A8 failure diff extractor
└── tests/                             # 37 passing unit tests covering data, eval, ONNX parity, and quant
```

### Ablation Matrix (`configs/experiments.yaml`)

| ID | Recipe | Linear Weights | Activations | Smoothing | Decoding Modes | Linear Avg Bits | Total Footprint (`INT8` Embed) |
|---|---|---|---|---|---|---|---|
| `E0` | FP32 Baseline | FP32 | FP32 | None | Unconstrained + Constrained | 16.0 (FP16 ref) | 988.01 MB |
| `E1` | Uniform `W8A8` | `INT8` sym/ch | `UINT8` asym/tensor | None | Unconstrained | 8.0 | 494.05 MB |
| `E2` | Naive `W4A8` | `INT4` sym/ch | `UINT8` asym/tensor | None | Unconstrained + Constrained | 4.0 | 315.14 MB |
| `E2a` | Naive `W4A16` (Weight-Only Attribution) | `INT4` sym/ch | `UINT16` asym/tensor | None | Unconstrained + Constrained | 4.0 | 315.14 MB |
| `E2b` | Naive `W16A8` (Activation-Only Attribution) | FP32 | `UINT8` asym/tensor | None | Unconstrained | 16.0 | 988.01 MB |
| `E3` | 2-Boundary SmoothQuant `W4A8` | `INT4` sym/ch | `UINT8` asym/tensor | `RMSNorm -> QKV/GateUp` ($\alpha=0.5$) | Unconstrained | 4.0 | 315.14 MB |
| `E4` | 3-Boundary SmoothQuant `W4A8` | `INT4` sym/ch | `UINT8` asym/tensor | `+ SwiGLU Up -> Down` ($\alpha=0.5$) | Unconstrained | 4.0 | 315.14 MB |
| `E4s` | 3-Boundary SmoothQuant (Searched $\alpha^*$) | `INT4` sym/ch | `UINT8` asym/tensor | Per-boundary MSE grid search | Unconstrained + Constrained | 4.0 | 315.14 MB |
| `E5` | SQNR Knapsack Mixed-Precision | `W4/W8` sym/ch | `UINT8` asym/tensor | `3-Boundary` ($\alpha^*$) | Unconstrained + Constrained | 4.6 | 341.99 MB |
| `E5h` | Hardware-Aligned Mixed + AdaRound | `W4/W8` + Clip + AdaRound | `UINT16` asym/tensor | `3-Boundary` ($\alpha^*$) | Unconstrained + Constrained | 4.6 | 341.99 MB |

---

## 3. Current State (Completed Through Day 2)

### Completed Subsystems & Verification Tests
- **Repository & Environment (`4039983`, `c25ee50`):** Package installed in `.venv` (`Python 3.11`, `torch 2.14.1`, `transformers 5.20.2`, `onnx 1.23.2`, `onnxruntime 1.31.0`, `qai-hub 0.56.0`, `lm-format-enforcer`). All **37 pytest unit tests** pass (`measured`).
- **Datasets & Disjointness Gate ([`build_datasets.py`](src/qnn_edge_agent/data/build_datasets.py), `a0af519`):**
  - `data/eval_tool_calls.jsonl` contains **460 items** (`300` `bfcl_simple`, `120` `bfcl_multiple`, `40` `custom_iot` spanning AI PC workspace tools, Industrial IoT tools, and Mobile system tools).
  - `data/calibration_128.jsonl` contains **128 prompt-only calibration items** with zero SHA-256 prompt overlap against the evaluation set (`measured`).
- **4-Tier Evaluator & Grammar Constraints ([`tool_call_eval.py`](src/qnn_edge_agent/eval/tool_call_eval.py), `c25ee50`):**
  - Implemented strict recursive JSON schema and exact value grading plus `lm-format-enforcer` JSON-schema logit processor.
  - On the 20-item `E0` smoke slice ([`results/smoke/E0.json`](results/smoke/E0.json)), `Qwen2.5-0.5B-Instruct` achieved **90.0%** unconstrained exact match (`95.0%` `json_valid`) and **95.0%** grammar-constrained exact match (`100.0%` `json_valid`), with WikiText-2 PPL of **13.59** (`measured`).

### Measured Snapdragon 8 Gen 3 (`SM8650`, Hexagon v75) Silicon Telemetry
We exported layer 0 of `Qwen2.5-0.5B-Instruct` (`(1, 128, 896)`) via [`block_export.py`](src/qnn_edge_agent/npu/block_export.py) and [`qdq_insert.py`](src/qnn_edge_agent/npu/qdq_insert.py), then compiled, profiled, and ran on-device inference on a physical `Samsung Galaxy S24 (Family)` (`SM8650`, same SoC as Xiaomi 14 Pro) via Qualcomm AI Hub ([`results/npu_probe/summary.json`](results/npu_probe/summary.json)):

| Variant | Compile / Profile / Inference Jobs | Median Latency | Peak RAM | NPU Offload | On-Device SQNR vs FP32 Torch | Sim-vs-Silicon SQNR (ORT vs NPU) |
|---|---|---|---|---|---|---|
| `fp16` | `jpxrn8d3p` / `jgnqld7kg` / `jp8r0d4kg` | **0.736 ms** | 224.55 MB | **100%** (61/61 ops) | **44.94 dB** | **44.94 dB** |
| `w8a16` | `jp37vlw3g` / `jgowk74qp` / `j5wz20435` | **1.113 ms** | 188.54 MB | **100%** (74/74 ops) | **36.92 dB** | **44.16 dB** |
| `w4a16` | `jgk07w2n5` / `jprd8mx7g` / `j5wz20rm5` | **0.945 ms** | **149.23 MB** | **100%** (74/74 ops) | **12.78 dB** | **44.33 dB** |

- **Key Hardware Finding (`8a13507`, `3398327`):** Standalone `INT4` `DequantizeLinear` initializers in ONNX opset 21 caused `qnn-onnx-converter` to promote the 7 linear weight tensors to external runtime graph inputs (`19.05 ms`). Rewriting weights as grid-aligned `FP32 initializer -> QuantizeLinear(axis=1) -> DequantizeLinear(axis=1) -> MatMul` in [`qdq_insert.py`](src/qnn_edge_agent/npu/qdq_insert.py) allowed QNN to fold all 4-bit weights into `qnn_dlc` constants, cutting latency to **0.945 ms** (`-15.1%` faster than `w8a16`), reducing peak memory to **149.23 MB** (`-33.5%` vs `fp16`), and reaching **44.33 dB** simulation-vs-silicon parity (`measured`).

### Measured SwiGLU Outlier & Kurtosis Telemetry
Running [`profile_outliers.py`](scripts/profile_outliers.py) over all 128 calibration prompts produced [`results/outlier_telemetry.json`](results/outlier_telemetry.json) and [`assets/outlier_profile.png`](assets/outlier_profile.png) (`3398327`, `measured`):
- **`mlp.down_proj` (SwiGLU Output) Has Persistent Extreme Kurtosis:** Excess kurtosis reaches **542.61** in Layer 0, **418.50** in Layer 18, and **335.00** in Layer 23 (where max activation reaches **119.57** and max-to-median channel outlier ratio hits **12.61x**).
- **Not an Initial-Token Sink Artifact:** Excluding token position 0 (`no_bos`) leaves `mlp.down_proj` kurtosis virtually unchanged (`541.28` in Layer 0, `418.00` in Layer 18, `335.62` in Layer 23). By contrast, `mlp.up_proj` in Layer 6 drops from max `46.60` (`12.67x` ratio) with token 0 to `19.90` (`5.41x` ratio) without token 0. This directly justifies Boundary C (`up_proj -> down_proj`) smoothing across all sequence positions.

---

## 4. Remaining Roadmap (Days 3 to 7)

- **Immediate Calibration Patch (Before Colab `E0..E2b` Full Run):**
  - In [`calibrate.py`](src/qnn_edge_agent/quant/calibrate.py), filter out left-padding positions (`attention_mask == 0`) when collecting activation samples and prevent `find_best_mse_range` from clipping `0.99` quantile outlier channels on `16-bit` activation recipes (`E2a`).
- **Day 3 (3-Boundary SmoothQuant & $\alpha$ Search):**
  - Implement [`src/qnn_edge_agent/quant/smooth.py`](src/qnn_edge_agent/quant/smooth.py) (Boundaries A, B, and C in FP32 with FP64 equivalence tests in `tests/test_smooth_equivalence.py`), run `scripts/sweep_alpha.py` (`assets/alpha_sweep_curve.png`), and evaluate `E3`, `E4`, `E4s`.
- **Day 4 (SQNR Knapsack Mixed-Precision & AdaRound):**
  - Implement `src/qnn_edge_agent/introspect/sqnr.py` (`assets/sqnr_heatmap.png`), parameter-weighted 0-1 knapsack bit allocation in `src/qnn_edge_agent/quant/allocate.py`, and weight clip search + Gram-matrix AdaRound in `src/qnn_edge_agent/quant/adaround.py` (`E5`, `E5h`).
- **Day 5 (3-Form-Factor Qualcomm NPU Sweep):**
  - Implement `scripts/sweep_devices.py` to compile and profile `E0` (`fp16`), `E1` (`w8a8`), `E4s` (smoothed `w4a8`), and `E5h` (mixed `w4/w8-a16`) across **AI PC** (`Snapdragon X Elite CRD`), **Industrial IoT** (`Dragonwing IQ-9075 EVK` / `RB3 Gen 2`), and **Mobile** (`Samsung Galaxy S24 (Family)`).
- **Days 6 to 7 (Diagnostics, Report, and Portfolio Polish):**
  - Generate final publication plots (`scripts/plot_diagnostics.py`), build `notebooks/colab_runner.ipynb` for full 460-item T4 evaluation, and compile the final technical `README.md` (`scripts/make_report.py`).
