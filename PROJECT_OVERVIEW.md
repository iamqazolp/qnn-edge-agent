# `qnn-edge-agent` Project Overview

**Target Role:** Qualcomm AI Residency Program (Engineering Track, `Job Req ID: 3097473`)  
**Base Model:** `Qwen/Qwen2.5-0.5B-Instruct` (494.0M parameters, 24 decoder blocks, `hidden_size=896`, `intermediate_size=4864`)  
**Scope:** Software-first engineering system. Combines a config-driven quantization pipeline (`qea`), a sandboxed local agent runtime for PC and IoT tool execution, a runtime reliability layer (validate -> grammar retry -> model fallback), and a 4-tier regression quality gate. A single Qualcomm AI Hub run on real silicon ([Appendix A](#appendix-a-silicon-validation-galaxy-s24)) confirms simulation fidelity.

---

## 1. System Goal & Engineering Pillars

### One-Sentence Pitch
> *"A local tool-calling agent runtime for PCs and IoT gateways that runs a 4-bit compressed SLM and stays reliable: it validates every tool call, recovers from syntax and schema failures via grammar-constrained retry and model fallback, and ships with a reproducible compression pipeline (`qea`) and regression test suite."*

### Four Core Engineering Subsystems

1. **Compression Pipeline & CLI (`qea quantize`, `qea diagnose`)**
   - In `Qwen2.5` SwiGLU MLPs, element-wise multiplication of `gate_proj` and `up_proj` creates extreme activation outliers entering `down_proj` (kurtosis up to `542.6`, max magnitude `119.57`, `measured`). Uniform 8-bit activation quantization stretches the step size $\Delta$ and crushes normal channels toward zero.
   - We implement a config-driven quantization engine in PyTorch:
     - **3-Boundary SmoothQuant (`E4`, `E4s`):** Exact function-preserving scaling across `input_layernorm -> q/k/v` (Boundary A), `post_attention_layernorm -> gate/up` (Boundary B), and SwiGLU `up_proj -> down_proj` (Boundary C, with per-boundary MSE-searched $\alpha^* \in [0.1, 0.9]$).
     - **Diagnostic Attribution (`E2a` / `E2b`):** Separates weight-only (`W4A16`) vs activation-only (`W16A8`) quantization damage.

2. **Local PC & IoT Agent Runtime (`qea agent`)**
   - The tool registry in [`edge_tools_schema.json`](configs/edge_tools_schema.json) models two local-agent workloads, 6 tools each:
     - **PC / developer workspace:** `local_file_search`, `git_diff_status`, `sqlite_query`, `process_monitor`, `calendar_schedule`, `system_clipboard` (executed in a sandboxed workspace).
     - **IoT / edge gateway:** `mqtt_publish`, `gpio_set`, `set_telemetry_threshold`, `ble_scan`, `thermal_throttle`, `camera_capture` (executed against deterministic in-process device/broker simulators).

3. **Runtime Reliability Layer (Validate -> Retry -> Fallback)**
   - Every generated tool call passes through a deterministic validation and recovery pipeline:
     1. Parse JSON and validate against the offered tool schemas.
     2. On syntax or schema failure, retry generation with **Grammar-Constrained** decoding (`lm-format-enforcer`).
     3. If still invalid, fall back to the `W8A8` (`E1`) model.
   - Structured telemetry records which path succeeded, fallback rate, and added latency.

4. **4-Tier Quality Gate (`qea eval`)**
   - Standard perplexity (`WikiText-2 PPL`) masks structural failures in tool-calling agents.
   - [`tool_call_eval.py`](src/qnn_edge_agent/eval/tool_call_eval.py) grades every generation through a 4-tier funnel (`json_valid` -> `tool_correct` -> `args_schema_ok` -> `values_exact`) with Wilson 95% confidence intervals and McNemar paired exact tests ([`stats.py`](src/qnn_edge_agent/eval/stats.py)), broken down by workload (`pc`, `iot`, `bfcl`).

---

## 2. System Architecture

```text
qnn-edge-agent/
├── configs/
│   ├── edge_tools_schema.json         # 12 local-agent tools (6 PC workspace, 6 IoT/edge)
│   └── experiments.yaml               # Declarative recipe registry (E0 through E4s)
├── data/
│   ├── calibration_128.jsonl          # 128 prompt-only calibration items (disjoint from eval)
│   ├── eval_tool_calls.jsonl          # 460 ground-truth tool-calling items (BFCL + 40 custom PC/IoT)
│   └── iot_items.json                 # 40 deterministic custom PC/IoT evaluation cases
├── src/qnn_edge_agent/
│   ├── cli.py                         # Unified `qea` CLI (recipes, quantize, eval, diagnose, agent)
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
│   │   ├── smooth.py                  # 3-boundary SmoothQuant (A: QKV, B: Gate/Up, C: SwiGLU Down)
│   │   └── fake_quant.py              # Config-driven recipe application + activation pre-hooks
│   └── npu/                           # Appendix A only: silicon validation
│       ├── block_export.py            # Static-shape Qwen2 decoder block ONNX exporter (opset 21)
│       ├── qdq_insert.py              # AIMET-aligned FP32 -> Q(axis=1) -> DQ(axis=1) -> MatMul rewriter
│       └── ai_hub.py                  # Qualcomm AI Hub compile, profile, and on-device inference client
├── scripts/
│   ├── run_e0.py                      # Baseline FP32 evaluation script
│   ├── probe_npu.py                   # Appendix A: single-block Snapdragon NPU validation probe
│   ├── profile_outliers.py            # 128-prompt outlier profiler & 3-panel heatmap generator
│   └── run_ablation_suite.py          # Multi-recipe ablation runner + W4A8 failure diff extractor
└── tests/                             # 51 passing unit tests covering data, eval, ONNX parity, CLI, and quant
```

### Recipe Registry (`configs/experiments.yaml`)

| ID | Recipe | Linear Weights | Activations | Smoothing | Decoding Modes | Linear Avg Bits | Total Footprint (`INT8` Embed) |
|---|---|---|---|---|---|---|---|
| `E0` | FP32 Baseline | FP32 | FP32 | None | Unconstrained + Constrained | 16.0 (FP16 ref) | 988.01 MB |
| `E1` | Uniform `W8A8` | `INT8` sym/ch | `UINT8` asym/tensor | None | Unconstrained | 8.0 | 494.05 MB |
| `E2` | Naive `W4A8` | `INT4` sym/ch | `UINT8` asym/tensor | None | Unconstrained + Constrained | 4.0 | 315.14 MB |
| `E2a` | Naive `W4A16` (Weight-Only Attribution) | `INT4` sym/ch | `UINT16` asym/tensor | None | Unconstrained + Constrained | 4.0 | 315.14 MB |
| `E2b` | Naive `W16A8` (Activation-Only Attribution) | FP32 | `UINT8` asym/tensor | None | Unconstrained | 16.0 | 988.01 MB |
| `E4` | 3-Boundary SmoothQuant `W4A8` | `INT4` sym/ch | `UINT8` asym/tensor | `ABC` ($\alpha=0.5$) | Unconstrained | 4.0 | 315.14 MB |
| `E4s` | 3-Boundary SmoothQuant (Searched $\alpha^*$) | `INT4` sym/ch | `UINT8` asym/tensor | `ABC` (per-boundary MSE search) | Unconstrained + Constrained | 4.0 | 315.14 MB |

---

## 3. Current State

### Completed Subsystems & Verification Tests
- **Repository, Environment & CI:** Package installed in `.venv` (`Python 3.11`, `torch 2.14.1`, `transformers 5.20.2`, `onnx 1.23.2`, `onnxruntime 1.31.0`, `qai-hub 0.56.0`, `lm-format-enforcer`). GitHub Actions runs ruff lint and the full pytest suite (**51 passing tests**) on every push/PR.
- **Datasets & Disjointness Gate ([`build_datasets.py`](src/qnn_edge_agent/data/build_datasets.py)):**
  - `data/eval_tool_calls.jsonl` contains **460 items** (`300` `bfcl_simple`, `120` `bfcl_multiple`, `40` custom items with source label `iot`: `19` PC-workspace and `21` IoT/edge).
  - `data/calibration_128.jsonl` contains **128 prompt-only calibration items** with zero SHA-256 prompt overlap against the evaluation set (`measured`).
- **4-Tier Evaluator & Grammar Constraints ([`tool_call_eval.py`](src/qnn_edge_agent/eval/tool_call_eval.py)):**
  - Implemented strict recursive JSON schema and exact value grading plus `lm-format-enforcer` JSON-schema logit processor.
  - On the 20-item `E0` smoke slice (`results/smoke/E0.json`, git-ignored), `Qwen2.5-0.5B-Instruct` achieved **90.0%** unconstrained exact match (`95.0%` `json_valid`) and **95.0%** grammar-constrained exact match (`100.0%` `json_valid`), with WikiText-2 PPL of **13.59** (`measured`).
- **Quantization Engine & `qea` CLI ([`smooth.py`](src/qnn_edge_agent/quant/smooth.py), [`calibrate.py`](src/qnn_edge_agent/quant/calibrate.py), [`cli.py`](src/qnn_edge_agent/cli.py)):**
  - Masked out left-padding positions (`attention_mask == 0`) during calibration and added unclipped min/max activation ranges for 16-bit recipes (`E2a`).
  - Implemented 3-boundary SmoothQuant (`A`, `B`, and SwiGLU `C`) with per-boundary $\alpha^*$ grid search (`E4`, `E4s`) and unified `qea` CLI (`recipes`, `quantize`, `eval`, `diagnose`).

### Measured SwiGLU Outlier & Kurtosis Telemetry
Running [`profile_outliers.py`](scripts/profile_outliers.py) over all 128 calibration prompts produced [`results/outlier_telemetry.json`](results/outlier_telemetry.json) and [`assets/outlier_profile.png`](assets/outlier_profile.png) (`3398327`, `measured`):
- **`mlp.down_proj` (SwiGLU Output) Has Persistent Extreme Kurtosis:** Excess kurtosis reaches **542.61** in Layer 0, **418.50** in Layer 18, and **335.00** in Layer 23 (where max activation reaches **119.57** and max-to-median channel outlier ratio hits **12.61x**).
- **Not an Initial-Token Sink Artifact:** Excluding token position 0 (`no_bos`) leaves `mlp.down_proj` kurtosis virtually unchanged (`541.28` in Layer 0, `418.00` in Layer 18, `335.62` in Layer 23). By contrast, `mlp.up_proj` in Layer 6 drops from max `46.60` (`12.67x` ratio) with token 0 to `19.90` (`5.41x` ratio) without token 0. This directly justifies Boundary C (`up_proj -> down_proj`) smoothing across all sequence positions.

---

## 4. Engineering Roadmap

- **Phase 0 (Full GPU Baseline & SmoothQuant Evaluation):**
  - Run `E0, E1, E2, E2a, E2b, E4, E4s` across all 460 items on Colab GPU ([`colab_runner.ipynb`](notebooks/colab_runner.ipynb)).
- **Phase 1 (Quantization Engine & CLI — Completed):**
  - 3-boundary SmoothQuant ([`smooth.py`](src/qnn_edge_agent/quant/smooth.py)), config-driven [`apply_recipe`](src/qnn_edge_agent/quant/fake_quant.py), and `qea` CLI ([`cli.py`](src/qnn_edge_agent/cli.py)).
- **Phase 2 (Local PC & IoT Agent Runtime):**
  - Implement `qea agent` with sandboxed PC workspace executors (`local_file_search`, `git_diff_status`, `sqlite_query`, `process_monitor`, `calendar_schedule`, `system_clipboard`) and deterministic in-process IoT simulators (`mqtt_publish`, `gpio_set`, `set_telemetry_threshold`, `ble_scan`, `thermal_throttle`, `camera_capture`).
- **Phase 3 (Runtime Reliability Layer):**
  - Add validate -> grammar-constrained retry -> `W8A8` fallback recovery loop with structured telemetry (recovery path, fallback rate, added latency).
- **Phase 4 (Documentation & Portfolio Polish):**
  - Write production `README.md` with CLI walkthrough, architecture diagram, and engineering trade-off & debugging write-up.

---

## Appendix A: Silicon Validation (Galaxy S24)

A one-time check that the PyTorch/ONNX simulation used throughout this project predicts real NPU behavior. It is not extended to further devices.

We exported layer 0 of `Qwen2.5-0.5B-Instruct` (`(1, 128, 896)`) via [`block_export.py`](src/qnn_edge_agent/npu/block_export.py) and [`qdq_insert.py`](src/qnn_edge_agent/npu/qdq_insert.py), then compiled, profiled, and ran on-device inference on a physical `Samsung Galaxy S24 (Family)` (`SM8650`, Snapdragon 8 Gen 3, Hexagon v75) via Qualcomm AI Hub ([`results/npu_probe/summary.json`](results/npu_probe/summary.json)):

| Variant | Compile / Profile / Inference Jobs | Median Latency | Peak RAM | NPU Offload | On-Device SQNR vs FP32 Torch | Sim-vs-Silicon SQNR (ORT vs NPU) |
|---|---|---|---|---|---|---|
| `fp16` | `jpxrn8d3p` / `jgnqld7kg` / `jp8r0d4kg` | **0.736 ms** | 224.55 MB | **100%** (61/61 ops) | **44.94 dB** | **44.94 dB** |
| `w8a16` | `jp37vlw3g` / `jgowk74qp` / `j5wz20435` | **1.113 ms** | 188.54 MB | **100%** (74/74 ops) | **36.92 dB** | **44.16 dB** |
| `w4a16` | `jgk07w2n5` / `jprd8mx7g` / `j5wz20rm5` | **0.945 ms** | **149.23 MB** | **100%** (74/74 ops) | **12.78 dB** | **44.33 dB** |

- **Simulation fidelity:** Sim-vs-silicon SQNR of ~44 dB across all variants means accuracy effects measured in simulation carry over to the NPU.
- **Deployment detail (`8a13507`, `3398327`):** Standalone `INT4` `DequantizeLinear` initializers in ONNX opset 21 caused `qnn-onnx-converter` to promote the 7 linear weight tensors to external runtime graph inputs (`19.05 ms`). Rewriting weights as grid-aligned `FP32 initializer -> QuantizeLinear(axis=1) -> DequantizeLinear(axis=1) -> MatMul` in [`qdq_insert.py`](src/qnn_edge_agent/npu/qdq_insert.py) allowed QNN to fold all 4-bit weights into `qnn_dlc` constants, cutting latency to **0.945 ms** (`-15.1%` faster than `w8a16`) and reducing peak memory to **149.23 MB** (`-33.5%` vs `fp16`) (`measured`).
