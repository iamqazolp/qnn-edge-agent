from pathlib import Path
from typing import Any

import numpy as np
import onnx
import qai_hub as hub

from qnn_edge_agent.env import NPU_DEVICE, hub_client


def compile_model(
    model: onnx.ModelProto | str | Path,
    name: str,
    device_name: str = NPU_DEVICE,
    compile_options: str = "--target_runtime qnn_dlc",
    input_specs: dict[str, tuple[tuple[int, ...], str]] | None = None,
    client: hub.Client | None = None,
) -> hub.CompileJob:
    if client is None:
        client = hub_client()

    devices = client.get_devices(name=device_name)
    if not devices:
        raise ValueError(f"Device '{device_name}' not found on Qualcomm AI Hub.")
    device = devices[0]

    model_to_upload = model
    if isinstance(model, (str, Path)):
        model_to_upload = onnx.load(str(model))

    return client.submit_compile_job(
        model=model_to_upload,
        device=device,
        name=name,
        options=compile_options,
        input_specs=input_specs,
    )


def profile_model(
    compile_job: hub.CompileJob | hub.Model,
    name: str,
    device_name: str = NPU_DEVICE,
    profile_options: str = "--compute_unit npu",
    client: hub.Client | None = None,
) -> hub.InferenceJob:
    if client is None:
        client = hub_client()

    devices = client.get_devices(name=device_name)
    if not devices:
        raise ValueError(f"Device '{device_name}' not found on Qualcomm AI Hub.")
    device = devices[0]

    if isinstance(compile_job, hub.CompileJob):
        target_model = compile_job.get_target_model()
        if target_model is None:
            raise RuntimeError(f"CompileJob {compile_job.job_id} produced no target model.")
    else:
        target_model = compile_job

    return client.submit_inference_job(
        model=target_model,
        device=device,
        inputs=None,
        name=name,
        options=profile_options,
        profile=True,
    )


def run_inference(
    target_model: hub.CompileJob | hub.Model,
    inputs: dict[str, np.ndarray | list[np.ndarray]],
    name: str,
    device_name: str = NPU_DEVICE,
    inference_options: str = "--compute_unit npu",
    client: hub.Client | None = None,
) -> dict[str, np.ndarray]:
    if client is None:
        client = hub_client()

    devices = client.get_devices(name=device_name)
    if not devices:
        raise ValueError(f"Device '{device_name}' not found on Qualcomm AI Hub.")
    device = devices[0]

    if isinstance(target_model, hub.CompileJob):
        model = target_model.get_target_model()
        if model is None:
            raise RuntimeError(f"CompileJob {target_model.job_id} produced no target model.")
    else:
        model = target_model

    dataset_inputs: dict[str, list[np.ndarray]] = {}
    for k, v in inputs.items():
        if isinstance(v, np.ndarray):
            dataset_inputs[k] = [v]
        else:
            dataset_inputs[k] = list(v)

    job = client.submit_inference_job(
        model=model,
        device=device,
        inputs=dataset_inputs,
        name=name,
        options=inference_options,
        profile=False,
    )

    status = job.wait()
    if not status.success:
        raise RuntimeError(f"Inference job {job.job_id} failed: {status.message}")

    output_dataset = job.download_output_data()
    if output_dataset is None:
        raise RuntimeError(f"Inference job {job.job_id} returned no output data.")

    result: dict[str, np.ndarray] = {}
    for key, val_list in output_dataset.items():
        if len(val_list) == 1:
            result[key] = val_list[0]
        else:
            result[key] = np.array(val_list)
    return result


def parse_profile_results(profile_dict: dict[str, Any]) -> dict[str, Any]:
    summary = profile_dict.get("execution_summary", {})
    details = profile_dict.get("execution_detail", [])

    est_inf_time = summary.get("estimated_inference_time", 0)
    all_times = summary.get("all_inference_times")
    if all_times and len(all_times) > 0:
        median_time_us = float(np.median(all_times))
        min_time_us = float(np.min(all_times))
    else:
        median_time_us = float(est_inf_time)
        min_time_us = float(est_inf_time)

    median_latency_ms = median_time_us / 1000.0
    min_latency_ms = min_time_us / 1000.0

    peak_mem_bytes = summary.get("estimated_inference_peak_memory", 0)
    if peak_mem_bytes is None or peak_mem_bytes < 0:
        peak_mem_bytes = 0
    peak_memory_mb = peak_mem_bytes / (1024.0 * 1024.0)

    breakdown = {"NPU": 0, "GPU": 0, "CPU": 0}
    for layer in details:
        cu = layer.get("compute_unit", "CPU").upper()
        if cu in breakdown:
            breakdown[cu] += 1
        else:
            breakdown[cu] = 1

    total_layers = len(details)
    npu_count = breakdown.get("NPU", 0)
    npu_pct = (npu_count / total_layers * 100.0) if total_layers > 0 else 0.0

    return {
        "median_latency_ms": round(median_latency_ms, 4),
        "min_latency_ms": round(min_latency_ms, 4),
        "peak_memory_mb": round(peak_memory_mb, 4),
        "compute_units": breakdown,
        "npu_percentage": round(npu_pct, 2),
        "total_layers": total_layers,
    }
