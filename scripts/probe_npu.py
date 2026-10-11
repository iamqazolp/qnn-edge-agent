import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
import qai_hub as hub
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qnn_edge_agent.env import MODEL_ID, NPU_DEVICE, REPO_ROOT, hub_client
from qnn_edge_agent.npu.block_export import (
    Qwen2BlockPrefillWrapper,
    export_prefill_block,
    get_real_block,
)
from qnn_edge_agent.npu.qdq_insert import insert_qdq
from qnn_edge_agent.npu.ai_hub import (
    compile_model,
    parse_profile_results,
    profile_model,
    run_inference,
)


def compute_sqnr(ref: np.ndarray, target: np.ndarray) -> float:
    ref_f = ref.astype(np.float64)
    target_f = target.astype(np.float64)
    noise = ref_f - target_f
    sig_pwr = np.sum(ref_f**2)
    noise_pwr = np.sum(noise**2)
    if noise_pwr == 0.0:
        return float("inf")
    return float(10.0 * np.log10(sig_pwr / noise_pwr))


def get_calibration_data(
    seq_len: int = 128,
    model_id: str = MODEL_ID,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any]:
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        attn_implementation="eager",
        torch_dtype=torch.float32,
    )
    layer0 = model.model.layers[0].eval()

    prompt = (
        "You are an edge agent. Use available tools to read the ambient temperature sensor "
        "and publish the reading via MQTT topic sensors/temperature."
    )
    tokens = tokenizer(
        prompt,
        return_tensors="pt",
        max_length=seq_len,
        padding="max_length",
        truncation=True,
    )
    input_ids = tokens["input_ids"]

    with torch.no_grad():
        h0 = model.model.embed_tokens(input_ids)
        pos = torch.arange(seq_len).unsqueeze(0)
        cos, sin = model.model.rotary_emb(h0, pos)
        wrapper = Qwen2BlockPrefillWrapper(layer0, mask_as_constant=True, seq_len=seq_len).eval()
        y_torch_fp = wrapper(h0, cos, sin).detach().numpy()

    return h0.numpy(), cos.numpy(), sin.numpy(), y_torch_fp, layer0


def probe_variant(
    variant: str,
    onnx_path: Path,
    feed: dict[str, np.ndarray],
    y_torch_fp: np.ndarray,
    client: hub.Client,
    device_name: str,
    output_dir: Path,
    input_specs: dict[str, tuple[tuple[int, ...], str]],
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "variant": variant,
        "device": device_name,
        "status": "UNKNOWN",
        "compile_job_id": None,
        "compile_job_url": None,
        "profile_job_id": None,
        "profile_job_url": None,
        "inference_job_id": None,
        "inference_job_url": None,
        "latency_median_ms": None,
        "latency_min_ms": None,
        "peak_memory_mb": None,
        "npu_percentage": None,
        "compute_units": None,
        "sqnr_vs_fp32_torch_db": None,
        "sqnr_sim_vs_silicon_db": None,
        "path_worked": None,
        "error_message": None,
    }

    sess = ort.InferenceSession(str(onnx_path))
    y_ort = sess.run(None, feed)[0]

    compile_options = "--target_runtime qnn_dlc"
    if variant == "fp16":
        compile_options += " --qnn_options default_graph_htp_precision=FLOAT16"

    model_to_compile: onnx.ModelProto | str | hub.Model = onnx.load(str(onnx_path))
    path_worked = "direct_qdq_qnn_dlc" if variant != "fp16" else "direct_fp16_qnn_dlc"

    print(f"\n--- Probing variant: {variant} on {device_name} ---")
    print(f"Compile options: {compile_options}")

    try:
        cjob = compile_model(
            model=model_to_compile,
            name=f"probe_{variant}_{device_name.split()[0]}",
            device_name=device_name,
            compile_options=compile_options,
            input_specs=input_specs,
            client=client,
        )
        record["compile_job_id"] = cjob.job_id
        record["compile_job_url"] = cjob.url
        print(f"Compile job submitted: {cjob.job_id} ({cjob.url})")

        status = cjob.wait()
        if not status.success:
            err_msg = f"Compile failed: {status.message}"
            print(err_msg)
            raise RuntimeError(err_msg)

        print("Compile succeeded! Target model produced.")
    except Exception as e:
        record["error_message"] = str(e)
        print(f"Direct qnn_dlc compile failed for {variant}: {e}")

        # Fallback 1: try precompiled_qnn_onnx
        try:
            print(f"Attempting fallback 1 (precompiled_qnn_onnx) for {variant}...")
            fb_opts = "--target_runtime precompiled_qnn_onnx"
            if variant == "fp16":
                fb_opts += " --qnn_options default_graph_htp_precision=FLOAT16"
            cjob = compile_model(
                model=model_to_compile,
                name=f"probe_fb1_{variant}",
                device_name=device_name,
                compile_options=fb_opts,
                input_specs=input_specs,
                client=client,
            )
            cstatus = cjob.wait()
            if not cstatus.success:
                raise RuntimeError(f"Fallback precompiled_qnn_onnx failed: {cstatus.message}")
            record["compile_job_id"] = cjob.job_id
            record["compile_job_url"] = cjob.url
            path_worked = "direct_qdq_precompiled_onnx" if variant != "fp16" else "direct_fp16_precompiled_onnx"
            print("Fallback 1 succeeded!")
        except Exception as fb1_err:
            print(f"Fallback 1 failed for {variant}: {fb1_err}")
            if variant in ("w8a16", "w4a16", "w8a8"):
                print(f"Attempting fallback 2 (AI Hub quantize job) for {variant}...")
                try:
                    target_w_dtype = hub.QuantizeDtype.INT4 if "w4" in variant else hub.QuantizeDtype.INT8
                    target_a_dtype = hub.QuantizeDtype.INT16 if "a16" in variant else hub.QuantizeDtype.INT8

                    calib_entries = {k: [v] for k, v in feed.items()}
                    fp_model_path = output_dir / "fp16_block.onnx"

                    qjob = client.submit_quantize_job(
                        model=str(fp_model_path),
                        calibration_data=calib_entries,
                        weights_dtype=target_w_dtype,
                        activations_dtype=target_a_dtype,
                        name=f"quantize_{variant}",
                    )
                    qstatus = qjob.wait()
                    if not qstatus.success:
                        raise RuntimeError(f"AI Hub quantize job failed: {qstatus.message}")

                    quant_model = qjob.get_target_model()
                    cjob = client.submit_compile_job(
                        model=quant_model,
                        device=client.get_devices(name=device_name)[0],
                        name=f"probe_fallback_{variant}",
                        options="--target_runtime qnn_dlc",
                        input_specs=input_specs,
                    )
                    cstatus = cjob.wait()
                    if not cstatus.success:
                        raise RuntimeError(f"Fallback compile failed: {cstatus.message}")

                    record["compile_job_id"] = cjob.job_id
                    record["compile_job_url"] = cjob.url
                    path_worked = "ai_hub_quantize_fallback"
                    record["error_message"] = f"Original direct compile failed ({str(e)}). Fallback succeeded."
                except Exception as fb_err:
                    record["status"] = "FAILED"
                    record["error_message"] = f"Direct failed ({str(e)}), FB1 failed ({str(fb1_err)}), FB2 failed ({str(fb_err)})"
                    return record
            else:
                record["status"] = "FAILED"
                return record

    record["path_worked"] = path_worked

    # Profile job
    try:
        pjob = profile_model(
            compile_job=cjob,
            name=f"profile_{variant}",
            device_name=device_name,
            profile_options="--compute_unit npu",
            client=client,
        )
        record["profile_job_id"] = pjob.job_id
        record["profile_job_url"] = pjob.url
        print(f"Profile job submitted: {pjob.job_id} ({pjob.url})")

        pstatus = pjob.wait()
        if not pstatus.success:
            print(f"Profile job failed: {pstatus.message}")
            record["error_message"] = (record["error_message"] or "") + f" | Profile failed: {pstatus.message}"
        else:
            prof_dict = pjob.download_profile()
            raw_prof_path = output_dir / f"{variant}_profile.json"
            with open(raw_prof_path, "w") as f:
                json.dump(prof_dict, f, indent=2)

            parsed = parse_profile_results(prof_dict)
            record["latency_median_ms"] = parsed["median_latency_ms"]
            record["latency_min_ms"] = parsed["min_latency_ms"]
            record["peak_memory_mb"] = parsed["peak_memory_mb"]
            record["compute_units"] = parsed["compute_units"]
            record["npu_percentage"] = parsed["npu_percentage"]
            print(
                f"Profile parsed: {parsed['median_latency_ms']} ms median, "
                f"{parsed['npu_percentage']}% NPU placement, {parsed['peak_memory_mb']} MB peak"
            )
    except Exception as e:
        print(f"Profile exception: {e}")
        record["error_message"] = (record["error_message"] or "") + f" | Profile exc: {str(e)}"

    # On-device inference job
    try:
        target_model = cjob.get_target_model()
        dev = client.get_devices(name=device_name)[0]
        formatted_inputs = {k: [v] for k, v in feed.items()}

        ijob = client.submit_inference_job(
            model=target_model,
            device=dev,
            inputs=formatted_inputs,
            name=f"infer_{variant}",
            options="--compute_unit npu",
            profile=False,
        )
        record["inference_job_id"] = ijob.job_id
        record["inference_job_url"] = ijob.url
        print(f"Inference job submitted: {ijob.job_id} ({ijob.url})")

        istatus = ijob.wait()
        if not istatus.success:
            print(f"Direct inference job failed: {istatus.message}")
            record["direct_inference_error"] = istatus.message
            
            # Trigger fallback if not already using fallback
            if variant in ("w8a16", "w4a16") and "fallback" not in path_worked:
                print(f"Triggering AI Hub quantize fallback for {variant} due to input mismatch / folding limitation...")
                target_w_dtype = hub.QuantizeDtype.INT4 if "w4" in variant else hub.QuantizeDtype.INT8
                target_a_dtype = hub.QuantizeDtype.INT16 if "a16" in variant else hub.QuantizeDtype.INT8
                calib_entries = {k: [v] for k, v in feed.items()}
                fp_model_path = output_dir / "fp16_block.onnx"

                qjob = client.submit_quantize_job(
                    model=str(fp_model_path),
                    calibration_data=calib_entries,
                    weights_dtype=target_w_dtype,
                    activations_dtype=target_a_dtype,
                    name=f"quantize_{variant}",
                )
                qjob.wait()

                quant_model = qjob.get_target_model()
                fb_cjob = client.submit_compile_job(
                    model=quant_model,
                    device=dev,
                    name=f"probe_fallback_{variant}",
                    options="--target_runtime qnn_dlc",
                    input_specs=input_specs,
                )
                fb_cjob.wait()
                record["fallback_compile_job_id"] = fb_cjob.job_id
                record["fallback_compile_job_url"] = fb_cjob.url

                # Profile fallback
                fb_pjob = client.submit_inference_job(
                    model=fb_cjob.get_target_model(),
                    device=dev,
                    inputs=None,
                    name=f"profile_fallback_{variant}",
                    options="--compute_unit npu",
                    profile=True,
                )
                fb_pjob.wait()
                record["fallback_profile_job_id"] = fb_pjob.job_id
                record["fallback_profile_job_url"] = fb_pjob.url
                fb_prof = parse_profile_results(fb_pjob.download_profile())
                record["fallback_latency_median_ms"] = fb_prof["median_latency_ms"]
                record["fallback_npu_percentage"] = fb_prof["npu_percentage"]
                record["fallback_peak_memory_mb"] = fb_prof["peak_memory_mb"]

                # Inference on fallback
                fb_ijob = client.submit_inference_job(
                    model=fb_cjob.get_target_model(),
                    device=dev,
                    inputs=formatted_inputs,
                    name=f"infer_fallback_{variant}",
                    options="--compute_unit npu",
                    profile=False,
                )
                fb_istatus = fb_ijob.wait()
                if fb_istatus.success:
                    record["fallback_inference_job_id"] = fb_ijob.job_id
                    record["fallback_inference_job_url"] = fb_ijob.url
                    fb_outs = fb_ijob.download_output_data()
                    y_dev_fb = list(fb_outs.values())[0][0]
                    record["fallback_sqnr_vs_fp32_torch_db"] = round(compute_sqnr(y_torch_fp, y_dev_fb), 2)
                    record["fallback_sqnr_sim_vs_silicon_db"] = round(compute_sqnr(y_ort, y_dev_fb), 2)
                    record["path_worked"] = "direct_compile_folded_inputs_ai_hub_quantize_fallback"
                    print(
                        f"Fallback for {variant} succeeded: {fb_prof['median_latency_ms']} ms, "
                        f"{fb_prof['npu_percentage']}% NPU, SQNR: {record['fallback_sqnr_vs_fp32_torch_db']} dB"
                    )
        else:
            output_entries = ijob.download_output_data()
            raw_out_path = output_dir / f"{variant}_inference.json"
            saveable = {k: [arr.tolist() for arr in v] for k, v in output_entries.items()}
            with open(raw_out_path, "w") as f:
                json.dump(saveable, f)

            y_dev = list(output_entries.values())[0][0]
            sqnr_torch = compute_sqnr(y_torch_fp, y_dev)
            sqnr_ort = compute_sqnr(y_ort, y_dev)
            record["sqnr_vs_fp32_torch_db"] = round(sqnr_torch, 2)
            record["sqnr_sim_vs_silicon_db"] = round(sqnr_ort, 2)
            print(f"On-device SQNR vs torch FP32: {sqnr_torch:.2f} dB, vs ORT: {sqnr_ort:.2f} dB")
    except Exception as e:
        print(f"Inference exception: {e}")
        record["error_message"] = (record["error_message"] or "") + f" | Inference exc: {str(e)}"

    record["status"] = "SUCCESS" if (record["compile_job_id"] and record["latency_median_ms"] is not None) else "PARTIAL"
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe NPU execution on Qualcomm AI Hub.")
    parser.add_argument(
        "--device",
        type=str,
        default=NPU_DEVICE,
        help="AI Hub device name (default: Samsung Galaxy S24 (Family)).",
    )
    parser.add_argument(
        "--layer-idx",
        type=int,
        default=0,
        help="Decoder layer index to probe.",
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
        help="Prefill sequence length.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "npu_probe",
        help="Directory to save artifacts and summary.",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = hub_client()

    print(f"Starting NPU hardware probe targeting '{args.device}'...")
    print(f"Extracting layer {args.layer_idx} from {MODEL_ID} and capturing calibration data...")

    h0, cos, sin, y_torch_fp, layer = get_calibration_data(
        seq_len=args.seq_len,
        model_id=MODEL_ID,
    )
    feed = {"hidden_states": h0, "cos": cos, "sin": sin}
    input_specs = {
        "hidden_states": ((1, args.seq_len, h0.shape[-1]), "float32"),
        "cos": ((1, args.seq_len, cos.shape[-1]), "float32"),
        "sin": ((1, args.seq_len, sin.shape[-1]), "float32"),
    }

    # 1. Export FP ONNX block
    fp_onnx_path = args.output_dir / "fp16_block.onnx"
    print(f"Exporting FP prefill block to {fp_onnx_path}...")
    export_prefill_block(
        layer=layer,
        seq_len=args.seq_len,
        output_path=fp_onnx_path,
        opset_version=21,
        mask_as_constant=True,
    )

    # 2. Export W8A16 QDQ block
    w8_onnx_path = args.output_dir / "w8a16_block.onnx"
    print(f"Generating W8A16 QDQ block to {w8_onnx_path}...")
    insert_qdq(
        onnx.load(str(fp_onnx_path)),
        weight_bits=8,
        act_bits=16,
        calibration_inputs=[feed],
        output_path=w8_onnx_path,
    )

    # 3. Export W4A16 QDQ block
    w4_onnx_path = args.output_dir / "w4a16_block.onnx"
    print(f"Generating W4A16 QDQ block to {w4_onnx_path}...")
    insert_qdq(
        onnx.load(str(fp_onnx_path)),
        weight_bits=4,
        act_bits=16,
        calibration_inputs=[feed],
        output_path=w4_onnx_path,
    )

    variants = [
        ("fp16", fp_onnx_path),
        ("w8a16", w8_onnx_path),
        ("w4a16", w4_onnx_path),
    ]

    summary_records = []
    for var_name, var_path in variants:
        rec = probe_variant(
            variant=var_name,
            onnx_path=var_path,
            feed=feed,
            y_torch_fp=y_torch_fp,
            client=client,
            device_name=args.device,
            output_dir=args.output_dir,
            input_specs=input_specs,
        )
        summary_records.append(rec)

    summary_file = args.output_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary_records, f, indent=2)

    print(f"\n================ NPU Probe Complete ================")
    print(f"Summary written to {summary_file}")
    for rec in summary_records:
        print(
            f"Variant: {rec['variant']:<6} | Status: {rec['status']:<7} | "
            f"Latency: {str(rec['latency_median_ms']):<6} ms | "
            f"NPU: {str(rec['npu_percentage']):<5}% | "
            f"SQNR: {str(rec['sqnr_vs_fp32_torch_db'])} dB"
        )


if __name__ == "__main__":
    main()
