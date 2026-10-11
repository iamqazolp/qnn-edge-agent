import copy
from pathlib import Path

import numpy as np
import onnx
import onnx.version_converter
import onnxruntime as ort
from onnx import helper, numpy_helper


def quantize_symmetric_per_channel(
    weight: np.ndarray,
    bits: int = 8,
    axis: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    qmin = -(1 << (bits - 1))
    qmax = (1 << (bits - 1)) - 1
    reduce_axis = 0 if axis == 1 else 1

    max_val = np.max(np.abs(weight), axis=reduce_axis)
    scale = np.maximum(max_val / float(qmax), 1e-8).astype(np.float32)
    zp = np.zeros_like(scale, dtype=np.int8)

    broadcast_scale = scale[None, :] if axis == 1 else scale[:, None]
    w_q = np.clip(np.round(weight / broadcast_scale), qmin, qmax).astype(np.int8)
    return w_q, scale, zp


def quantize_asymmetric_per_tensor(
    data: np.ndarray,
    bits: int = 16,
    rmin: float | None = None,
    rmax: float | None = None,
) -> tuple[float, int]:
    qmin = 0
    qmax = (1 << bits) - 1

    if rmin is None:
        rmin = float(data.min())
    if rmax is None:
        rmax = float(data.max())

    rmin = min(rmin, 0.0)
    rmax = max(rmax, 0.0)

    if rmax == rmin:
        return 1.0, 0

    scale = float((rmax - rmin) / float(qmax - qmin))
    zp = int(np.clip(np.round(-rmin / scale), qmin, qmax))
    return scale, zp


def insert_qdq(
    model: onnx.ModelProto,
    weight_bits: int = 8,
    act_bits: int | None = 16,
    calibration_inputs: dict[str, np.ndarray] | list[dict[str, np.ndarray]] | None = None,
    output_path: str | Path | None = None,
) -> onnx.ModelProto:
    model = copy.deepcopy(model)
    if model.opset_import[0].version < 21:
        model = onnx.version_converter.convert_version(model, 21)
    model.ir_version = 10

    inits = {init.name: init for init in model.graph.initializer}

    act_to_nodes: dict[str, list[onnx.NodeProto]] = {}
    weight_nodes: list[tuple[onnx.NodeProto, str]] = []

    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2:
            if node.input[1] in inits:
                act_to_nodes.setdefault(node.input[0], []).append(node)
                weight_nodes.append((node, node.input[1]))

    calib_ranges: dict[str, tuple[float, float]] = {}
    if act_bits is not None:
        if calibration_inputs is None:
            synth_inputs = {}
            for inp in model.graph.input:
                shape = [
                    dim.dim_value if dim.dim_value > 0 else 1
                    for dim in inp.type.tensor_type.shape.dim
                ]
                synth_inputs[inp.name] = np.random.randn(*shape).astype(np.float32)
            calib_list = [synth_inputs]
        elif isinstance(calibration_inputs, dict):
            calib_list = [calibration_inputs]
        else:
            calib_list = calibration_inputs

        temp_model = copy.deepcopy(model)
        for act_name in act_to_nodes:
            temp_model.graph.output.append(
                helper.make_tensor_value_info(act_name, onnx.TensorProto.FLOAT, None)
            )

        sess_temp = ort.InferenceSession(temp_model.SerializeToString())
        for feed in calib_list:
            outs = sess_temp.run(list(act_to_nodes.keys()), feed)
            for name, val in zip(act_to_nodes.keys(), outs):
                cur_min, cur_max = float(val.min()), float(val.max())
                if name not in calib_ranges:
                    calib_ranges[name] = (cur_min, cur_max)
                else:
                    prev_min, prev_max = calib_ranges[name]
                    calib_ranges[name] = (min(prev_min, cur_min), max(prev_max, cur_max))

    new_inits: list[onnx.TensorProto] = []
    inits_to_remove: set[str] = set()

    act_to_qdq: dict[str, tuple[str, list[onnx.NodeProto]]] = {}
    if act_bits is not None:
        zp_dtype = onnx.TensorProto.UINT8 if act_bits == 8 else onnx.TensorProto.UINT16
        for act_name in act_to_nodes:
            rmin, rmax = calib_ranges[act_name]
            scale, zp = quantize_asymmetric_per_tensor(
                np.array([rmin, rmax], dtype=np.float32),
                bits=act_bits,
                rmin=rmin,
                rmax=rmax,
            )

            scale_name = f"{act_name}_scale"
            zp_name = f"{act_name}_zp"
            q_out_name = f"{act_name}_q"
            dq_out_name = f"{act_name}_dq"

            new_inits.append(helper.make_tensor(scale_name, onnx.TensorProto.FLOAT, [], [scale]))
            new_inits.append(helper.make_tensor(zp_name, zp_dtype, [], [zp]))

            q_node = helper.make_node(
                "QuantizeLinear",
                inputs=[act_name, scale_name, zp_name],
                outputs=[q_out_name],
                name=f"{act_name}_quant",
            )
            dq_node = helper.make_node(
                "DequantizeLinear",
                inputs=[q_out_name, scale_name, zp_name],
                outputs=[dq_out_name],
                name=f"{act_name}_dequant",
            )
            act_to_qdq[act_name] = (dq_out_name, [q_node, dq_node])

    new_nodes: list[onnx.NodeProto] = []
    inserted_acts: set[str] = set()

    for node in model.graph.node:
        if node.op_type in ("MatMul", "Gemm") and len(node.input) >= 2 and node.input[1] in inits:
            act_name = node.input[0]
            if act_bits is not None and act_name in act_to_qdq and act_name not in inserted_acts:
                dq_out, qdq_nodes = act_to_qdq[act_name]
                new_nodes.extend(qdq_nodes)
                inserted_acts.add(act_name)

            if act_bits is not None and act_name in act_to_qdq:
                node.input[0] = act_to_qdq[act_name][0]

            w_name = node.input[1]
            w_onnx = numpy_helper.to_array(inits[w_name])
            axis = 1

            w_q, scale, zp = quantize_symmetric_per_channel(w_onnx, bits=weight_bits, axis=axis)
            w_q_cont = np.ascontiguousarray(w_q)
            scale_cont = np.ascontiguousarray(scale)
            zp_cont = np.ascontiguousarray(zp)

            w_on_grid = (w_q_cont.astype(np.float32) * scale_cont[None, :]).astype(np.float32)

            w_grid_name = f"{w_name}_grid"
            w_scale_name = f"{w_name}_scale"
            w_zp_name = f"{w_name}_zp"
            w_q_name = f"{w_name}_q"
            w_dq_name = f"{w_name}_dq"

            w_grid_proto = numpy_helper.from_array(w_on_grid, name=w_grid_name)
            scale_proto = numpy_helper.from_array(scale_cont, name=w_scale_name)

            if weight_bits == 4:
                zp_proto = helper.make_tensor(
                    w_zp_name,
                    onnx.TensorProto.INT4,
                    [len(zp_cont)],
                    zp_cont.tolist(),
                )
            else:
                zp_proto = numpy_helper.from_array(zp_cont, name=w_zp_name)

            new_inits.extend([w_grid_proto, scale_proto, zp_proto])

            q_w_node = helper.make_node(
                "QuantizeLinear",
                inputs=[w_grid_name, w_scale_name, w_zp_name],
                outputs=[w_q_name],
                name=f"{w_name}_quant",
                axis=axis,
            )
            dq_w_node = helper.make_node(
                "DequantizeLinear",
                inputs=[w_q_name, w_scale_name, w_zp_name],
                outputs=[w_dq_name],
                name=f"{w_name}_dequant",
                axis=axis,
            )
            new_nodes.extend([q_w_node, dq_w_node])
            node.input[1] = w_dq_name
            inits_to_remove.add(w_name)

        new_nodes.append(node)

    filtered_inits = [init for init in model.graph.initializer if init.name not in inits_to_remove]
    del model.graph.initializer[:]
    model.graph.initializer.extend(filtered_inits)
    model.graph.initializer.extend(new_inits)

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)

    onnx.checker.check_model(model)

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        onnx.save_model(model, str(output_path))

    return model
