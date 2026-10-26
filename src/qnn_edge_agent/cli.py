from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import tempfile
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from qnn_edge_agent.agent.reliability import (
    HuggingFaceToolGenerator,
    InPlaceW4A16FallbackGenerator,
    ReliableToolAgent,
)
from qnn_edge_agent.agent.server import (
    EdgeConsoleService,
    create_http_server,
    get_lan_ip,
)
from qnn_edge_agent.agent.tools import EdgeToolEnvironment, load_edge_tool_schemas
from qnn_edge_agent.env import MODEL_ID, REPO_ROOT
from qnn_edge_agent.eval.perplexity import evaluate_wikitext2_perplexity
from qnn_edge_agent.eval.tool_call_eval import compute_eval_scores, evaluate_tool_calls
from qnn_edge_agent.quant.calibrate import CalibStats, calibrate_model
from qnn_edge_agent.quant.fake_quant import apply_recipe
from qnn_edge_agent.types import EvalMode, ItemOutcome, LayerQuant, QuantFormat, Recipe

PC_TOOL_NAMES: frozenset[str] = frozenset({
    "local_file_search",
    "git_diff_status",
    "sqlite_query",
    "process_monitor",
    "calendar_schedule",
    "system_clipboard",
})

IOT_TOOL_NAMES: frozenset[str] = frozenset({
    "mqtt_publish",
    "gpio_set",
    "set_telemetry_threshold",
    "ble_scan",
    "thermal_throttle",
    "camera_capture",
})


def load_recipe_registry(config_path: Path | None = None) -> dict[str, Recipe]:
    path = config_path or (REPO_ROOT / "configs" / "experiments.yaml")
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)["experiments"]

    registry: dict[str, Recipe] = {}
    for rid, d in raw.items():
        dq = d.get("default", {})
        w_d = dq.get("weight")
        a_d = dq.get("act")
        w_fmt = QuantFormat(**w_d) if w_d else None
        a_fmt = QuantFormat(**a_d) if a_d else None
        lq = LayerQuant(weight=w_fmt, act=a_fmt, adaround=dq.get("adaround", False))
        registry[rid] = Recipe(
            id=d["id"],
            smooth=d.get("smooth", "none"),
            alpha=d.get("alpha"),
            default=lq,
            allocator=d.get("allocator", "uniform"),
            bit_budget=d.get("bit_budget"),
            clip_search=d.get("clip_search", False),
            kv_bits=d.get("kv_bits", 16),
        )
    return registry


def recipe_is_supported(recipe: Recipe) -> bool:
    if recipe.allocator not in ("uniform", None):
        return False
    if recipe.default.adaround or recipe.clip_search:
        return False
    return True


def infer_item_scenario(item: dict[str, Any]) -> str:
    src = item.get("source", "unknown")
    if src in ("bfcl_simple", "bfcl_multiple"):
        return src
    gt = item.get("ground_truth", {})
    tool_name = gt.get("name", "") if isinstance(gt, dict) else ""
    if tool_name in PC_TOOL_NAMES:
        return "pc"
    if tool_name in IOT_TOOL_NAMES:
        return "iot"
    return str(src)


def classify_failure(outcome: dict[str, Any]) -> str | None:
    if outcome.get("values_exact"):
        return None
    if not outcome.get("json_valid"):
        return "corrupted_json"
    if not outcome.get("tool_correct"):
        return "wrong_tool"
    if not outcome.get("args_schema_ok"):
        return "invalid_schema"
    return "value_mismatch"


def compute_scenario_breakdown(
    outcomes_raw: list[dict[str, Any]],
    eval_items_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    buckets: dict[str, list[ItemOutcome]] = {}
    taxonomy: dict[str, dict[str, int]] = {}

    for o in outcomes_raw:
        item_id = o["item_id"]
        raw_item = eval_items_by_id.get(item_id, {"source": o.get("source", "unknown")})
        scenario = infer_item_scenario(raw_item)
        buckets.setdefault(scenario, []).append(
            ItemOutcome(
                item_id=item_id,
                source=o.get("source", "iot"),
                raw_output=o.get("raw_output", ""),
                json_valid=bool(o.get("json_valid")),
                tool_correct=bool(o.get("tool_correct")),
                args_schema_ok=bool(o.get("args_schema_ok")),
                values_exact=bool(o.get("values_exact")),
            )
        )
        tax = taxonomy.setdefault(
            scenario,
            {
                "total": 0,
                "exact": 0,
                "corrupted_json": 0,
                "wrong_tool": 0,
                "invalid_schema": 0,
                "value_mismatch": 0,
            },
        )
        tax["total"] += 1
        fail_mode = classify_failure(o)
        if fail_mode is None:
            tax["exact"] += 1
        else:
            tax[fail_mode] += 1

    scores_by_scenario = {
        scen: dataclasses.asdict(compute_eval_scores(items))
        for scen, items in sorted(buckets.items())
    }
    return {"scores": scores_by_scenario, "taxonomy": taxonomy}


def compute_calib_sha256(calib_items: list[dict[str, Any]]) -> str:
    h = hashlib.sha256()
    for it in calib_items:
        h.update(json.dumps(it, sort_keys=True, ensure_ascii=False).encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def quantize_to_artifact(
    model: torch.nn.Module,
    tokenizer: Any,
    recipe: Recipe,
    calib_items: list[dict[str, Any]],
    output_dir: Path,
    model_id: str = MODEL_ID,
    batch_size: int = 8,
    save_weights: bool = False,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    calib_sha = compute_calib_sha256(calib_items)

    needs_calib = recipe.default.act is not None or recipe.smooth not in ("none", None)
    calib_stats: CalibStats | None = None
    if needs_calib:
        calib_stats = calibrate_model(
            model=model,
            tokenizer=tokenizer,
            calib_items=calib_items,
            batch_size=batch_size,
        )
        with open(output_dir / "calib_stats.json", "w", encoding="utf-8") as f:
            json.dump(calib_stats.to_dict(), f, indent=2)

    apply_info = apply_recipe(model=model, recipe=recipe, calib_stats=calib_stats)
    for h in apply_info["hook_handles"]:
        h.remove()

    if save_weights:
        torch.save(model.state_dict(), output_dir / "state_dict.pt")

    manifest = {
        "recipe": dataclasses.asdict(recipe),
        "model_id": model_id,
        "calib_num_items": len(calib_items),
        "calib_sha256": calib_sha,
        "avg_bits_linear": apply_info["avg_bits_linear"],
        "footprint_mb": apply_info["footprint_mb"],
        "chosen_alphas": apply_info.get("chosen_alphas", {}),
    }
    with open(output_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def cmd_recipes(args: argparse.Namespace) -> int:
    registry = load_recipe_registry(Path(args.config) if args.config else None)
    print(f"{'ID':<6} | {'Weights':<10} | {'Acts':<8} | {'Smooth':<8} | {'Alpha':<8} | {'Status'}")
    print("-" * 62)
    for rid, r in registry.items():
        w_str = f"W{r.default.weight.bits}" if r.default.weight else "FP32"
        a_str = f"A{r.default.act.bits}" if r.default.act else "FP32"
        alp_str = str(r.alpha) if r.alpha is not None else "-"
        status = "ready" if recipe_is_supported(r) else "pending (Phase 4)"
        print(f"{rid:<6} | {w_str:<10} | {a_str:<8} | {r.smooth:<8} | {alp_str:<8} | {status}")
    return 0


def _load_model_and_tokenizer(model_id: str, device: str) -> tuple[Any, Any]:
    try:
        tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
        mod = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32, local_files_only=True
        ).to(device)
    except Exception:
        tok = AutoTokenizer.from_pretrained(model_id)
        mod = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.float32
        ).to(device)
    return mod, tok


def _resolve_device(device_arg: str | None) -> str:
    if device_arg:
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cmd_quantize(args: argparse.Namespace) -> int:
    registry = load_recipe_registry(Path(args.config) if args.config else None)
    if args.recipe not in registry:
        raise KeyError(f"Unknown recipe {args.recipe!r}")
    recipe = registry[args.recipe]

    calib_path = Path(args.calib_file) if args.calib_file else (REPO_ROOT / "data" / "calibration_128.jsonl")
    with open(calib_path, "r", encoding="utf-8") as f:
        calib_items = [json.loads(line) for line in f if line.strip()]
    if args.calib_limit:
        calib_items = calib_items[: args.calib_limit]

    device = _resolve_device(args.device)
    model, tokenizer = _load_model_and_tokenizer(args.model, device)
    out_dir = Path(args.output_dir) if args.output_dir else (REPO_ROOT / "results" / "artifacts" / recipe.id)

    manifest = quantize_to_artifact(
        model=model,
        tokenizer=tokenizer,
        recipe=recipe,
        calib_items=calib_items,
        output_dir=out_dir,
        model_id=args.model,
        batch_size=args.batch_size,
        save_weights=args.save_weights,
    )
    print(
        f"Quantized {recipe.id}: avg_bits={manifest['avg_bits_linear']:.1f}, "
        f"footprint={manifest['footprint_mb']:.2f} MB -> {out_dir / 'manifest.json'}"
    )
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    registry = load_recipe_registry(Path(args.config) if args.config else None)
    recipe_ids = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for rid in recipe_ids:
        if rid not in registry:
            raise KeyError(f"Unknown recipe {rid!r}")

    eval_path = Path(args.eval_file) if args.eval_file else (REPO_ROOT / "data" / "eval_tool_calls.jsonl")
    with open(eval_path, "r", encoding="utf-8") as f:
        all_eval = [json.loads(line) for line in f if line.strip()]
    eval_items = all_eval[: args.limit] if args.limit else all_eval
    items_by_id = {it["id"]: it for it in eval_items}

    calib_path = Path(args.calib_file) if args.calib_file else (REPO_ROOT / "data" / "calibration_128.jsonl")
    with open(calib_path, "r", encoding="utf-8") as f:
        all_calib = [json.loads(line) for line in f if line.strip()]
    calib_items = all_calib[:16] if args.limit else all_calib

    out_dir = (
        Path(args.output_dir)
        if args.output_dir
        else (REPO_ROOT / ("results/smoke" if args.limit else "results"))
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    device = _resolve_device(args.device)
    model, tokenizer = _load_model_and_tokenizer(args.model, device)
    base_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    needs_calib = any(
        registry[rid].default.act is not None or registry[rid].smooth not in ("none", None)
        for rid in recipe_ids
    )
    calib_stats = (
        calibrate_model(model, tokenizer, calib_items, batch_size=args.batch_size, device=device)
        if needs_calib
        else None
    )

    gate_failed = False
    for rid in recipe_ids:
        recipe = registry[rid]
        model.load_state_dict({k: v.to(device) for k, v in base_state.items()})
        apply_info = apply_recipe(model, recipe, calib_stats=calib_stats)

        modes = (
            [m.strip() for m in args.modes.split(",") if m.strip()]
            if args.modes
            else (["unconstrained", "constrained"] if rid in ("E0", "E2", "E2a", "E4s") else ["unconstrained"])
        )
        eval_results: dict[str, Any] = {}
        per_item: dict[str, Any] = {}
        scenarios: dict[str, Any] = {}

        for mode in modes:
            t0 = time.time()
            outcomes, scores = evaluate_tool_calls(
                model=model,
                tokenizer=tokenizer,
                items=eval_items,
                mode=mode,
                device=device,
                batch_size=args.batch_size,
            )
            elapsed = time.time() - t0
            raw_outcomes = [dataclasses.asdict(o) for o in outcomes]
            eval_results[mode] = dataclasses.asdict(scores)
            per_item[mode] = raw_outcomes
            scenarios[mode] = compute_scenario_breakdown(raw_outcomes, items_by_id)
            print(
                f"[{rid}:{mode}] ({elapsed:.1f}s) "
                f"json={scores.json_valid.rate * 100:.1f}% "
                f"exact={scores.values_exact.rate * 100:.1f}%"
            )
            if args.min_exact is not None and scores.values_exact.rate < args.min_exact:
                gate_failed = True
            if args.min_json_valid is not None and scores.json_valid.rate < args.min_json_valid:
                gate_failed = True

        ppl = (
            None
            if args.skip_ppl
            else evaluate_wikitext2_perplexity(model, tokenizer, device=device, batch_size=args.batch_size)
        )
        for h in apply_info["hook_handles"]:
            h.remove()

        payload = {
            "recipe": dataclasses.asdict(recipe),
            "model_id": args.model,
            "avg_bits_linear": apply_info["avg_bits_linear"],
            "footprint_mb": apply_info["footprint_mb"],
            "wikitext2_ppl": ppl,
            "chosen_alphas": apply_info.get("chosen_alphas", {}),
            "eval": eval_results,
            "scenarios": scenarios,
            "per_item": per_item,
        }
        with open(out_dir / f"{rid}.json", "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

    return 1 if gate_failed else 0


def diagnose_results_dir(
    results_dir: Path,
    eval_items: list[dict[str, Any]],
) -> dict[str, Any]:
    items_by_id = {it["id"]: it for it in eval_items}
    summary: dict[str, Any] = {"recipes": {}}

    for path in sorted(results_dir.glob("E*.json")):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        rid = data.get("recipe", {}).get("id", path.stem)
        per_item_modes = data.get("per_item", {})
        mode_breakdown: dict[str, Any] = {}
        for mode, outcomes in per_item_modes.items():
            mode_breakdown[mode] = compute_scenario_breakdown(outcomes, items_by_id)
        summary["recipes"][rid] = {
            "avg_bits_linear": data.get("avg_bits_linear"),
            "footprint_mb": data.get("footprint_mb"),
            "wikitext2_ppl": data.get("wikitext2_ppl"),
            "eval": data.get("eval", {}),
            "scenarios": mode_breakdown,
        }

    out_path = results_dir / "failure_taxonomy.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return summary


def cmd_diagnose(args: argparse.Namespace) -> int:
    results_dir = Path(args.results_dir) if args.results_dir else (REPO_ROOT / "results")
    eval_path = Path(args.eval_file) if args.eval_file else (REPO_ROOT / "data" / "eval_tool_calls.jsonl")
    with open(eval_path, "r", encoding="utf-8") as f:
        eval_items = [json.loads(line) for line in f if line.strip()]

    summary = diagnose_results_dir(results_dir, eval_items)
    print(f"{'Recipe':<7} | {'Mode':<13} | {'Overall':<8} | {'PC':<8} | {'IoT':<8} | {'BFCL-S':<8} | {'BFCL-M':<8}")
    print("-" * 74)
    for rid, info in summary["recipes"].items():
        for mode, scen_data in info["scenarios"].items():
            overall = info["eval"].get(mode, {}).get("values_exact", {}).get("rate", 0.0) * 100
            sc = scen_data["scores"]
            pc = sc.get("pc", {}).get("values_exact", {}).get("rate", 0.0) * 100
            iot = sc.get("iot", {}).get("values_exact", {}).get("rate", 0.0) * 100
            bs = sc.get("bfcl_simple", {}).get("values_exact", {}).get("rate", 0.0) * 100
            bm = sc.get("bfcl_multiple", {}).get("values_exact", {}).get("rate", 0.0) * 100
            print(f"{rid:<7} | {mode:<13} | {overall:6.1f}% | {pc:6.1f}% | {iot:6.1f}% | {bs:6.1f}% | {bm:6.1f}%")
    return 0


def _load_iot_item(item_id: str) -> dict[str, Any]:
    iot_data = json.loads((REPO_ROOT / "data" / "iot_items.json").read_text(encoding="utf-8"))
    for it in iot_data.get("eval", []) + iot_data.get("calibration", []):
        if it["id"] == item_id:
            return dict(it)
    raise KeyError(f"Unknown item_id {item_id!r} in data/iot_items.json")


def cmd_agent(args: argparse.Namespace) -> int:
    ws_dir = Path(args.workspace_dir) if args.workspace_dir else Path(tempfile.mkdtemp(prefix="qea_ws_"))
    env = EdgeToolEnvironment(ws_dir, seed_workspace=True)

    item = _load_iot_item(args.item_id)
    messages = (
        [{"role": "user", "content": args.prompt}]
        if args.prompt
        else item["messages"]
    )
    tools = item["tools"] if not args.prompt else load_edge_tool_schemas()

    if args.offline_demo:
        gt = item["ground_truth"]
        valid_call = json.dumps({"name": gt["name"], "arguments": gt["args"]})

        def primary_gen(_m: list[dict[str, Any]], _t: list[dict[str, Any]], mode: EvalMode) -> str:
            if args.simulate_fault == "corrupted_json" and mode == "unconstrained":
                return '<tool_call>\n{"name": "' + gt["name"] + '", "arguments": {bad_json'
            if args.simulate_fault == "wrong_schema":
                return '<tool_call>\n{"name": "' + gt["name"] + '", "arguments": {"invalid_arg": 999}}\n</tool_call>'
            return f"<tool_call>\n{valid_call}\n</tool_call>"

        def fallback_gen(_m: list[dict[str, Any]], _t: list[dict[str, Any]], _mode: EvalMode) -> str:
            return f"<tool_call>\n{valid_call}\n</tool_call>"

        agent = ReliableToolAgent(
            env=env,
            primary_generator=primary_gen,
            fallback_generator=None if args.fallback_recipe == "none" else fallback_gen,
            enable_grammar_retry=not args.no_grammar_retry,
        )
    else:
        registry = load_recipe_registry(Path(args.config) if args.config else None)
        if args.recipe not in registry:
            raise KeyError(f"Unknown primary recipe {args.recipe!r}")
        primary_recipe = registry[args.recipe]

        device = _resolve_device(args.device)
        primary_model, tokenizer = _load_model_and_tokenizer(args.model, device)
        calib_path = REPO_ROOT / "data" / "calibration_128.jsonl"
        with open(calib_path, "r", encoding="utf-8") as f:
            calib_items = [json.loads(line) for line in f if line.strip()][:16]

        needs_calib = primary_recipe.default.act is not None or primary_recipe.smooth not in ("none", None)
        calib_stats = (
            calibrate_model(primary_model, tokenizer, calib_items, batch_size=4, device=device)
            if needs_calib
            else None
        )
        primary_info = apply_recipe(primary_model, primary_recipe, calib_stats=calib_stats)
        primary_gen_hf = HuggingFaceToolGenerator(primary_model, tokenizer, device=device)

        fallback_gen_hf: Any = None
        fallback_overhead_mb = 0.0
        if args.fallback_recipe and args.fallback_recipe.lower() != "none":
            if args.fallback_recipe not in registry:
                raise KeyError(f"Unknown fallback recipe {args.fallback_recipe!r}")
            fb_recipe = registry[args.fallback_recipe]
            can_promote_inplace = (
                primary_recipe.default.weight is not None
                and fb_recipe.default.weight is not None
                and primary_recipe.default.weight.bits == fb_recipe.default.weight.bits
                and fb_recipe.default.act is not None
                and fb_recipe.default.act.bits == 16
            )
            if can_promote_inplace:
                fallback_gen_hf = InPlaceW4A16FallbackGenerator(
                    primary_generator=primary_gen_hf,
                    hook_handles=primary_info["hook_handles"],
                    active_calib=primary_info.get("active_calib", calib_stats),
                    primary_act_bits=primary_recipe.default.act.bits if primary_recipe.default.act else 8,
                    fallback_act_bits=16,
                )
            else:
                fb_model, _ = _load_model_and_tokenizer(args.model, device)
                fb_info = apply_recipe(fb_model, fb_recipe, calib_stats=calib_stats)
                fallback_gen_hf = HuggingFaceToolGenerator(fb_model, tokenizer, device=device)
                fallback_overhead_mb = float(fb_info["footprint_mb"])

        agent = ReliableToolAgent(
            env=env,
            primary_generator=primary_gen_hf,
            fallback_generator=fallback_gen_hf,
            enable_grammar_retry=not args.no_grammar_retry,
            fallback_memory_overhead_mb=fallback_overhead_mb,
        )

    result = agent.run_turn(messages=messages, tools=tools, execute_tool=True)
    print(json.dumps(result.to_dict(), indent=2))
    return 0 if result.ok else 1


def cmd_serve(args: argparse.Namespace) -> int:
    ws_dir = Path(args.workspace_dir) if args.workspace_dir else None
    svc = EdgeConsoleService(workspace_dir=ws_dir)
    server = create_http_server(host=args.host, port=args.port, service=svc)
    lan_ip = get_lan_ip()
    print("qnn-edge-agent Mobile/Web Edge Console listening on:")
    print(f"  Local:   http://127.0.0.1:{args.port}")
    print(f"  Phone:   http://{lan_ip}:{args.port}  (open on your Xiaomi 14 Pro over Wi-Fi)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qea",
        description="qnn-edge-agent CLI: quantization pipeline, 4-tier evaluation, diagnostics, and reliable agent runtime.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_rec = sub.add_parser("recipes", help="List available quantization recipes.")
    p_rec.add_argument("--config", type=str, default=None, help="Path to experiments.yaml")
    p_rec.set_defaults(func=cmd_recipes)

    p_q = sub.add_parser("quantize", help="Calibrate and apply a quantization recipe, emitting a manifest.")
    p_q.add_argument("--recipe", type=str, required=True, help="Recipe ID (e.g. E1, E2, E4, E4s)")
    p_q.add_argument("--model", type=str, default=MODEL_ID, help="Hugging Face model ID")
    p_q.add_argument("--config", type=str, default=None, help="Path to experiments.yaml")
    p_q.add_argument("--calib-file", type=str, default=None, help="Path to calibration JSONL")
    p_q.add_argument("--calib-limit", type=int, default=None, help="Subset of calibration prompts")
    p_q.add_argument("--batch-size", type=int, default=8, help="Calibration batch size")
    p_q.add_argument("--device", type=str, default=None, help="Execution device")
    p_q.add_argument("--output-dir", type=str, default=None, help="Artifact output directory")
    p_q.add_argument("--save-weights", action="store_true", help="Also save quantized state_dict.pt")
    p_q.set_defaults(func=cmd_quantize)

    p_ev = sub.add_parser("eval", help="Run 4-tier evaluation and optional quality-gate checks.")
    p_ev.add_argument("--recipes", type=str, default="E0,E1,E2,E4,E4s", help="Comma-separated recipe IDs")
    p_ev.add_argument("--model", type=str, default=MODEL_ID, help="Hugging Face model ID")
    p_ev.add_argument("--config", type=str, default=None, help="Path to experiments.yaml")
    p_ev.add_argument("--eval-file", type=str, default=None, help="Path to evaluation JSONL")
    p_ev.add_argument("--calib-file", type=str, default=None, help="Path to calibration JSONL")
    p_ev.add_argument("--limit", type=int, default=None, help="Limit number of eval items")
    p_ev.add_argument("--modes", type=str, default=None, help="Comma-separated decoding modes")
    p_ev.add_argument("--skip-ppl", action="store_true", help="Skip WikiText-2 perplexity")
    p_ev.add_argument("--batch-size", type=int, default=8, help="Batch size")
    p_ev.add_argument("--device", type=str, default=None, help="Execution device")
    p_ev.add_argument("--output-dir", type=str, default=None, help="Output directory")
    p_ev.add_argument("--min-exact", type=float, default=None, help="Quality gate: minimum exact match rate [0..1]")
    p_ev.add_argument("--min-json-valid", type=float, default=None, help="Quality gate: minimum valid JSON rate [0..1]")
    p_ev.set_defaults(func=cmd_eval)

    p_diag = sub.add_parser("diagnose", help="Build per-scenario (PC vs IoT vs BFCL) breakdown and failure taxonomy.")
    p_diag.add_argument("--results-dir", type=str, default=None, help="Directory containing recipe JSON results")
    p_diag.add_argument("--eval-file", type=str, default=None, help="Path to eval_tool_calls.jsonl")
    p_diag.set_defaults(func=cmd_diagnose)

    p_ag = sub.add_parser("agent", help="Run a local PC/IoT tool-calling turn with validate-retry-fallback reliability.")
    p_ag.add_argument("--prompt", type=str, default=None, help="User prompt (overrides --item-id)")
    p_ag.add_argument("--item-id", type=str, default="iot_eval_00", help="Item ID from data/iot_items.json")
    p_ag.add_argument("--recipe", type=str, default="E4s", help="Primary quantized recipe ID")
    p_ag.add_argument(
        "--fallback-recipe",
        type=str,
        default="E2a",
        help="Fallback recipe ID (default E2a for zero-RAM in-place W4A16 promotion, E1 for W8A8, or 'none')",
    )
    p_ag.add_argument("--no-grammar-retry", action="store_true", help="Disable grammar-constrained retry stage")
    p_ag.add_argument("--workspace-dir", type=str, default=None, help="Sandboxed workspace directory for PC tools")
    p_ag.add_argument("--model", type=str, default=MODEL_ID, help="Hugging Face model ID")
    p_ag.add_argument("--config", type=str, default=None, help="Path to experiments.yaml")
    p_ag.add_argument("--device", type=str, default=None, help="Execution device")
    p_ag.add_argument("--offline-demo", action="store_true", help="Run deterministic offline reliability demo")
    p_ag.add_argument(
        "--simulate-fault",
        type=str,
        choices=["none", "corrupted_json", "wrong_schema"],
        default="none",
        help="Simulate a quantization fault in --offline-demo mode to demonstrate recovery",
    )
    p_ag.set_defaults(func=cmd_agent)

    p_srv = sub.add_parser("serve", help="Launch mobile/web Edge Console for live phone or browser demos.")
    p_srv.add_argument("--host", type=str, default="0.0.0.0", help="Bind address (default 0.0.0.0 for LAN/phone access)")
    p_srv.add_argument("--port", type=int, default=8080, help="HTTP port (default 8080)")
    p_srv.add_argument("--workspace-dir", type=str, default=None, help="Sandboxed workspace directory")
    p_srv.set_defaults(func=cmd_serve)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
