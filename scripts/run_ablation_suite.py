from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer

from qnn_edge_agent.env import MODEL_ID, REPO_ROOT
from qnn_edge_agent.eval.perplexity import evaluate_wikitext2_perplexity
from qnn_edge_agent.eval.tool_call_eval import evaluate_tool_calls
from qnn_edge_agent.quant.calibrate import CalibStats, calibrate_model
from qnn_edge_agent.quant.fake_quant import apply_recipe
from qnn_edge_agent.types import LayerQuant, QuantFormat, Recipe


def get_git_sha() -> str:
    try:
        res = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=str(REPO_ROOT),
        )
        return res.stdout.strip()
    except Exception:
        return "unknown"


def parse_recipe_dict(d: dict[str, Any]) -> Recipe:
    dq = d.get("default", {})
    w_d = dq.get("weight")
    a_d = dq.get("act")
    w_fmt = QuantFormat(**w_d) if w_d else None
    a_fmt = QuantFormat(**a_d) if a_d else None
    lq = LayerQuant(weight=w_fmt, act=a_fmt, adaround=dq.get("adaround", False))
    return Recipe(
        id=d["id"],
        smooth=d.get("smooth", "none"),
        alpha=d.get("alpha"),
        default=lq,
        allocator=d.get("allocator", "uniform"),
        bit_budget=d.get("bit_budget"),
        clip_search=d.get("clip_search", False),
        kv_bits=d.get("kv_bits", 16),
    )


def extract_w4a8_failures(
    output_dir: Path,
    eval_items: list[dict[str, Any]],
) -> None:
    e0_file = output_dir / "E0.json"
    e2_file = output_dir / "E2.json"
    if not (e0_file.exists() and e2_file.exists()):
        return

    with open(e0_file, "r", encoding="utf-8") as f:
        e0_data = json.load(f)
    with open(e2_file, "r", encoding="utf-8") as f:
        e2_data = json.load(f)

    mode = "unconstrained"
    if mode not in e0_data.get("per_item", {}) or mode not in e2_data.get("per_item", {}):
        available_modes = list(e0_data.get("per_item", {}).keys())
        if not available_modes:
            return
        mode = available_modes[0]

    e0_items = {it["item_id"]: it for it in e0_data["per_item"][mode]}
    e2_items = {it["item_id"]: it for it in e2_data["per_item"][mode]}
    items_by_id = {it["id"]: it for it in eval_items}

    diffs: list[dict[str, Any]] = []
    for item_id, e0_res in e0_items.items():
        if item_id not in e2_items:
            continue
        e2_res = e2_items[item_id]

        if e0_res.get("values_exact") and not e2_res.get("values_exact"):
            if not e2_res.get("json_valid"):
                fail_mode = "corrupted_json"
            elif not e2_res.get("tool_correct"):
                fail_mode = "wrong_tool"
            elif not e2_res.get("args_schema_ok"):
                fail_mode = "invalid_schema"
            else:
                fail_mode = "value_mismatch"

            raw_item = items_by_id.get(item_id, {})
            messages = raw_item.get("messages", [])
            user_prompt = messages[0]["content"] if messages else ""

            diffs.append({
                "item_id": item_id,
                "source": e0_res.get("source", "unknown"),
                "prompt": user_prompt,
                "ground_truth": raw_item.get("ground_truth", {}),
                "failure_mode": fail_mode,
                "e0_output": e0_res.get("raw_output", ""),
                "e0_status": {
                    "json_valid": e0_res.get("json_valid"),
                    "tool_correct": e0_res.get("tool_correct"),
                    "args_schema_ok": e0_res.get("args_schema_ok"),
                    "values_exact": e0_res.get("values_exact"),
                },
                "e2_output": e2_res.get("raw_output", ""),
                "e2_status": {
                    "json_valid": e2_res.get("json_valid"),
                    "tool_correct": e2_res.get("tool_correct"),
                    "args_schema_ok": e2_res.get("args_schema_ok"),
                    "values_exact": e2_res.get("values_exact"),
                },
            })

            if len(diffs) >= 5:
                break

    target_file = output_dir / "w4a8_failure_examples.json"
    with open(target_file, "w", encoding="utf-8") as f:
        json.dump(diffs, f, indent=2, ensure_ascii=False)
    print(f"Extracted {len(diffs)} W4A8 failure diffs to {target_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ablation suite across quantization recipes.")
    parser.add_argument("--ids", type=str, default="E0,E1,E2,E2a,E2b", help="Comma-separated recipe IDs")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval items")
    parser.add_argument("--modes", type=str, default=None, help="Evaluation modes override")
    parser.add_argument("--skip-ppl", action="store_true", help="Skip perplexity calculation")
    parser.add_argument("--force", action="store_true", help="Force rerun even if results exist")
    parser.add_argument("--device", type=str, default=None, help="Execution device")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size for evaluation")
    parser.add_argument("--output-dir", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()

    if args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = REPO_ROOT / ("results/smoke" if args.limit else "results")
    output_dir.mkdir(parents=True, exist_ok=True)

    recipe_ids = [rid.strip() for rid in args.ids.split(",") if rid.strip()]

    experiments_path = REPO_ROOT / "configs" / "experiments.yaml"
    with open(experiments_path, "r", encoding="utf-8") as f:
        exp_registry = yaml.safe_load(f)["experiments"]

    for rid in recipe_ids:
        if rid not in exp_registry:
            raise KeyError(f"Recipe '{rid}' not defined in {experiments_path}")

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    eval_path = REPO_ROOT / "data" / "eval_tool_calls.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        all_eval_items = [json.loads(line) for line in f if line.strip()]

    eval_items = all_eval_items[: args.limit] if args.limit else all_eval_items

    calib_path = REPO_ROOT / "data" / "calibration_128.jsonl"
    if not calib_path.exists():
        raise FileNotFoundError(f"Missing calib dataset: {calib_path}")

    with open(calib_path, "r", encoding="utf-8") as f:
        all_calib_items = [json.loads(line) for line in f if line.strip()]

    to_run = []
    for rid in recipe_ids:
        target_path = output_dir / f"{rid}.json"
        if target_path.exists() and not args.force:
            print(f"Skipping {rid} (already exists at {target_path})")
        else:
            to_run.append(rid)

    if not to_run:
        print("All requested recipe runs exist. Extracting failure examples if available...")
        extract_w4a8_failures(output_dir, eval_items)
        return

    print(f"Loading base model {MODEL_ID} on {device}...", flush=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, local_files_only=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
            local_files_only=True,
        ).to(device)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
        base_model = AutoModelForCausalLM.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.float32,
        ).to(device)

    base_state_dict = {k: v.cpu().clone() for k, v in base_model.state_dict().items()}

    calib_stats: CalibStats | None = None
    needs_calib = any(
        exp_registry[rid].get("default", {}).get("act") is not None or exp_registry[rid].get("smooth", "none") not in ("none", None)
        for rid in to_run
    )

    if needs_calib:
        calib_items_subset = all_calib_items[:16] if args.limit else all_calib_items
        print(f"Calibrating activation ranges over {len(calib_items_subset)} items...", flush=True)
        t0_calib = time.time()
        calib_stats = calibrate_model(
            model=base_model,
            tokenizer=tokenizer,
            calib_items=calib_items_subset,
            max_length=256,
            batch_size=args.batch_size,
            device=device,
        )
        print(f"Calibration completed in {time.time() - t0_calib:.2f}s.", flush=True)

    for rid in to_run:
        print("\n==========================================", flush=True)
        print(f"Running Recipe {rid}", flush=True)
        print("==========================================", flush=True)
        recipe_dict = exp_registry[rid]
        recipe = parse_recipe_dict(recipe_dict)

        base_model.load_state_dict({k: v.to(device) for k, v in base_state_dict.items()})

        apply_info = apply_recipe(
            model=base_model,
            recipe=recipe,
            calib_stats=calib_stats,
        )

        modes: list[str]
        if args.modes is not None:
            modes = [m.strip() for m in args.modes.split(",") if m.strip()]
        else:
            modes = recipe_dict.get("modes", ["unconstrained"])

        eval_results: dict[str, Any] = {}
        per_item_results: dict[str, Any] = {}

        for mode in modes:
            print(f"\n--- Evaluating {rid} under {mode} mode ({len(eval_items)} items) ---")
            t0 = time.time()
            outcomes, scores = evaluate_tool_calls(
                model=base_model,
                tokenizer=tokenizer,
                items=eval_items,
                mode=mode,
                device=device,
                batch_size=args.batch_size,
            )
            elapsed = time.time() - t0
            eval_results[mode] = dataclasses.asdict(scores)
            per_item_results[mode] = [dataclasses.asdict(o) for o in outcomes]

            print(f"Completed {mode} in {elapsed:.2f}s")
            print(f"  JSON valid:      {scores.json_valid.rate * 100:.1f}%")
            print(f"  Tool correct:    {scores.tool_correct.rate * 100:.1f}%")
            print(f"  Args schema ok:  {scores.args_schema_ok.rate * 100:.1f}%")
            print(f"  Exact value:     {scores.values_exact.rate * 100:.1f}%")

        ppl: float | None = None
        if not args.skip_ppl:
            print(f"\nEvaluating WikiText-2 perplexity for {rid}...")
            ppl = evaluate_wikitext2_perplexity(
                model=base_model,
                tokenizer=tokenizer,
                device=device,
                batch_size=args.batch_size,
            )
            print(f"WikiText-2 PPL: {ppl:.2f}")

        for h in apply_info["hook_handles"]:
            h.remove()

        out_data = {
            "recipe": dataclasses.asdict(recipe),
            "model_id": MODEL_ID,
            "git_sha": get_git_sha(),
            "seed": 42,
            "avg_bits_linear": apply_info["avg_bits_linear"],
            "footprint_mb": apply_info["footprint_mb"],
            "wikitext2_ppl": ppl,
            "chosen_alphas": apply_info.get("chosen_alphas", {}),
            "layer_sqnr_db": {},
            "eval": eval_results,
            "per_item": per_item_results,
        }

        target_file = output_dir / f"{rid}.json"
        with open(target_file, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {rid} result to {target_file}")

    extract_w4a8_failures(output_dir, eval_items)


if __name__ == "__main__":
    main()
