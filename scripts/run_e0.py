import argparse
import dataclasses
import json
import subprocess
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from qnn_edge_agent.env import MODEL_ID, REPO_ROOT
from qnn_edge_agent.eval.perplexity import evaluate_wikitext2_perplexity
from qnn_edge_agent.eval.tool_call_eval import evaluate_tool_calls
from qnn_edge_agent.types import LayerQuant, Recipe


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run E0 baseline evaluation.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--modes", type=str, default="unconstrained,constrained")
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--skip-ppl", action="store_true")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    if args.output is None:
        args.output = "results/smoke/E0.json" if args.limit else "results/E0.json"

    device = args.device
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    print(f"Loading model {MODEL_ID} in FP32 on {device}...")
    t0_load = time.time()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.float32,
    ).to(device)
    print(f"Model loaded in {time.time() - t0_load:.2f}s")

    eval_path = REPO_ROOT / "data" / "eval_tool_calls.jsonl"
    if not eval_path.exists():
        raise FileNotFoundError(f"Missing eval dataset: {eval_path}")

    with open(eval_path, "r", encoding="utf-8") as f:
        items = [json.loads(line) for line in f if line.strip()]

    if args.limit is not None and args.limit > 0:
        items = items[: args.limit]

    print(f"Evaluating {len(items)} items...")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    eval_results: dict[str, Any] = {}
    per_item_results: dict[str, Any] = {}

    for mode in modes:
        print(f"\n--- Running mode: {mode} ---")
        t0_mode = time.time()
        outcomes, scores = evaluate_tool_calls(
            model=model,
            tokenizer=tokenizer,
            items=items,
            mode=mode,
            device=device,
            batch_size=args.batch_size,
        )
        elapsed = time.time() - t0_mode
        per_item_sec = elapsed / len(items) if items else 0.0

        eval_results[mode] = dataclasses.asdict(scores)
        per_item_results[mode] = [dataclasses.asdict(o) for o in outcomes]

        print(f"Completed {mode} in {elapsed:.2f}s ({per_item_sec * 1000:.1f} ms/item)")
        print(
            f"  JSON valid:      {scores.json_valid.rate * 100:.1f}% [{scores.json_valid.ci_low * 100:.1f}%, {scores.json_valid.ci_high * 100:.1f}%]"
        )
        print(
            f"  Tool correct:    {scores.tool_correct.rate * 100:.1f}% [{scores.tool_correct.ci_low * 100:.1f}%, {scores.tool_correct.ci_high * 100:.1f}%]"
        )
        print(
            f"  Args schema ok:  {scores.args_schema_ok.rate * 100:.1f}% [{scores.args_schema_ok.ci_low * 100:.1f}%, {scores.args_schema_ok.ci_high * 100:.1f}%]"
        )
        print(
            f"  Exact value:     {scores.values_exact.rate * 100:.1f}% [{scores.values_exact.ci_low * 100:.1f}%, {scores.values_exact.ci_high * 100:.1f}%]"
        )

    ppl = None
    if not args.skip_ppl:
        print("\nEvaluating WikiText-2 perplexity...")
        t0_ppl = time.time()
        ppl = evaluate_wikitext2_perplexity(
            model=model,
            tokenizer=tokenizer,
            device=device,
            batch_size=args.batch_size,
        )
        print(f"WikiText-2 PPL: {ppl:.2f} (took {time.time() - t0_ppl:.2f}s)")

    recipe = Recipe(
        id="E0",
        smooth="none",
        alpha=None,
        default=LayerQuant(weight=None, act=None, adaround=False),
        allocator="uniform",
        bit_budget=None,
        clip_search=False,
        kv_bits=16,
    )

    out_file = REPO_ROOT / args.output
    out_file.parent.mkdir(parents=True, exist_ok=True)

    result_data = {
        "recipe": dataclasses.asdict(recipe),
        "model_id": MODEL_ID,
        "git_sha": get_git_sha(),
        "seed": 42,
        "avg_bits_linear": 32.0,
        "footprint_mb": 988.0,
        "wikitext2_ppl": ppl,
        "layer_sqnr_db": {},
        "eval": eval_results,
        "per_item": per_item_results,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(result_data, f, indent=2)

    print(f"\nSaved E0 result to {out_file}")


if __name__ == "__main__":
    main()
