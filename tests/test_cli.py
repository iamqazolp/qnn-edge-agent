import json

import torch
from transformers import Qwen2Config, Qwen2ForCausalLM

from qnn_edge_agent.cli import (
    diagnose_results_dir,
    infer_item_scenario,
    load_recipe_registry,
    main,
    quantize_to_artifact,
    recipe_is_supported,
)
from qnn_edge_agent.types import LayerQuant, Recipe


def test_qea_recipes_command(capsys):
    rc = main(["recipes"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "E0" in out
    assert "E4s" in out
    assert "ready" in out

    reg = load_recipe_registry()
    assert all(recipe_is_supported(r) for r in reg.values())
    unsupported = Recipe(
        id="custom_knapsack",
        smooth="none",
        alpha=None,
        default=LayerQuant(weight=None, act=None),
        allocator="sqnr_knapsack",
    )
    assert recipe_is_supported(unsupported) is False


def test_infer_item_scenario_splits_pc_and_iot():
    assert infer_item_scenario({"source": "bfcl_simple", "ground_truth": {"name": "any"}}) == "bfcl_simple"
    assert infer_item_scenario({"source": "iot", "ground_truth": {"name": "git_diff_status"}}) == "pc"
    assert infer_item_scenario({"source": "iot", "ground_truth": {"name": "local_file_search"}}) == "pc"
    assert infer_item_scenario({"source": "iot", "ground_truth": {"name": "sqlite_query"}}) == "pc"
    assert infer_item_scenario({"source": "iot", "ground_truth": {"name": "gpio_set"}}) == "iot"
    assert infer_item_scenario({"source": "iot", "ground_truth": {"name": "mqtt_publish"}}) == "iot"


def test_quantize_to_artifact_writes_manifest_and_calib(tmp_path):
    torch.manual_seed(0)
    cfg = Qwen2Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=100,
    )
    model = Qwen2ForCausalLM(cfg).eval()

    class Tok:
        pad_token = "<p>"
        eos_token = "<e>"
        padding_side = "left"

        def apply_chat_template(self, messages, tools=None, add_generation_prompt=True, tokenize=False):
            return "prompt"

        def __call__(self, texts, padding=True, truncation=True, max_length=32, return_tensors="pt"):
            ids = torch.randint(1, 100, (len(texts), 8))
            return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}

    reg = load_recipe_registry()
    calib_items = [{"messages": [{"role": "user", "content": "test"}]}]
    manifest = quantize_to_artifact(
        model=model,
        tokenizer=Tok(),
        recipe=reg["E4s"],
        calib_items=calib_items,
        output_dir=tmp_path,
        model_id="tiny-qwen",
    )

    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "calib_stats.json").exists()
    assert manifest["recipe"]["id"] == "E4s"
    assert len(manifest["calib_sha256"]) == 64
    assert set(manifest["chosen_alphas"]) == {
        "model.layers.0.A",
        "model.layers.0.B",
        "model.layers.0.C",
    }


def test_diagnose_results_dir_builds_scenario_taxonomy(tmp_path):
    eval_items = [
        {"id": "pc_1", "source": "iot", "ground_truth": {"name": "git_diff_status", "args": {}}},
        {"id": "iot_1", "source": "iot", "ground_truth": {"name": "gpio_set", "args": {"pin": 4, "state": "LOW"}}},
        {"id": "bf_1", "source": "bfcl_simple", "ground_truth": {"name": "calc", "args": {"x": 1}}},
    ]
    e2_payload = {
        "recipe": {"id": "E2"},
        "avg_bits_linear": 4.0,
        "footprint_mb": 315.14,
        "wikitext2_ppl": 18.2,
        "eval": {"unconstrained": {"values_exact": {"rate": 1.0 / 3.0}}},
        "per_item": {
            "unconstrained": [
                {
                    "item_id": "pc_1",
                    "source": "iot",
                    "raw_output": '{"name":"git_diff_status","arguments":{}}',
                    "json_valid": True,
                    "tool_correct": True,
                    "args_schema_ok": True,
                    "values_exact": True,
                },
                {
                    "item_id": "iot_1",
                    "source": "iot",
                    "raw_output": "{broken",
                    "json_valid": False,
                    "tool_correct": False,
                    "args_schema_ok": False,
                    "values_exact": False,
                },
                {
                    "item_id": "bf_1",
                    "source": "bfcl_simple",
                    "raw_output": '{"name":"calc","arguments":{"x":99}}',
                    "json_valid": True,
                    "tool_correct": True,
                    "args_schema_ok": True,
                    "values_exact": False,
                },
            ]
        },
    }
    (tmp_path / "E2.json").write_text(json.dumps(e2_payload), encoding="utf-8")

    summary = diagnose_results_dir(tmp_path, eval_items)
    assert (tmp_path / "failure_taxonomy.json").exists()
    sc = summary["recipes"]["E2"]["scenarios"]["unconstrained"]
    assert sc["scores"]["pc"]["values_exact"]["rate"] == 1.0
    assert sc["scores"]["iot"]["values_exact"]["rate"] == 0.0
    assert sc["taxonomy"]["iot"]["corrupted_json"] == 1
    assert sc["taxonomy"]["bfcl_simple"]["value_mismatch"] == 1
