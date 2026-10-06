import pytest

from qnn_edge_agent.eval.tool_call_eval import (
    compute_eval_scores,
    parse_tool_call,
    score_item_output,
)
from qnn_edge_agent.types import ItemOutcome

SAMPLE_TOOL = {
    "type": "function",
    "function": {
        "name": "calculate_area",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number"},
                "height": {"type": "number"},
                "unit": {"type": "string"},
            },
            "required": ["base", "height"],
        },
    },
}

BASE_ITEM = {
    "id": "item_001",
    "source": "bfcl_simple",
    "tools": [SAMPLE_TOOL],
    "ground_truth": {
        "name": "calculate_area",
        "args": {
            "base": [10, 10.0],
            "height": [5, 5.0],
            "unit": ["units", ""],
        },
    },
}


def test_tier1_failure_invalid_json():
    raw = "I'm sorry, I cannot help with that."
    res = score_item_output(BASE_ITEM, raw)
    assert not res.json_valid
    assert not res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_tier1_failure_malformed_tool_call():
    raw = "<tool_call>{name: invalid json}</tool_call>"
    res = score_item_output(BASE_ITEM, raw)
    assert not res.json_valid
    assert not res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_tier2_failure_wrong_tool():
    raw = '<tool_call>\n{"name": "wrong_function", "arguments": {"base": 10, "height": 5}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert not res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_tier3_failure_missing_required_arg():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": 10}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_tier3_failure_wrong_type():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": "ten", "height": 5}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_tier3_failure_unknown_argument():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": 10, "height": 5, "unknown_arg": 1}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_tier4_failure_wrong_value():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": 999, "height": 5}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert res.args_schema_ok
    assert not res.values_exact


def test_all_tiers_pass_exact_match():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": 10, "height": 5, "unit": "units"}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert res.args_schema_ok
    assert res.values_exact


def test_bfcl_optional_arg_omitted():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": 10, "height": 5}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert res.args_schema_ok
    assert res.values_exact


def test_int_float_equality_tolerance():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": 10.0, "height": 5.0}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert res.args_schema_ok
    assert res.values_exact


def test_boolean_is_not_treated_as_numeric_one():
    int_item = {
        "id": "pin_item",
        "source": "iot",
        "tools": [
            {
                "function": {
                    "name": "set_pin",
                    "parameters": {
                        "type": "object",
                        "properties": {"pin": {"type": "integer"}},
                        "required": ["pin"],
                    },
                }
            }
        ],
        "ground_truth": {"name": "set_pin", "args": {"pin": [1]}},
    }
    raw = '<tool_call>\n{"name": "set_pin", "arguments": {"pin": true}}\n</tool_call>'
    res = score_item_output(int_item, raw)
    assert res.json_valid
    assert res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_string_normalization_case_and_punctuation():
    str_item = {
        "id": "poker_item",
        "source": "bfcl_simple",
        "tools": [
            {
                "function": {
                    "name": "play_poker",
                    "parameters": {
                        "type": "object",
                        "properties": {"game": {"type": "string"}},
                        "required": ["game"],
                    },
                }
            }
        ],
        "ground_truth": {
            "name": "play_poker",
            "args": {"game": ["Texas Hold'em"]},
        },
    }
    raw = '<tool_call>\n{"name": "play_poker", "arguments": {"game": "  texas holdem  "}}\n</tool_call>'
    res = score_item_output(str_item, raw)
    assert res.json_valid
    assert res.tool_correct
    assert res.args_schema_ok
    assert res.values_exact


def test_hallucinated_key():
    raw = '<tool_call>\n{"name": "calculate_area", "arguments": {"base": 10, "height": 5, "extra": "hallucinated"}}\n</tool_call>'
    res = score_item_output(BASE_ITEM, raw)
    assert res.json_valid
    assert res.tool_correct
    assert not res.args_schema_ok
    assert not res.values_exact


def test_compute_eval_scores_breakdown():
    outcomes = [
        ItemOutcome("1", "bfcl_simple", "", True, True, True, True),
        ItemOutcome("2", "bfcl_simple", "", True, True, False, False),
        ItemOutcome("3", "iot", "", True, True, True, True),
        ItemOutcome("4", "iot", "", False, False, False, False),
    ]
    scores = compute_eval_scores(outcomes)
    assert scores.json_valid.rate == 0.75
    assert scores.tool_correct.rate == 0.75
    assert scores.args_schema_ok.rate == 0.50
    assert scores.values_exact.rate == 0.50
    assert "bfcl_simple" in scores.by_source
    assert "iot" in scores.by_source
    assert scores.by_source["bfcl_simple"].values_exact.rate == 0.50
    assert scores.by_source["iot"].values_exact.rate == 0.50


def test_parse_tool_call_plain_and_tags():
    assert parse_tool_call('{"name": "f", "arguments": {}}')[0]
    assert parse_tool_call('<tool_call>\n{"name": "f", "arguments": {}}\n</tool_call>')[0]
    assert not parse_tool_call("No JSON here")[0]
    assert not parse_tool_call('{"name": "f"}')[0]
