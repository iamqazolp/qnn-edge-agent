from __future__ import annotations

import json
import math
import re
import string
from typing import Any

try:
    import torch
    import transformers
    import transformers.tokenization_utils

    if not hasattr(transformers.tokenization_utils, "PreTrainedTokenizerBase"):
        transformers.tokenization_utils.PreTrainedTokenizerBase = (
            transformers.PreTrainedTokenizerBase
        )

    from lmformatenforcer import (
        CharacterLevelParserConfig,
        JsonSchemaParser,
        SequenceParser,
        StringParser,
        TokenEnforcer,
        UnionParser,
    )
    from lmformatenforcer.consts import COMPLETE_ALPHABET
    from lmformatenforcer.integrations.transformers import (
        TokenEnforcerTokenizerData,
        build_token_enforcer_tokenizer_data,
    )
    from lmformatenforcer.jsonschemaparser import StringParsingState
except ImportError:
    torch = None  # type: ignore[assignment]
    transformers = None  # type: ignore[assignment]
    CharacterLevelParserConfig = None  # type: ignore[assignment,misc]
    COMPLETE_ALPHABET = ""
    JsonSchemaParser = None  # type: ignore[assignment,misc]
    SequenceParser = None  # type: ignore[assignment,misc]
    StringParser = None  # type: ignore[assignment,misc]
    StringParsingState = None  # type: ignore[assignment,misc]
    TokenEnforcer = Any  # type: ignore[assignment,misc]
    TokenEnforcerTokenizerData = Any  # type: ignore[assignment,misc]
    UnionParser = None  # type: ignore[assignment,misc]
    build_token_enforcer_tokenizer_data = None  # type: ignore[assignment]

from qnn_edge_agent.eval.stats import wilson_ci
from qnn_edge_agent.types import EvalMode, EvalScores, ItemOutcome


def clean_string_for_comparison(s: str) -> str:
    pattern = r"[\s" + re.escape(string.punctuation) + r"]+"
    return re.sub(pattern, "", s.lower())


def match_single_value(actual: Any, accepted: Any) -> bool:
    if accepted == "":
        return actual is None or actual == ""

    if isinstance(actual, bool) or isinstance(accepted, bool):
        return (
            isinstance(actual, bool)
            and isinstance(accepted, bool)
            and actual == accepted
        )

    if isinstance(actual, (int, float)) and isinstance(accepted, (int, float)):
        return math.isclose(
            float(actual), float(accepted), rel_tol=1e-5, abs_tol=1e-7
        )

    if isinstance(actual, str) and isinstance(accepted, str):
        if actual == accepted:
            return True
        if actual.strip().lower() == accepted.strip().lower():
            return True
        return clean_string_for_comparison(
            actual
        ) == clean_string_for_comparison(accepted)

    if isinstance(actual, list) and isinstance(accepted, list):
        if len(actual) != len(accepted):
            return False
        return all(match_single_value(a, b) for a, b in zip(actual, accepted))

    if isinstance(actual, dict) and isinstance(accepted, dict):
        if not all(k in accepted for k in actual.keys()):
            return False
        for k, acc_val in accepted.items():
            acc_choices = acc_val if isinstance(acc_val, list) else [acc_val]
            if k not in actual:
                if "" in acc_choices or None in acc_choices:
                    continue
                return False
            if not (
                match_single_value(actual[k], acc_val)
                or any(match_single_value(actual[k], c) for c in acc_choices)
            ):
                return False
        return True

    return actual == accepted


def validate_schema_type(val: Any, schema: dict[str, Any]) -> bool:
    if not isinstance(schema, dict) or not schema:
        return True

    expected_type = schema.get("type")
    if expected_type is None:
        return True

    if expected_type == "integer":
        if not isinstance(val, int) or isinstance(val, bool):
            return False
    elif expected_type == "number":
        if not isinstance(val, (int, float)) or isinstance(val, bool):
            return False
    elif expected_type == "string":
        if not isinstance(val, str):
            return False
        if "enum" in schema and val not in schema["enum"]:
            if not any(match_single_value(val, e) for e in schema["enum"]):
                return False
    elif expected_type == "boolean":
        if not isinstance(val, bool):
            return False
    elif expected_type == "array":
        if not isinstance(val, list):
            return False
        if "items" in schema and isinstance(schema["items"], dict):
            for it in val:
                if not validate_schema_type(it, schema["items"]):
                    return False
    elif expected_type == "object":
        if not isinstance(val, dict):
            return False
        if "properties" in schema and isinstance(schema["properties"], dict):
            for pk, pv in schema["properties"].items():
                if pk in val and not validate_schema_type(val[pk], pv):
                    return False
    return True


def parse_tool_call(raw: str) -> tuple[bool, str | None, dict | None]:
    if not isinstance(raw, str):
        return False, None, None

    text = raw
    start_tag = "<tool_call>"
    end_tag = "</tool_call>"

    if start_tag in text:
        text = text.split(start_tag, 1)[1]
        if end_tag in text:
            text = text.split(end_tag, 1)[0]
    else:
        start_idx = text.find("{")
        end_idx = text.rfind("}")
        if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
            text = text[start_idx : end_idx + 1]
        else:
            return False, None, None

    text = text.strip()
    try:
        data = json.loads(text)
    except Exception:
        end_idx = text.rfind("}")
        if end_idx != -1:
            try:
                data = json.loads(text[: end_idx + 1])
            except Exception:
                return False, None, None
        else:
            return False, None, None

    if not isinstance(data, dict):
        return False, None, None

    name = data.get("name")
    arguments = data.get("arguments")

    if not isinstance(name, str) or not isinstance(arguments, dict):
        return False, None, None

    return True, name, arguments


def _normalize_accepted_list(
    accepted_raw: Any,
    prop_schema: dict[str, Any] | None = None,
) -> list[Any]:
    if not isinstance(accepted_raw, list):
        return [accepted_raw]
    prop_type = (prop_schema or {}).get("type")
    if prop_type == "array" and all(not isinstance(x, list) for x in accepted_raw):
        return [accepted_raw]
    return accepted_raw


def score_item_output(item: dict[str, Any], raw_output: str) -> ItemOutcome:
    json_valid, name, arguments = parse_tool_call(raw_output)

    gt = item.get("ground_truth", {})
    gt_name = gt.get("name", "")
    tool_correct = bool(json_valid and name == gt_name)

    tool_schema: dict[str, Any] | None = None
    if tool_correct and isinstance(arguments, dict):
        for t in item.get("tools", []):
            t_fn = t.get("function", t)
            if t_fn.get("name") == name:
                tool_schema = t_fn.get(
                    "parameters", {"type": "object", "properties": {}}
                )
                break

    args_schema_ok = False
    if tool_correct and isinstance(arguments, dict):
        if tool_schema is not None and isinstance(tool_schema, dict):
            required = tool_schema.get("required", [])
            properties = tool_schema.get("properties", {})

            has_required = all(k in arguments for k in required)
            has_no_unknown = all(k in properties for k in arguments.keys())

            types_ok = True
            if has_required and has_no_unknown:
                for k, v in arguments.items():
                    prop_def = properties.get(k, {})
                    if not validate_schema_type(v, prop_def):
                        types_ok = False
                        break

            args_schema_ok = has_required and has_no_unknown and types_ok

    values_exact = False
    if tool_correct and isinstance(arguments, dict):
        gt_args = gt.get("args", {})
        no_hallucinated = all(k in gt_args for k in arguments.keys())

        all_args_match = True
        if no_hallucinated:
            schema_props = (
                tool_schema.get("properties", {})
                if isinstance(tool_schema, dict)
                else {}
            )
            for k, accepted_raw in gt_args.items():
                accepted_list = _normalize_accepted_list(
                    accepted_raw, schema_props.get(k)
                )
                if k not in arguments:
                    if "" in accepted_list or None in accepted_list:
                        continue
                    all_args_match = False
                    break
                else:
                    v = arguments[k]
                    if not any(
                        match_single_value(v, acc) for acc in accepted_list
                    ):
                        all_args_match = False
                        break

        values_exact = no_hallucinated and all_args_match

    return ItemOutcome(
        item_id=str(item.get("id", "")),
        source=item.get("source", "bfcl_simple"),
        raw_output=raw_output,
        json_valid=json_valid,
        tool_correct=tool_correct,
        args_schema_ok=args_schema_ok,
        values_exact=values_exact,
    )


def compute_eval_scores(outcomes: list[ItemOutcome]) -> EvalScores:
    n = len(outcomes)
    if n == 0:
        empty = wilson_ci(0, 0)
        return EvalScores(
            json_valid=empty,
            tool_correct=empty,
            args_schema_ok=empty,
            values_exact=empty,
            by_source={},
        )

    json_valid_score = wilson_ci(sum(1 for o in outcomes if o.json_valid), n)
    tool_correct_score = wilson_ci(sum(1 for o in outcomes if o.tool_correct), n)
    args_schema_score = wilson_ci(
        sum(1 for o in outcomes if o.args_schema_ok), n
    )
    values_exact_score = wilson_ci(
        sum(1 for o in outcomes if o.values_exact), n
    )

    by_source: dict[str, EvalScores] = {}
    sources = set(o.source for o in outcomes)
    for src in sources:
        subset = [o for o in outcomes if o.source == src]
        m = len(subset)
        by_source[src] = EvalScores(
            json_valid=wilson_ci(sum(1 for o in subset if o.json_valid), m),
            tool_correct=wilson_ci(sum(1 for o in subset if o.tool_correct), m),
            args_schema_ok=wilson_ci(
                sum(1 for o in subset if o.args_schema_ok), m
            ),
            values_exact=wilson_ci(sum(1 for o in subset if o.values_exact), m),
            by_source={},
        )

    return EvalScores(
        json_valid=json_valid_score,
        tool_correct=tool_correct_score,
        args_schema_ok=args_schema_score,
        values_exact=values_exact_score,
        by_source=by_source,
    )


_TYPE_ALIASES: dict[str, str] = {
    "dict": "object",
    "float": "number",
    "int": "integer",
    "str": "string",
    "bool": "boolean",
    "tuple": "array",
    "list": "array",
    "any": "string",
}


def sanitize_schema_for_enforcer(schema: dict[str, Any]) -> dict[str, Any]:
    """Ensure every JSON Schema node has a valid type for lm-format-enforcer's JsonSchemaParser."""
    if not isinstance(schema, dict):
        return {"type": "string"}

    out: dict[str, Any] = dict(schema)
    raw_type = out.get("type")
    if isinstance(raw_type, str) and raw_type.lower() in _TYPE_ALIASES:
        out["type"] = _TYPE_ALIASES[raw_type.lower()]
    elif raw_type is None and not any(
        k in out for k in ("anyOf", "oneOf", "allOf", "enum", "const")
    ):
        if "properties" in out:
            out["type"] = "object"
        elif "items" in out:
            out["type"] = "array"
        else:
            out["type"] = "string"

    if "properties" in out and isinstance(out["properties"], dict):
        out["properties"] = {
            k: sanitize_schema_for_enforcer(v) if isinstance(v, dict) else {"type": "string"}
            for k, v in out["properties"].items()
        }
        if "required" in out and isinstance(out["required"], list):
            req_set = set(out["required"])
            out["required"] = [k for k in out["properties"] if k in req_set]

    if "items" in out and isinstance(out["items"], dict):
        out["items"] = sanitize_schema_for_enforcer(out["items"])

    return out


_LATIN1_AND_PUNCT: str = (
    "".join(chr(c) for c in range(0xA0, 0x100)) + "–—…’‘“”•€£¥℃™|~♥♠♦♣"
)
_COMPACT_ALPHABET: str = COMPLETE_ALPHABET + _LATIN1_AND_PUNCT


if JsonSchemaParser is not None and SequenceParser is not None:

    def _upgrade_parser_tree(p: Any) -> Any:
        if isinstance(p, JsonSchemaParser) and not isinstance(
            p, FastJsonSchemaParser
        ):
            p.__class__ = FastJsonSchemaParser
        elif isinstance(p, SequenceParser) and not isinstance(
            p, FastSequenceParser
        ):
            p.__class__ = FastSequenceParser
            for child in p.parsers:
                _upgrade_parser_tree(child)
        elif UnionParser is not None and isinstance(p, UnionParser):
            for child in p.parsers:
                _upgrade_parser_tree(child)
        return p

    class FastJsonSchemaParser(JsonSchemaParser):
        """JsonSchemaParser with token-mask caching for unconstrained string value states."""

        def _in_open_freetext_string(self) -> bool:
            if not self.object_stack:
                return False
            curr = self.object_stack[-1]
            return (
                StringParsingState is not None
                and isinstance(curr, StringParsingState)
                and curr.seen_opening_quote
                and not curr.seen_closing_quote
                and not curr.allowed_strings
                and not curr.regex_parser
                and (
                    curr.max_length is None
                    or len(curr.parsed_string) < curr.max_length
                )
            )

        def cache_key(self) -> Any:
            if self.object_stack and not self.can_end():
                curr = self.object_stack[-1]
                if (
                    StringParsingState is not None
                    and isinstance(curr, StringParsingState)
                    and not curr.allowed_strings
                    and not curr.regex_parser
                ):
                    ancestor_chars = tuple(
                        p.get_allowed_characters() for p in self.object_stack[:-1]
                    )
                    if not curr.seen_opening_quote:
                        return (
                            "json_str_start",
                            self.num_consecutive_whitespaces,
                            curr.min_length,
                            curr.max_length,
                            ancestor_chars,
                        )
                    if (
                        not curr.seen_closing_quote
                        and not curr.min_length
                        and not curr.max_length
                    ):
                        return ("json_str_body", ancestor_chars)
            return None

        def add_character(self, new_character: str) -> Any:
            if self._in_open_freetext_string() and new_character not in ('"', "\\"):
                self.context.active_parser = self
                curr = self.object_stack[-1]
                updated_stack = self.object_stack[:-1] + [
                    curr.add_character(new_character)
                ]
                updated = FastJsonSchemaParser(
                    self.context, self.config, updated_stack, 0
                )
                updated.context.active_parser = updated
                updated.last_parsed_string = self.last_parsed_string
                updated.last_non_whitespace_character = (
                    self.last_non_whitespace_character
                    if new_character in " \t\n\r"
                    else new_character
                )
                return updated
            res = super().add_character(new_character)
            res = _upgrade_parser_tree(res)
            if (
                isinstance(res, FastJsonSchemaParser)
                and res._in_open_freetext_string()
            ):
                res.num_consecutive_whitespaces = 0
            return res

    class FastSequenceParser(SequenceParser):
        """SequenceParser that forwards shortcut_key and cache_key while head parser cannot end."""

        def shortcut_key(self) -> Any:
            if len(self.parsers) == 1 or not self.parsers[0].can_end():
                return self.parsers[0].shortcut_key()
            return None

        def cache_key(self) -> Any:
            if len(self.parsers) == 1 or not self.parsers[0].can_end():
                return self.parsers[0].cache_key()
            return super().cache_key()

        def add_character(self, new_character: str) -> Any:
            if (
                self.parsers
                and isinstance(self.parsers[0], FastJsonSchemaParser)
                and not self.parsers[0].can_end()
                and self.parsers[0]._in_open_freetext_string()
                and new_character not in ('"', "\\")
            ):
                updated_head = self.parsers[0].add_character(new_character)
                if len(self.parsers) == 1:
                    return updated_head
                return FastSequenceParser([updated_head] + self.parsers[1:])
            res = super().add_character(new_character)
            return _upgrade_parser_tree(res)

else:
    FastJsonSchemaParser = None  # type: ignore[assignment,misc]
    FastSequenceParser = None  # type: ignore[assignment,misc]


def build_constrained_enforcer(
    tools: list[dict[str, Any]],
    tokenizer_data: TokenEnforcerTokenizerData,
) -> TokenEnforcer:
    anyof_branches: list[dict[str, Any]] = []
    for t in tools:
        fn = t.get("function", t)
        fn_name = fn["name"]
        raw_params = fn.get("parameters", {"type": "object", "properties": {}})
        fn_params = sanitize_schema_for_enforcer(raw_params)
        anyof_branches.append(
            {
                "type": "object",
                "properties": {
                    "name": {"const": fn_name},
                    "arguments": fn_params,
                },
                "required": ["name", "arguments"],
            }
        )

    schema = (
        anyof_branches[0]
        if len(anyof_branches) == 1
        else {"type": "object", "anyOf": anyof_branches}
    )
    cfg = None
    if CharacterLevelParserConfig is not None:
        cfg = CharacterLevelParserConfig(
            alphabet=_COMPACT_ALPHABET,
            max_consecutive_whitespaces=2,
            force_json_field_order=True,
        )
    json_parser = (
        FastJsonSchemaParser(schema, config=cfg)
        if FastJsonSchemaParser is not None
        else JsonSchemaParser(schema, config=cfg)
    )
    seq_cls = FastSequenceParser if FastSequenceParser is not None else SequenceParser
    parser = seq_cls(
        [
            StringParser("<tool_call>\n"),
            json_parser,
            StringParser("\n</tool_call>"),
        ]
    )
    enforcer = TokenEnforcer(tokenizer_data, parser)
    if cfg is not None:
        enforcer.root_parser.config = cfg
    shared_cache = getattr(tokenizer_data, "_shared_allowed_token_cache", None)
    if shared_cache is None:
        shared_cache = {}
        try:
            setattr(tokenizer_data, "_shared_allowed_token_cache", shared_cache)
        except Exception:
            pass
    enforcer.allowed_token_cache = shared_cache
    return enforcer


def evaluate_tool_calls(
    model: Any,
    tokenizer: Any,
    items: list[dict[str, Any]],
    mode: EvalMode = "unconstrained",
    device: str | torch.device | None = None,
    batch_size: int = 16,
    max_new_tokens: int = 192,
) -> tuple[list[ItemOutcome], EvalScores]:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    tokenizer_data = None
    if mode == "constrained":
        tokenizer_data = build_token_enforcer_tokenizer_data(tokenizer)

    outcomes: list[ItemOutcome] = []
    n_items = len(items)

    model.eval()
    for batch_start in range(0, n_items, batch_size):
        batch_items = items[batch_start : batch_start + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                it["messages"],
                tools=it.get("tools"),
                add_generation_prompt=True,
                tokenize=False,
            )
            for it in batch_items
        ]

        inputs = tokenizer(prompts, padding=True, return_tensors="pt").to(
            device
        )
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            if mode == "constrained":
                assert tokenizer_data is not None
                batch_enforcers = [
                    build_constrained_enforcer(it["tools"], tokenizer_data)
                    for it in batch_items
                ]
                eos_id = tokenizer.eos_token_id

                def prefix_fn(batch_id: int, sent: torch.Tensor) -> list[int]:
                    gen_list = sent[input_len:].tolist()
                    if eos_id is not None and eos_id in gen_list:
                        return [eos_id]
                    allowed = (
                        batch_enforcers[batch_id]
                        .get_allowed_tokens([0, *gen_list])
                        .allowed_tokens
                    )
                    if not allowed and eos_id is not None:
                        return [eos_id]
                    return allowed

                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    prefix_allowed_tokens_fn=prefix_fn,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            else:
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

        for j, it in enumerate(batch_items):
            gen_tokens = outputs[j][input_len:]
            raw_output = tokenizer.decode(
                gen_tokens, skip_special_tokens=False
            )
            outcome = score_item_output(it, raw_output)
            outcomes.append(outcome)

    scores = compute_eval_scores(outcomes)
    return outcomes, scores
