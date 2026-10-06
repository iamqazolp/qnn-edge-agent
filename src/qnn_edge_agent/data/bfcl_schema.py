from copy import deepcopy
from typing import Any

TYPE_MAPPING = {
    "dict": "object",
    "float": "number",
    "tuple": "array",
    "integer": "integer",
    "string": "string",
    "boolean": "boolean",
    "array": "array",
}


def bfcl_type_to_json_schema(schema: Any) -> Any:
    if not isinstance(schema, dict):
        return schema

    out = deepcopy(schema)

    raw_type = out.get("type")
    if isinstance(raw_type, str):
        if raw_type == "any":
            del out["type"]
        elif raw_type in TYPE_MAPPING:
            out["type"] = TYPE_MAPPING[raw_type]

    if "properties" in out and isinstance(out["properties"], dict):
        if "type" not in out:
            out["type"] = "object"
        out["properties"] = {
            k: bfcl_type_to_json_schema(v) for k, v in out["properties"].items()
        }

    if "items" in out:
        if "type" not in out:
            out["type"] = "array"
        if isinstance(out["items"], dict):
            out["items"] = bfcl_type_to_json_schema(out["items"])
        elif isinstance(out["items"], list):
            out["items"] = [bfcl_type_to_json_schema(it) for it in out["items"]]

    return out


def bfcl_to_openai_tool(fn_doc: dict[str, Any]) -> dict[str, Any]:
    params = fn_doc.get("parameters", {})
    converted_params = bfcl_type_to_json_schema(params)
    if isinstance(converted_params, dict) and "type" not in converted_params:
        converted_params["type"] = "object"
    if isinstance(converted_params, dict) and "properties" not in converted_params:
        converted_params["properties"] = {}

    return {
        "type": "function",
        "function": {
            "name": fn_doc["name"],
            "description": fn_doc.get("description", ""),
            "parameters": converted_params,
        },
    }
