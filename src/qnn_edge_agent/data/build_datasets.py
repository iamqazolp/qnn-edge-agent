import json
import random
from pathlib import Path
from typing import Any

from huggingface_hub import hf_hub_download

from qnn_edge_agent.data.bfcl_schema import bfcl_to_openai_tool
from qnn_edge_agent.env import REPO_ROOT

BFCL_REPO = "gorilla-llm/Berkeley-Function-Calling-Leaderboard"
SEED = 42


def download_bfcl_file(filename: str) -> Path:
    return Path(
        hf_hub_download(repo_id=BFCL_REPO, filename=filename, repo_type="dataset")
    )


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_bfcl_split(
    prompt_file: str, answer_file: str
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    prompt_path = download_bfcl_file(prompt_file)
    ans_path = download_bfcl_file(answer_file)

    prompts = load_jsonl(prompt_path)
    answers_list = load_jsonl(ans_path)
    answers = {item["id"]: item for item in answers_list}

    joined: list[tuple[dict[str, Any], dict[str, Any]]] = []
    for p in prompts:
        pid = p["id"]
        if pid in answers:
            joined.append((p, answers[pid]))
    return joined


def normalize_messages(question: Any) -> list[dict[str, str]]:
    if isinstance(question, list) and len(question) > 0:
        if isinstance(question[0], list):
            return question[0]
        if isinstance(question[0], dict):
            return question
    if isinstance(question, str):
        return [{"role": "user", "content": question}]
    raise ValueError(f"Unexpected question structure: {type(question)}")


def normalize_record(
    item_id: str,
    source: str,
    raw_prompt: dict[str, Any],
    raw_answer: dict[str, Any],
) -> dict[str, Any]:
    messages = normalize_messages(raw_prompt.get("question"))
    raw_funcs = raw_prompt.get("function", [])
    tools = [bfcl_to_openai_tool(fn) for fn in raw_funcs]

    gt_list = raw_answer.get("ground_truth", [])
    if not gt_list or not isinstance(gt_list[0], dict):
        raise ValueError(f"Invalid ground truth in item {item_id}")

    gt_dict = gt_list[0]
    func_name = next(iter(gt_dict.keys()))
    args = gt_dict[func_name]
    if not isinstance(args, dict):
        raise ValueError(f"Args for {func_name} is not dict in item {item_id}")

    return {
        "id": item_id,
        "source": source,
        "messages": messages,
        "tools": tools,
        "ground_truth": {
            "name": func_name,
            "args": args,
        },
    }


def extract_prompt_texts(messages: list[dict[str, str]]) -> set[str]:
    return {
        m["content"].strip()
        for m in messages
        if isinstance(m, dict) and m.get("role") == "user"
    }


def extract_func_names(tools: list[dict[str, Any]]) -> set[str]:
    names = set()
    for t in tools:
        if "function" in t and "name" in t["function"]:
            names.add(t["function"]["name"])
        elif "name" in t:
            names.add(t["name"])
    return names


def build_datasets(
    output_dir: Path | None = None,
    seed: int = SEED,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    out_dir = output_dir or (REPO_ROOT / "data")
    out_dir.mkdir(parents=True, exist_ok=True)

    iot_path = out_dir / "iot_items.json"
    if not iot_path.exists():
        raise FileNotFoundError(f"Missing hand-written IoT items file: {iot_path}")

    with open(iot_path, "r", encoding="utf-8") as f:
        iot_data = json.load(f)

    eval_iot = iot_data.get("eval", [])
    calib_iot = iot_data.get("calibration", [])

    if len(eval_iot) != 40:
        raise ValueError(f"Expected 40 eval IoT items, got {len(eval_iot)}")
    if len(calib_iot) != 32:
        raise ValueError(f"Expected 32 calib IoT items, got {len(calib_iot)}")

    raw_simple = load_bfcl_split(
        "BFCL_v3_simple.json", "possible_answer/BFCL_v3_simple.json"
    )
    raw_multiple = load_bfcl_split(
        "BFCL_v3_multiple.json", "possible_answer/BFCL_v3_multiple.json"
    )

    sampled_simple = rng.sample(raw_simple, 300)
    sampled_multiple = rng.sample(raw_multiple, 120)

    eval_records: list[dict[str, Any]] = []
    for prompt_it, ans_it in sampled_simple:
        eval_records.append(
            normalize_record(
                item_id=f"eval_{prompt_it['id']}",
                source="bfcl_simple",
                raw_prompt=prompt_it,
                raw_answer=ans_it,
            )
        )
    for prompt_it, ans_it in sampled_multiple:
        eval_records.append(
            normalize_record(
                item_id=f"eval_{prompt_it['id']}",
                source="bfcl_multiple",
                raw_prompt=prompt_it,
                raw_answer=ans_it,
            )
        )
    for it in eval_iot:
        eval_records.append(it)

    eval_func_names: set[str] = set()
    eval_prompt_texts: set[str] = set()
    for rec in eval_records:
        eval_func_names.update(extract_func_names(rec["tools"]))
        eval_prompt_texts.update(extract_prompt_texts(rec["messages"]))

    raw_live_simple = load_bfcl_split(
        "BFCL_v3_live_simple.json", "possible_answer/BFCL_v3_live_simple.json"
    )
    raw_live_multiple = load_bfcl_split(
        "BFCL_v3_live_multiple.json", "possible_answer/BFCL_v3_live_multiple.json"
    )

    def is_disjoint(prompt_it: dict[str, Any]) -> bool:
        fns = {fn["name"] for fn in prompt_it.get("function", [])}
        if fns & eval_func_names:
            return False
        msgs = normalize_messages(prompt_it.get("question"))
        prompts = extract_prompt_texts(msgs)
        if prompts & eval_prompt_texts:
            return False
        return True

    clean_live_simple = [pair for pair in raw_live_simple if is_disjoint(pair[0])]
    clean_live_multiple = [pair for pair in raw_live_multiple if is_disjoint(pair[0])]

    sampled_live_simple = rng.sample(clean_live_simple, 48)
    sampled_live_multiple = rng.sample(clean_live_multiple, 48)

    calib_records: list[dict[str, Any]] = []
    for prompt_it, ans_it in sampled_live_simple:
        calib_records.append(
            normalize_record(
                item_id=f"calib_{prompt_it['id']}",
                source="live_simple",
                raw_prompt=prompt_it,
                raw_answer=ans_it,
            )
        )
    for prompt_it, ans_it in sampled_live_multiple:
        calib_records.append(
            normalize_record(
                item_id=f"calib_{prompt_it['id']}",
                source="live_multiple",
                raw_prompt=prompt_it,
                raw_answer=ans_it,
            )
        )
    for it in calib_iot:
        calib_records.append(it)

    calib_func_names: set[str] = set()
    calib_prompt_texts: set[str] = set()
    for rec in calib_records:
        calib_func_names.update(extract_func_names(rec["tools"]))
        calib_prompt_texts.update(extract_prompt_texts(rec["messages"]))

    func_overlap = eval_func_names & calib_func_names
    prompt_overlap = eval_prompt_texts & calib_prompt_texts

    if func_overlap:
        raise AssertionError(
            f"Function name overlap between eval and calib ({len(func_overlap)}): {func_overlap}"
        )
    if prompt_overlap:
        raise AssertionError(
            f"Prompt text overlap between eval and calib ({len(prompt_overlap)}): {prompt_overlap}"
        )

    eval_out = out_dir / "eval_tool_calls.jsonl"
    calib_out = out_dir / "calibration_128.jsonl"

    with open(eval_out, "w", encoding="utf-8") as f:
        for rec in eval_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with open(calib_out, "w", encoding="utf-8") as f:
        for rec in calib_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return eval_records, calib_records


if __name__ == "__main__":
    eval_recs, calib_recs = build_datasets()
    print(f"Successfully generated datasets:")
    print(f"  eval_tool_calls.jsonl: {len(eval_recs)} records")
    print(
        f"    - bfcl_simple: {sum(1 for r in eval_recs if r['source'] == 'bfcl_simple')}"
    )
    print(
        f"    - bfcl_multiple: {sum(1 for r in eval_recs if r['source'] == 'bfcl_multiple')}"
    )
    print(f"    - iot: {sum(1 for r in eval_recs if r['source'] == 'iot')}")
    print(f"  calibration_128.jsonl: {len(calib_recs)} records")
    print(
        f"    - live_simple: {sum(1 for r in calib_recs if r['source'] == 'live_simple')}"
    )
    print(
        f"    - live_multiple: {sum(1 for r in calib_recs if r['source'] == 'live_multiple')}"
    )
    print(f"    - iot: {sum(1 for r in calib_recs if r['source'] == 'iot')}")
