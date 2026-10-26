from __future__ import annotations

import dataclasses
import time
from typing import Any, Callable

import torch

from qnn_edge_agent.agent.tools import (
    EdgeToolEnvironment,
    ToolExecutionResult,
    load_edge_tool_schemas,
    validate_args_against_schema,
)
from qnn_edge_agent.eval.tool_call_eval import (
    TokenEnforcerTokenizerData,
    build_constrained_enforcer,
    build_token_enforcer_tokenizer_data,
    parse_tool_call,
)
from qnn_edge_agent.quant.calibrate import CalibStats
from qnn_edge_agent.quant.fake_quant import attach_activation_hooks
from qnn_edge_agent.types import EvalMode

GeneratorFn = Callable[[list[dict[str, Any]], list[dict[str, Any]], EvalMode], str]


@dataclasses.dataclass
class AttemptRecord:
    stage: str
    mode: EvalMode
    raw_output: str
    json_valid: bool
    tool_name: str | None
    arguments: dict[str, Any] | None
    schema_valid: bool
    validation_error: str | None
    latency_ms: float


@dataclasses.dataclass
class AgentTurnResult:
    ok: bool
    path_taken: str
    tool_name: str | None
    arguments: dict[str, Any] | None
    execution: ToolExecutionResult | None
    attempts: list[AttemptRecord]
    total_latency_ms: float
    fallback_memory_overhead_mb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class HuggingFaceToolGenerator:
    """Wraps a (quantized or FP) HuggingFace CausalLM for unconstrained and grammar-constrained decoding."""

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        device: str | torch.device = "cpu",
        max_new_tokens: int = 256,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.max_new_tokens = max_new_tokens
        self._tokenizer_data: TokenEnforcerTokenizerData | None = None

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

    def _get_tokenizer_data(self) -> TokenEnforcerTokenizerData:
        if self._tokenizer_data is None:
            self._tokenizer_data = build_token_enforcer_tokenizer_data(self.tokenizer)
        return self._tokenizer_data

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        mode: EvalMode = "unconstrained",
    ) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tools=tools,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = self.tokenizer([prompt], padding=True, return_tensors="pt").to(self.device)
        input_len = inputs["input_ids"].shape[1]

        self.model.eval()
        with torch.no_grad():
            if mode == "constrained":
                enforcer = build_constrained_enforcer(tools, self._get_tokenizer_data())
                eos_id = self.tokenizer.eos_token_id

                def prefix_fn(batch_id: int, sent: torch.Tensor) -> list[int]:
                    sent_list = sent.tolist()
                    if eos_id is not None and eos_id in sent_list[input_len:]:
                        return [eos_id]
                    return enforcer.get_allowed_tokens(sent_list).allowed_tokens

                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    prefix_allowed_tokens_fn=prefix_fn,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )
            else:
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                )

        gen_tokens = outputs[0][input_len:]
        return str(self.tokenizer.decode(gen_tokens, skip_special_tokens=False))


class InPlaceW4A16FallbackGenerator:
    """
    Zero-RAM-overhead Tier-3 fallback generator: promotes a live W4A8 model to W4A16 (E2a)
    in-place by swapping its 8-bit activation pre-hooks to unclipped 16-bit activation hooks
    for the fallback turn, then restoring 8-bit hooks afterward.
    """

    memory_overhead_mb: float = 0.0

    def __init__(
        self,
        primary_generator: HuggingFaceToolGenerator,
        hook_handles: list[torch.utils.hooks.RemovableHandle],
        active_calib: CalibStats | dict[str, Any] | None = None,
        primary_act_bits: int = 8,
        fallback_act_bits: int = 16,
    ) -> None:
        self.primary_generator = primary_generator
        self.hook_handles = hook_handles
        self.active_calib = active_calib
        self.primary_act_bits = primary_act_bits
        self.fallback_act_bits = fallback_act_bits

    def __call__(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]],
        mode: EvalMode = "constrained",
    ) -> str:
        model = self.primary_generator.model
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

        a16_handles = attach_activation_hooks(
            model=model,
            act_bits=self.fallback_act_bits,
            active_calib=self.active_calib,
        )
        try:
            return self.primary_generator(messages, tools, mode)
        finally:
            for h in a16_handles:
                h.remove()
            restored = attach_activation_hooks(
                model=model,
                act_bits=self.primary_act_bits,
                active_calib=self.active_calib,
            )
            self.hook_handles.extend(restored)


class ReliableToolAgent:
    """
    3-tier runtime reliability layer for quantized edge tool-calling agents:
      1. Primary model unconstrained decoding -> JSON + schema validation
      2. Primary model grammar-constrained retry (lm-format-enforcer) on syntax/schema fault
      3. Zero-RAM W4A16 in-place activation promotion (or W8A8 fallback) if still invalid
      4. Sandboxed PC / IoT tool execution via EdgeToolEnvironment
    """

    def __init__(
        self,
        env: EdgeToolEnvironment,
        primary_generator: GeneratorFn,
        fallback_generator: GeneratorFn | None = None,
        enable_grammar_retry: bool = True,
        fallback_stage_name: str = "fallback_w8a8",
        fallback_memory_overhead_mb: float = 0.0,
    ) -> None:
        self.env = env
        self.primary_generator = primary_generator
        self.fallback_generator = fallback_generator
        self.enable_grammar_retry = enable_grammar_retry
        self.fallback_stage_name = fallback_stage_name
        self.fallback_memory_overhead_mb = float(
            getattr(fallback_generator, "memory_overhead_mb", fallback_memory_overhead_mb)
        )

    def _validate_candidate(
        self,
        raw_output: str,
        allowed_tools: list[dict[str, Any]],
    ) -> tuple[bool, str | None, dict[str, Any] | None, bool, str | None]:
        json_valid, name, args = parse_tool_call(raw_output)
        if not json_valid or name is None or args is None:
            return False, None, None, False, "Malformed tool_call JSON or missing name/arguments"

        allowed_schemas: dict[str, dict[str, Any]] = {}
        for t in allowed_tools:
            fn = t.get("function", t)
            if "name" in fn:
                allowed_schemas[fn["name"]] = fn.get(
                    "parameters", {"type": "object", "properties": {}}
                )

        if allowed_schemas and name not in allowed_schemas:
            return (
                True,
                name,
                args,
                False,
                f"Tool {name!r} is not in the candidate tool list ({ sorted(allowed_schemas.keys()) })",
            )

        schema = allowed_schemas.get(name) or self.env.schemas_by_name.get(name)
        if schema is None:
            return True, name, args, False, f"Unknown tool: {name!r}"

        schema_ok = validate_args_against_schema(args, schema)
        err = None if schema_ok else f"Arguments for {name!r} failed JSON Schema validation"
        return True, name, args, schema_ok, err

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        execute_tool: bool = True,
    ) -> AgentTurnResult:
        t_start = time.perf_counter()
        candidate_tools = tools if tools is not None else load_edge_tool_schemas()
        attempts: list[AttemptRecord] = []

        stages: list[tuple[str, GeneratorFn, EvalMode]] = [
            ("primary_unconstrained", self.primary_generator, "unconstrained"),
        ]
        if self.enable_grammar_retry:
            stages.append(("primary_grammar_retry", self.primary_generator, "constrained"))
        if self.fallback_generator is not None:
            stages.append((self.fallback_stage_name, self.fallback_generator, "constrained"))

        for stage_name, gen_fn, mode in stages:
            t0 = time.perf_counter()
            raw = gen_fn(messages, candidate_tools, mode)
            dt_ms = (time.perf_counter() - t0) * 1000.0

            json_ok, name, args, schema_ok, err = self._validate_candidate(raw, candidate_tools)
            attempts.append(
                AttemptRecord(
                    stage=stage_name,
                    mode=mode,
                    raw_output=raw,
                    json_valid=json_ok,
                    tool_name=name,
                    arguments=args,
                    schema_valid=schema_ok,
                    validation_error=err,
                    latency_ms=dt_ms,
                )
            )

            if json_ok and schema_ok and name is not None and args is not None:
                exec_res = self.env.execute(name, args) if execute_tool else None
                total_ms = (time.perf_counter() - t_start) * 1000.0
                return AgentTurnResult(
                    ok=True if exec_res is None else exec_res.ok,
                    path_taken=stage_name,
                    tool_name=name,
                    arguments=args,
                    execution=exec_res,
                    attempts=attempts,
                    total_latency_ms=total_ms,
                    fallback_memory_overhead_mb=self.fallback_memory_overhead_mb,
                )

        total_ms = (time.perf_counter() - t_start) * 1000.0
        return AgentTurnResult(
            ok=False,
            path_taken="failed",
            tool_name=attempts[-1].tool_name if attempts else None,
            arguments=attempts[-1].arguments if attempts else None,
            execution=None,
            attempts=attempts,
            total_latency_ms=total_ms,
            fallback_memory_overhead_mb=self.fallback_memory_overhead_mb,
        )
