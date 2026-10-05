from dataclasses import dataclass, field
from typing import Literal


@dataclass(frozen=True)
class QuantFormat:
    bits: int
    symmetric: bool
    granularity: Literal["per_channel", "per_tensor"]


@dataclass(frozen=True)
class LayerQuant:
    weight: QuantFormat | None
    act: QuantFormat | None
    adaround: bool = False


@dataclass(frozen=True)
class Recipe:
    id: str
    smooth: Literal["none", "AB", "ABC", "ABCV"]
    alpha: float | Literal["search"] | None
    default: LayerQuant
    allocator: Literal["uniform", "sqnr_knapsack"] = "uniform"
    bit_budget: float | None = None
    clip_search: bool = False
    kv_bits: int = 16


EvalMode = Literal["unconstrained", "constrained"]


@dataclass(frozen=True)
class ItemOutcome:
    item_id: str
    source: Literal["bfcl_simple", "bfcl_multiple", "iot"]
    raw_output: str
    json_valid: bool
    tool_correct: bool
    args_schema_ok: bool
    values_exact: bool


@dataclass(frozen=True)
class TierScore:
    rate: float
    ci_low: float
    ci_high: float
    n: int


@dataclass(frozen=True)
class EvalScores:
    json_valid: TierScore
    tool_correct: TierScore
    args_schema_ok: TierScore
    values_exact: TierScore
    by_source: dict[str, "EvalScores"] = field(default_factory=dict)
