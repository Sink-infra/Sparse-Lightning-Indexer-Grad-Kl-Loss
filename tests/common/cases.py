"""Shared case schema and registry; concrete cases live in st/cases/*.json."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from math import sqrt

CORE_DTYPES = ("float16", "bfloat16")
GENERIC_N1 = (32, 64, 128)
GENERIC_NIDX1 = (8, 16, 32, 64)
SUPPORTED_K = tuple(range(1024, 8193, 1024))


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    description: str = ""
    b: int = 1
    s1: int = 1
    s2: int = 1024
    n1: int = 32
    nidx1: int = 8
    d_query: int = 512
    d_query_index: int = 128
    d_rope: int = 64
    k: int = 1024
    core_dtype: str = "float16"
    weights_dtype: str = "float16"
    query_rope_dtype: str = ""
    key_rope_dtype: str = ""
    actual_seq_dtype: str = "int64"
    actual_q: tuple = ()
    actual_k: tuple = ()
    layout: str = "BSND"
    indices_layout: str = "BSNK"
    statistics_layout: str = "BNSG"
    omit_lengths: bool = False
    value_profile: str = "random"
    rope_profile: str = "zero"
    index_pattern: str = "random"
    scale_value: float = 1.0 / sqrt(512.0)
    seed: int = 20260908
    repetitions: int = 1
    deterministic: bool = False
    output_view_offsets: tuple = ()
    tier: str = "regular"
    tags: tuple = ("platform_a2_a3",)
    expected: str = "accept"
    oracle_status: str = "exact"

    def actual_query_lengths(self):
        return self.actual_q or (self.s1,) * self.b

    def actual_key_lengths(self):
        return self.actual_k or (self.s2,) * self.b

    def input_shapes(self):
        b, q, k, h, ih = self.b, self.s1, self.s2, self.n1, self.nidx1
        shapes = {
            "query": (b, q, h, self.d_query), "key": (b, k, 1, self.d_query),
            "queryIndex": (b, q, ih, self.d_query_index),
            "keyIndex": (b, k, 1, self.d_query_index), "weights": (b, q, ih),
            "sparseIndices": (b, q, 1, self.k),
            "softmaxMax": (b, 1, q, h), "softmaxSum": (b, 1, q, h),
            "queryRope": (b, q, h, self.d_rope), "keyRope": (b, k, 1, self.d_rope),
            "actual_seq_lengths_query": (b,), "actual_seq_lengths_key": (b,),
        }
        if self.indices_layout == "BNSK":
            shapes["sparseIndices"] = (b, 1, q, self.k)
        elif self.indices_layout == "BSK":
            shapes["sparseIndices"] = (b, q, self.k)
        if self.statistics_layout == "BSNG":
            stats = (b, q, 1, h)
        elif self.statistics_layout == "BSG":
            stats = (b, q, h)
        elif self.statistics_layout == "BGS":
            stats = (b, h, q)
        else:
            stats = (b, 1, q, h)
        shapes["softmaxMax"] = shapes["softmaxSum"] = stats
        if self.layout == "TND":
            for name in ("query", "key", "queryIndex", "keyIndex", "weights", "queryRope", "keyRope"):
                old = shapes[name]
                shapes[name] = (old[0] * old[1], *old[2:])
            shapes["sparseIndices"] = (b * q, 1, self.k)
            shapes["softmaxMax"] = shapes["softmaxSum"] = (1, b * q, h)
        if self.omit_lengths:
            del shapes["actual_seq_lengths_query"]
            del shapes["actual_seq_lengths_key"]
        return shapes

    def input_storage_shapes(self):
        return self.input_shapes()

    def input_dtypes(self):
        return {
            "query": self.core_dtype, "key": self.core_dtype,
            "queryIndex": self.core_dtype, "keyIndex": self.core_dtype,
            "weights": self.weights_dtype, "sparseIndices": "int32",
            "softmaxMax": "float32", "softmaxSum": "float32",
            "queryRope": self.query_rope_dtype or self.core_dtype,
            "keyRope": self.key_rope_dtype or self.core_dtype,
            "actual_seq_lengths_query": self.actual_seq_dtype,
            "actual_seq_lengths_key": self.actual_seq_dtype,
        }

    def output_shapes(self):
        shapes = self.input_shapes()
        return {"dQueryIndex": shapes["queryIndex"], "dKeyIndex": shapes["keyIndex"],
                "dWeights": shapes["weights"], "loss": (1,)}

    def output_dtypes(self):
        return {"dQueryIndex": self.core_dtype, "dKeyIndex": self.core_dtype,
                "dWeights": self.weights_dtype, "loss": "float32"}

    def estimated_input_bytes(self):
        sizes = {"float16": 2, "bfloat16": 2, "float32": 4, "int32": 4, "int64": 8}
        from math import prod
        return sum(prod(shape) * sizes[self.input_dtypes()[name]]
                   for name, shape in self.input_shapes().items())

    def to_manifest(self):
        return {**asdict(self), "input_shapes": self.input_shapes(),
                "input_dtypes": self.input_dtypes(), "output_shapes": self.output_shapes(),
                "output_dtypes": self.output_dtypes(),
                "estimated_input_bytes": self.estimated_input_bytes(),
                "rope_formula_assumption": self.rope_profile != "zero"}


def _load_cases(group):
    path = Path(__file__).resolve().parents[1] / "st/cases" / (group + ".json")
    payload = json.loads(path.read_text())
    if payload["category"] != group:
        raise ValueError(f"Case category does not match file: {path}")
    cases = []
    for record in payload["cases"]:
        values = dict(record)
        if "scale_value" in values:
            values["scale_value"] = float(values["scale_value"])
        for name in ("actual_q", "actual_k", "tags"):
            if name in values:
                values[name] = tuple(values[name])
        if "output_view_offsets" in values:
            values["output_view_offsets"] = tuple(tuple(row) for row in values["output_view_offsets"])
        cases.append(CaseSpec(**values))
    return tuple(cases)


CASE_GROUPS = {group: _load_cases(group) for group in ("coverage", "correctness")}
CASES = tuple(case for group in CASE_GROUPS.values() for case in group)
_CASES = {case.case_id: case for case in CASES}
if len(_CASES) != len(CASES):
    raise ValueError("Duplicate case identifier across coverage/correctness files")


def get_case(name):
    return _CASES[name]


def accepted_cases(group=None):
    cases = CASES if group is None else CASE_GROUPS[group]
    return tuple(case for case in cases if case.expected == "accept")


def rejected_cases():
    return tuple(case for case in CASES if case.expected == "reject")
