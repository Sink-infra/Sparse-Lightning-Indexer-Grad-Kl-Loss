"""Fixed inputs and reference outputs for the six measured Mock cases."""
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tests.common.reference import canonical_arrays, reference_backward
from tests.st.scripts.gen_data import generate_inputs, _stats, expected_storage, write_runner_fixture

DATA_VERSION = "mock-six-v3-forward-stats"
CANCELLATION_VERSION = "mock-cancellation-v1"
RTOL, ATOL = 0.002, 0.000001
CONFIG = Path(__file__).resolve().parents[1] / "st/cases/mock.json"


MEASURED_COLUMNS = ("b", "s1", "s2", "n1", "nidx1", "d_query", "d_query_index", "d_rope", "k", "scale_value", "core_dtype")
MEASURED_CASES = (
    (1, 2, 6, 4, 4, 32, 16, 8, 2, 0.1768, "float16"),
    (2, 3, 8, 4, 4, 32, 16, 8, 3, 0.1768, "float16"),
    (1, 4, 10, 8, 8, 32, 16, 8, 3, 0.125, "float32"),
    (1, 8, 16, 8, 8, 64, 32, 16, 3, 0.125, "float16"),
    (2, 4, 16, 8, 8, 64, 32, 16, 3, 0.125, "float16"),
    (1, 8, 32, 16, 16, 64, 32, 16, 4, 0.125, "float32"),
)


def validate_mock_suite(suite):
    items = suite["cases"]
    if (suite["data_version"] != DATA_VERSION or len(items) != 6 or
            [item["official_case"] for item in items] != list(range(1, 7))):
        raise ValueError("Mock requires the six ordered cases and the matching data version")
    for number, (item, measured) in enumerate(zip(items, MEASURED_CASES), 1):
        spec = item["spec"]
        if (spec["case_id"] != f"mock_case_{number}" or
                tuple(spec[name] for name in MEASURED_COLUMNS) != measured):
            raise ValueError(f"Mock case {number} differs from measured shape/scale/dtype")
        if any(spec[name] != value for name, value in
               (("layout", "BSND"), ("statistics_layout", "BNSG"), ("indices_layout", "BSNK"))):
            raise ValueError(f"Mock case {number} differs from measured layouts")
        for name in ("weights_dtype", "query_rope_dtype", "key_rope_dtype"):
            if (spec.get(name) or spec["core_dtype"]) != spec["core_dtype"]:
                raise ValueError(f"Mock case {number} differs from measured dtypes")
        if number == 2:
            valid = (not spec["omit_lengths"] and spec["actual_seq_dtype"] == "int32" and
                     tuple(spec["actual_q"]) == (2, 3) and tuple(spec["actual_k"]) == (5, 8))
        else:
            valid = spec["omit_lengths"] and not spec["actual_q"] and not spec["actual_k"]
        if not valid:
            raise ValueError(f"Mock case {number} differs from measured actual lengths")
    return suite


def load_mock_suite():
    return validate_mock_suite(json.loads(CONFIG.read_text()))


def _expanded_attention(data, case):
    """Use the independent dot-product oracle on concatenated main/RoPE features."""
    if not np.any(data["queryRope"]) and not np.any(data["keyRope"]):
        return data, case
    expanded = {**data,
                "query": np.concatenate((data["query"], data["queryRope"]), axis=-1),
                "key": np.concatenate((data["key"], data["keyRope"]), axis=-1),
                "queryRope": np.zeros_like(data["queryRope"]),
                "keyRope": np.zeros_like(data["keyRope"])}
    return expanded, replace(case, d_query=case.d_query + case.d_rope, rope_profile="zero")


def forward_statistics(maxima, sums):
    """Round supplied statistics the way a half-precision forward pass would.

    The six measured cases pass softmaxMax/softmaxSum straight from the sparse
    attention forward pass.  On the FP32 measurements those values disagree with
    the fp64 normalizer of the visible window by more than 1e-4 relative, which
    is exactly what makes the on-device path rebuild the normalizer instead of
    trusting them.  Fixtures keep the fp64 oracle untouched and only degrade the
    supplied inputs, so local timing exercises the same branch as the official
    runner.
    """
    return (_round_half_significand(maxima), _round_half_significand(sums))


def _round_half_significand(value):
    """Round-to-nearest-even to a half-precision significand, kept as float32.

    Only the significand is reduced (exponent range preserved), so the supplied
    pair stays usable as an fp32 tensor while carrying the ~5e-4 relative
    rounding a half-precision forward pass leaves behind.
    """
    bits = np.asarray(value, dtype=np.float32).view(np.uint32)
    dropped = np.uint32(13)                       # fp32 keeps 23, half keeps 10
    half = np.uint32(1) << (dropped - np.uint32(1))
    rounded = (bits + half - np.uint32(1) + ((bits >> dropped) & np.uint32(1)))
    rounded &= np.uint32(0xFFFFFFFF) << dropped
    return rounded.view(np.float32)


def generate_mock_inputs(case):
    # A single, commit-independent workload. Dense FP32 features exercise the
    # scalar product/gradient paths which zero-heavy cancellation probes skip.
    data = generate_inputs(replace(case, rope_profile="zero"), 64 << 20)
    if case.core_dtype == "float32":
        rng = np.random.default_rng(case.seed + 1)
        for name in ("queryRope", "keyRope"):
            data[name][:] = rng.normal(0, 0.25, data[name].shape).astype(np.float32)
        expanded, expanded_case = _expanded_attention(data, case)
        maxima, sums = _stats(expanded_case, expanded)
        data["softmaxMax"], data["softmaxSum"] = forward_statistics(maxima, sums)
    return data


def generate_cancellation_inputs(case):
    """Supplemental precision regression; never used for official timing acceptance."""
    if case.case_id not in {f"mock_case_{i}" for i in range(1, 7)}:
        raise ValueError("Only the six fixed Mock cases are supported")
    data = generate_inputs(replace(case, rope_profile="zero"), 64 << 20)
    if case.core_dtype != "float32":
        return data
    for array in data.values():
        array.fill(0)
    data["sparseIndices"][:] = np.arange(case.k, dtype=np.int32)
    pairs = 4 if case.case_id == "mock_case_3" else 3
    for row in range(case.s1):
        visible = case.s2 - case.s1 + row + 1
        score = math.log((visible - 1) * 2 ** 24)
        high = np.float32(score)
        for head in range(2 * pairs):
            peak = 0 if head % 2 == 0 else 2
            component = 0 if peak == 0 else 1
            data["key"][0, peak, 0, component] = 1
            data["key"][0, peak, 0, component + 4] = 1
            data["query"][0, row, head, component] = np.float32(8 * high)
            low = score - float(high) + (-2 ** -24 if head % 2 == 0 else 2 ** -24)
            data["query"][0, row, head, component + 4] = np.float32(8 * low)
    if case.case_id == "mock_case_3":
        for index in range(3):
            data["keyIndex"][0, index, 0, index] = 1
        for row in range(case.s1):
            for head in range(case.nidx1):
                edge = 1 + head / 8
                data["queryIndex"][0, row, head, :3] = (edge, 3 + row / 4, edge)
        data["queryIndex"][0, 2, 0, :3] = (72, 96, 120)
    else:
        for index, (anti, contrast) in enumerate(((1, 1), (0, -1), (-1, 1), (0, -1))):
            data["keyIndex"][0, index, 0, :2] = 32 * anti
            for dimension in range(2, case.d_query_index):
                data["keyIndex"][0, index, 0, dimension] = contrast + 1 + dimension % 3
        data["queryIndex"][..., 0] = 1 / 32
        data["weights"].fill(1)
        data["queryIndex"][0, 0, 0].fill(0)
        data["queryIndex"][0, 0, 0, 2:4] = (1, -1)
        for head in range(1, 16):
            data["queryIndex"][0, 0, head, 0] = (1 if head <= 8 else -1) / 32
            data["weights"][0, 0, head] = 7 if head <= 8 else 8
    data["softmaxMax"], data["softmaxSum"] = _stats(case, data)
    return data


def _reference(data, case):
    data, case = _expanded_attention(data, case)
    if case.core_dtype != "float32":
        return reference_backward(data, case)
    logical = canonical_arrays(data, case)
    maxima = np.zeros((case.b, case.s1, case.n1), dtype=np.float64)
    sums = np.ones_like(maxima)
    for batch in range(case.b):
        qlen, klen = case.actual_query_lengths()[batch], case.actual_key_lengths()[batch]
        for row in range(qlen):
            visible = row + klen - qlen + 1
            scores = logical["query"][batch, row].astype(np.float64) @ logical["key"][batch, :visible, 0].astype(np.float64).T
            scores *= case.scale_value
            maxima[batch, row] = scores.max(axis=1)
            sums[batch, row] = np.exp(scores - maxima[batch, row, :, None]).sum(axis=1)
    # Only the FP32 Mock oracle recomputes the visible softmax. Runtime inputs
    # retain their fixed FP32 max/sum; the formal reference remains unchanged.
    exact = {**data, "softmaxMax": maxima.reshape(data["softmaxMax"].shape),
             "softmaxSum": sums.reshape(data["softmaxSum"].shape)}
    return reference_backward(exact, case)


def materialize_mock_fixture(case, cache_root, profile="official"):
    if profile not in ("official", "cancellation"):
        raise ValueError("Unknown Mock input profile")
    version = DATA_VERSION if profile == "official" else CANCELLATION_VERSION
    identity = hashlib.sha256((version + json.dumps(case.to_manifest(), sort_keys=True)).encode())
    for path in (Path(__file__), Path(__file__).with_name("reference.py"),
                 CONFIG.parents[1] / "scripts/gen_data.py"):
        identity.update(path.read_bytes())
    folder = Path(cache_root) / version / (case.case_id + "-" + identity.hexdigest()[:20])
    marker = folder / ".complete"
    if marker.is_file():
        records = json.loads(marker.read_text())
        if all((folder / name).is_file() and hashlib.sha256((folder / name).read_bytes()).hexdigest() == value
               for name, value in records.items()):
            return folder
    data = (generate_mock_inputs if profile == "official" else generate_cancellation_inputs)(case)
    write_runner_fixture(case, folder, data, expected_storage(_reference(data, case), case), version)
    return folder
