"""Fixed inputs and reference outputs for the six measured Mock cases."""
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from tests.common.reference import canonical_arrays, reference_backward
from tests.st.scripts.gen_data import generate_inputs, _stats, expected_storage, write_runner_fixture

DATA_VERSION = "mock-six-v1"
RTOL, ATOL = 0.002, 0.000001
CONFIG = Path(__file__).resolve().parents[1] / "st/cases/mock.json"


def load_mock_suite():
    suite = json.loads(CONFIG.read_text())
    items = suite["cases"]
    if (suite["data_version"] != DATA_VERSION or
            [item["official_case"] for item in items] != list(range(1, 7)) or
            [item["spec"]["case_id"] for item in items] != [f"mock_case_{i}" for i in range(1, 7)]):
        raise ValueError("Mock requires the six ordered cases and the matching data version")
    return suite


def generate_mock_inputs(case):
    if case.case_id not in {f"mock_case_{i}" for i in range(1, 7)}:
        raise ValueError("Only the six fixed Mock cases are supported")
    data = generate_inputs(case, 64 << 20)
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


def materialize_mock_fixture(case, cache_root):
    identity = hashlib.sha256((DATA_VERSION + json.dumps(case.to_manifest(), sort_keys=True)).encode())
    for path in (Path(__file__), Path(__file__).with_name("reference.py"),
                 CONFIG.parents[1] / "scripts/gen_data.py"):
        identity.update(path.read_bytes())
    folder = Path(cache_root) / DATA_VERSION / (case.case_id + "-" + identity.hexdigest()[:20])
    marker = folder / ".complete"
    if marker.is_file():
        records = json.loads(marker.read_text())
        if all((folder / name).is_file() and hashlib.sha256((folder / name).read_bytes()).hexdigest() == value
               for name, value in records.items()):
            return folder
    data = generate_mock_inputs(case)
    write_runner_fixture(case, folder, data, expected_storage(_reference(data, case), case), DATA_VERSION)
    return folder
