"""Seeded, lazy test data; fixtures are keyed by case and oracle version."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np

from tests.common.cases import CASES, CASE_GROUPS, get_case
from tests.common.tensor_io import encode_logical, decode_logical

DEFAULT_MAX_BYTES = 256 * 1024 * 1024


def _values(case, rng):
    if case.rope_profile != "zero":
        raise ValueError("Only zero-RoPE fixtures have a confirmed reference formula")
    shapes = case.input_shapes()
    result = {}
    for name in ("query", "key", "queryIndex", "keyIndex", "weights", "queryRope", "keyRope"):
        shape = shapes[name]
        if "Rope" in name and case.rope_profile == "zero":
            value = np.zeros(shape, dtype=np.float32)
        else:
            value = rng.standard_normal(shape, dtype=np.float32)
            value *= 0.2 if "Index" in name or name == "weights" else 0.25
        if case.value_profile == "zero":
            value.fill(0)
        if case.value_profile == "weights_zero" and name == "weights":
            value.fill(0)
        if case.value_profile == "relu_zero" and name == "queryIndex":
            value.fill(0)
        if case.value_profile == "relu_negative" and name in ("queryIndex", "keyIndex"):
            value = (np.abs(value) + 0.01) * (-1 if name == "queryIndex" else 1)
        if case.value_profile == "sharp_main" and name in ("query", "key"):
            value *= 16
        if case.value_profile == "sharp_indexer" and name in ("queryIndex", "keyIndex", "weights"):
            value *= 8
        if case.value_profile == "large_finite" and name in ("queryIndex", "keyIndex"):
            value *= 16
        if case.value_profile == "selected_underflow" and name in ("query", "key"):
            value.fill(0)
            canonical = value.reshape(case.b, case.s1 if name == "query" else case.s2,
                                      case.n1 if name == "query" else 1, case.d_query)
            if name == "query":
                canonical[..., 0] = 1
            else:
                canonical[:, 0, 0, 0] = 200
                canonical[:, 1:, 0, 0] = np.linspace(-1, 1, case.s2 - 1)
        result[name] = encode_logical(value, case.input_dtypes()[name])
    return result


def _indices(case, rng):
    indices = np.full((case.b, case.s1, case.k), -1, dtype=np.int32)
    for b in range(case.b):
        qlength = case.actual_query_lengths()[b]
        klength = case.actual_key_lengths()[b]
        for t in range(qlength):
            visible = t + klength - qlength + 1
            if visible <= 0 or case.index_pattern == "invalid":
                continue
            if case.index_pattern == "duplicates":
                row = rng.choice(max(1, min(5, visible)), case.k)
            elif case.index_pattern == "endpoints":
                row = np.resize(np.asarray([0, visible - 1]), case.k)
            elif case.index_pattern == "exclude_first":
                row = rng.choice(np.arange(1, visible), case.k, replace=case.k >= visible)
            else:
                row = rng.choice(visible, case.k, replace=case.k > visible)
            if case.index_pattern == "mixed" and case.k >= 5:
                row[:5] = (0, visible - 1, visible, -1, case.s2)
            indices[b, t] = row
    return indices.reshape(case.input_shapes()["sparseIndices"])


def _stats(case, data):
    q = decode_logical(data["query"], case.core_dtype).reshape(case.b, case.s1, case.n1, case.d_query)
    k = decode_logical(data["key"], case.core_dtype).reshape(case.b, case.s2, case.d_query)
    if any(np.any(decode_logical(data[name], case.input_dtypes()[name]))
           for name in ("queryRope", "keyRope")):
        raise ValueError("Nonzero RoPE has no confirmed reference formula in this suite")
    maxima = np.zeros((case.b, case.s1, case.n1), dtype=np.float32)
    sums = np.ones_like(maxima)
    for b in range(case.b):
        qlength = case.actual_query_lengths()[b]
        klength = case.actual_key_lengths()[b]
        for t in range(qlength):
            visible = t + klength - qlength + 1
            # Stream the key dimension, so boundary fixtures do not materialize
            # a heads x full-sequence temporary.
            running_max = np.full(case.n1, -np.inf)
            running_sum = np.zeros(case.n1)
            for begin in range(0, visible, 4096):
                end = min(begin + 4096, visible)
                scores = q[b, t].astype(np.float64) @ k[b, begin:end].astype(np.float64).T
                scores *= float(case.scale_value)
                maximum = np.maximum(running_max, scores.max(axis=1))
                running_sum = running_sum * np.exp(running_max - maximum)
                running_sum += np.exp(scores - maximum[:, None]).sum(axis=1)
                running_max = maximum
            if visible:
                maxima[b, t] = running_max
                sums[b, t] = running_sum
    if case.statistics_layout == "BGS" and case.layout != "TND":
        maxima = maxima.transpose(0, 2, 1)
        sums = sums.transpose(0, 2, 1)
    return maxima.reshape(case.input_shapes()["softmaxMax"]), sums.reshape(case.input_shapes()["softmaxSum"])


def generate_inputs(case, max_bytes=DEFAULT_MAX_BYTES):
    if case.expected != "accept":
        raise ValueError("negative descriptors cannot be materialized as valid numerical inputs")
    if case.estimated_input_bytes() > max_bytes:
        raise MemoryError("large fixture requires --allow-large")
    rng = np.random.default_rng(case.seed)
    data = _values(case, rng)
    data["sparseIndices"] = _indices(case, rng)
    data["softmaxMax"], data["softmaxSum"] = _stats(case, data)
    if not case.omit_lengths:
        q = np.asarray(case.actual_query_lengths(), dtype=case.actual_seq_dtype)
        k = np.asarray(case.actual_key_lengths(), dtype=case.actual_seq_dtype)
        if case.layout == "TND":
            q = np.cumsum(q, dtype=q.dtype)
            k = np.cumsum(k, dtype=k.dtype)
        data["actual_seq_lengths_query"] = q
        data["actual_seq_lengths_key"] = k
    return data


def expected_storage(reference, case):
    return {name: encode_logical(value, case.output_dtypes()[name]) for name, value in reference.items()}


def fixture_cache_key(case):
    from tests.common.reference import REFERENCE_VERSION
    payload = {"case": case.to_manifest(), "reference": REFERENCE_VERSION, "generator": "spec-v2-zero-mask"}
    return case.case_id + "-" + hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]


def case_data_mode(case):
    return "empty" if case.s1 == 0 else ("zero" if case.value_profile == "zero" else "nonzero")


def materialize_runner_fixture(case, cache_root, max_bytes=DEFAULT_MAX_BYTES):
    from tests.common.reference import REFERENCE_VERSION, reference_backward
    root = Path(cache_root) / fixture_cache_key(case)
    marker = root / ".complete"
    if marker.exists():
        records = json.loads(marker.read_text())
        if all((root / name).is_file() and hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
               for name, digest in records.items()):
            return root, False
    data = generate_inputs(case, max_bytes)
    golden = expected_storage(reference_backward(data, case), case)
    write_runner_fixture(case, root, data, golden, REFERENCE_VERSION)
    return root, True


def write_runner_fixture(case, root, data, golden, reference_version):
    """Write an explicitly supplied experiment without changing the cached oracle."""
    root = Path(root)
    marker = root / ".complete"
    root.mkdir(parents=True, exist_ok=True)
    config = ["fixture_format=2", f"case_id={case.case_id}", f"reference_version={reference_version}",
              f"comparison={'launch_only' if case.s1 == 0 else 'exact'}", f"data_mode={case_data_mode(case)}", f"seed={case.seed}",
              f"scale_value={case.scale_value:.17g}", f"layout={case.layout}", "sparse_mode=3",
              f"deterministic={int(case.deterministic)}", f"repetitions={case.repetitions}"]
    zero_rows = np.ones((case.b, case.s2), dtype=np.uint8)
    indices = data["sparseIndices"].reshape(-1, case.k)
    query_offset = key_offset = 0
    for b, (q_length, k_length) in enumerate(zip(case.actual_query_lengths(), case.actual_key_lengths())):
        for row in range(q_length):
            token = query_offset + row if case.layout == "TND" else b * case.s1 + row
            selected = indices[token]
            selected = selected[(selected >= 0) & (selected < k_length) &
                                (selected <= row + k_length - q_length)]
            if case.layout == "TND":
                zero_rows.reshape(-1)[key_offset + selected] = 0
            else:
                zero_rows[b, selected] = 0
        query_offset += q_length
        key_offset += k_length
    zero_rows.tofile(root / "zero_key_rows.bin")
    config.append("zero_key_rows_file=zero_key_rows.bin")
    all_names = ("query", "key", "queryIndex", "keyIndex", "weights", "sparseIndices",
                 "softmaxMax", "softmaxSum", "queryRope", "keyRope",
                 "actual_seq_lengths_query", "actual_seq_lengths_key")
    for name in all_names:
        config.append(f"input.{name}.present={int(name in data)}")
        if name not in data:
            continue
        array = np.ascontiguousarray(data[name])
        array.tofile(root / f"input_{name}.bin")
        shape = ",".join(map(str, case.input_shapes()[name]))
        config += [f"input.{name}.shape={shape}", f"input.{name}.storage_shape={shape}",
                   f"input.{name}.dtype={case.input_dtypes()[name]}", f"input.{name}.file=input_{name}.bin"]
    for name, value in golden.items():
        np.ascontiguousarray(value).tofile(root / f"golden_{name}.bin")
        shape = ",".join(map(str, case.output_shapes()[name]))
        config += [f"output.{name}.shape={shape}", f"output.{name}.dtype={case.output_dtypes()[name]}",
                   f"output.{name}.golden=golden_{name}.bin"]
        offsets = dict(case.output_view_offsets)
        if name in offsets:
            config.append(f"output.{name}.view_offset_bytes={offsets[name]}")
    (root / "case.conf").write_text("\n".join(config) + "\n")
    (root / "case.json").write_text(json.dumps(case.to_manifest(), indent=2) + "\n")
    records = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
               for p in sorted(root.iterdir()) if p.is_file() and p.name != ".complete"}
    marker.write_text(json.dumps(records, sort_keys=True) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--list", action="store_true")
    parser.add_argument("--suite", choices=("all", "coverage", "correctness"), default="all")
    parser.add_argument("--runner-cache", type=Path)
    parser.add_argument("--allow-large", action="store_true")
    args = parser.parse_args()
    selected = CASES if args.suite == "all" else CASE_GROUPS[args.suite]
    selected_ids = {case.case_id for case in selected}
    if any(name not in selected_ids for name in args.case):
        parser.error("--case must belong to the selected --suite")
    if args.list:
        for case in selected:
            print(case.case_id, case.estimated_input_bytes(), case.tier)
    if args.runner_cache:
        for name in args.case:
            root, created = materialize_runner_fixture(
                get_case(name), args.runner_cache, (1 << 63) - 1 if args.allow_large else DEFAULT_MAX_BYTES)
            print("generated" if created else "reused", root)


if __name__ == "__main__":
    main()
