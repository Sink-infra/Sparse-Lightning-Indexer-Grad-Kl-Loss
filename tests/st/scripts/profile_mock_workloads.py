"""Run and profile the six fixed Mock workloads on the current built operator."""
import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import time
import numpy as np

from tests.common.cases import CaseSpec
from tests.common.mock_cases import (
    CONFIG, DATA_VERSION, RTOL, ATOL, generate_mock_inputs, load_mock_suite, materialize_mock_fixture,
)
from tests.common.reference import canonical_arrays
from tests.st.scripts.kernel_cache import content_record
from tests.st.scripts.run_tests import (
    DEFAULT_FINAL_BUILD, DEFAULT_FIXTURE_CACHE, DEFAULT_RESULTS, ROOT,
    build_runner, existing_final_build_dir, resource_check, serialized_npu, _toolkit_root,
)


def workload_signature(case):
    data = canonical_arrays(generate_mock_inputs(case), case)
    batches = []
    for batch, (qlen, klen) in enumerate(zip(case.actual_query_lengths(), case.actual_key_lengths())):
        pairs, positive = 0, 0
        for row in range(qlen):
            indices = data["sparseIndices"][batch, row]
            indices = indices[(indices >= 0) & (indices < klen) & (indices <= row + klen - qlen)]
            pairs += len(indices)
            similarity = (data["queryIndex"][batch, row].astype(np.float64)
                          @ data["keyIndex"][batch, indices, 0].astype(np.float64).T)
            positive += int(np.count_nonzero(similarity > 0))
        batches.append({"visible_pairs": pairs, "positive_relu_pairs": positive,
                        "main_dot_elements": pairs * case.n1 * (case.d_query + case.d_rope),
                        "index_dot_elements": 2 * pairs * case.nidx1 * case.d_query_index,
                        "gradient_elements": 2 * positive * case.d_query_index})
    return batches


def read_profile(output, configurations, launch_log):
    rows = []
    observed = {}
    # The runner flushes the case label before the actual runtime kernel name.
    current = -1
    sequence = {}
    seen = set()
    for line in Path(launch_log).read_text().splitlines():
        case = re.match(r"case=(\S+) network_deterministic=", line)
        if case:
            current += 1
            if current >= len(configurations) or case[1] != configurations[current]["case_id"]:
                raise RuntimeError("Profiler runner case order differs from configurations")
        name = re.search(r"<ProfData> Start profiling on device \d+, kernel: (\S+)", line)
        if name:
            if current < 0 or current in seen:
                raise RuntimeError("Unmapped or repeated profiler launch")
            seen.add(current)
            local_index = sequence.get(name[1], 0)
            observed[name[1], local_index] = current
            sequence[name[1]] = local_index + 1
    if seen != set(range(len(configurations))):
        raise RuntimeError("Missing profiler launch in runner log")
    for path in sorted(output.glob("profiler/OPPROF_*/*/*/OpBasicInfo_*.csv")):
        with path.open() as file:
            info = list(csv.DictReader(file))
        if len(info) != 1:
            raise RuntimeError(f"Unexpected profiling output: {path}")
        if not info[0]["Op Name"].startswith("SparseLightningIndexerGradKlLoss_"):
            raise RuntimeError(f"Unexpected operator: {path}")
        local_index = int(path.parent.name)
        info = info[0]
        index = observed.get((info["Op Name"], local_index))
        if index is None:
            raise RuntimeError(f"CSV kernel/sequence does not match runner log: {path}")
        rows.append({"launch_index": index, "case_id": configurations[index]["case_id"],
                     "task_duration_us": float(info["Task Duration(us)"]),
                     "profile": info, "csv": str(path)})
    if sorted(row["launch_index"] for row in rows) != list(range(len(configurations))):
        raise RuntimeError("Missing or duplicate profiler launches")
    return sorted(rows, key=lambda row: row["launch_index"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_FINAL_BUILD)
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--metrics", choices=("BasicInfo", "PipeUtilization"), default="BasicInfo")
    args = parser.parse_args()
    config_path = CONFIG
    candidate = load_mock_suite()
    root = existing_final_build_dir(args.build_dir)
    resource_check("cpu")
    binary, _ = build_runner(root, root)
    fixtures, configs = [], []
    for item in candidate["cases"]:
        case = CaseSpec(**item["spec"])
        fixture = materialize_mock_fixture(case, DEFAULT_FIXTURE_CACHE)
        fixtures.append(fixture)
        configs.append({**case.to_manifest(), "official_case": item["official_case"],
                        "workload_signature": workload_signature(case)})
    fixtures *= args.rounds
    configs *= args.rounds
    output = DEFAULT_RESULTS / ("mock-profile-" + time.strftime("%Y%m%d-%H%M%S"))
    output.mkdir(mode=0o700, parents=True)
    command = ["msprof", "op", "--aic-metrics=" + args.metrics, "--warm-up=5",
               "--launch-count=" + str(len(fixtures)), "--output=" + str(output / "profiler"),
               str(binary), "--length-api", "tensor", "--continue-comparison-errors",
               "--rtol", str(RTOL), "--atol", str(ATOL)]
    for fixture in fixtures:
        command.extend(("--fixture", str(fixture)))
    env = dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(root / "tmp/vendors/custom"),
               SLI_EXPECTED_OPAPI=str(root / "libcust_opapi.so"))
    env["LD_LIBRARY_PATH"] = str(root) + ":" + str(_toolkit_root() / "lib64") + ":" + env.get("LD_LIBRARY_PATH", "")
    manifest = {"suite": candidate, "data_version": DATA_VERSION,
                "suite_file": content_record(config_path, str(config_path)), "configurations": configs, "command": command,
                "runner": content_record(binary, str(binary)),
                "opapi": content_record(root / "libcust_opapi.so", str(root / "libcust_opapi.so")),
                "build_receipt": content_record(root / "tests/final_build_state.json", str(root / "tests/final_build_state.json")),
                "fixtures": [content_record(p / ".complete", str(p / ".complete")) for p in fixtures]}
    with serialized_npu(), (output / "profiler.log").open("w") as log:
        start = time.monotonic()
        result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                timeout=180)
        manifest.update(returncode=result.returncode, elapsed_seconds=time.monotonic() - start)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print("profile evidence:", output, "returncode:", result.returncode)
    completed = sum(" comparison=" in line and "status=PASS" in line
                    for line in (output / "profiler.log").read_text().splitlines())
    if result.returncode or completed != len(fixtures):
        raise SystemExit(f"Mock execution or output comparison failed; inspect {output / 'profiler.log'}")
    summary = read_profile(output, configs, output / "profiler.log")
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary:
        print(row["case_id"], "task_duration_us=", row["task_duration_us"])


if __name__ == "__main__":
    main()
