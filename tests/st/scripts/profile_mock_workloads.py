"""Run and profile the six fixed Mock workloads on the current built operator."""
import argparse
import csv
import json
import os
import signal
from pathlib import Path
import re
import subprocess
import time
import numpy as np

from tests.common.cases import CaseSpec
from tests.common.mock_cases import (
    CONFIG, DATA_VERSION, RTOL, ATOL, generate_mock_inputs, generate_cancellation_inputs, load_mock_suite, materialize_mock_fixture,
)
from tests.common.reference import canonical_arrays
from tests.st.scripts.archive_build import commit_source, verify_recorded_build
from tests.st.scripts.kernel_cache import content_record
from tests.st.scripts.run_tests import (
    DEFAULT_FINAL_BUILD, DEFAULT_FIXTURE_CACHE, DEFAULT_RESULTS, ROOT,
    build_runner, existing_final_build_dir, resource_check, serialized_npu, _toolkit_root,
)


def workload_signature(case, profile="official"):
    generator = generate_mock_inputs if profile == "official" else generate_cancellation_inputs
    data = canonical_arrays(generator(case), case)
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
                        "gradient_elements": 2 * positive * case.d_query_index,
                        "input_nonzero_elements": {name: int(np.count_nonzero(data[name][batch]))
                            for name in ("query", "key", "queryIndex", "keyIndex", "weights", "queryRope", "keyRope")}})
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


MAX_LOG_BYTES = 16 << 20


def _comparison_outcome(log, case_id, rounds, returncode, timed_out):
    """Require all four numerical outputs and a completed comparison each round."""
    if timed_out or returncode:
        return {"status": "FAIL", "reason": "timeout" if timed_out else "process_failed"}
    prefix = "case=" + case_id + " "
    blocks = [block for block in re.split(r"(?=^case=\S+ network_deterministic=)", log, flags=re.M)
              if block.startswith(prefix + "network_deterministic=")]
    if len(blocks) != rounds:
        return {"status": "INFRA", "reason": "missing_or_duplicate_case_rounds"}
    expected = {"dQueryIndex", "dKeyIndex", "dWeights", "loss"}
    for block in blocks:
        outputs, completed = [], []
        for line in block.splitlines():
            if line.startswith(prefix + "output="):
                fields = dict(re.findall(r"(\w+)=([^\s]+)", line))
                outputs.append(fields["output"])
                if fields.get("status") != "PASS" or fields.get("error_count") != "0":
                    return {"status": "FAIL", "reason": "output_comparison_failed"}
            elif line.startswith(prefix + "comparison="):
                completed.append(line.startswith(prefix + "comparison=exact ") and line.endswith(" status=PASS"))
        if len(outputs) != 4 or set(outputs) != expected or len(completed) != 1:
            return {"status": "INFRA", "reason": "missing_or_duplicate_comparison_results"}
        if not completed[0]:
            return {"status": "FAIL", "reason": "case_comparison_failed"}
    return {"status": "PASS"}


def _stop_process_group(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def profile_cases(root, specs, output, *, rounds=3, metrics="PipeUtilization",
                  expected_source=None, timeout=180, input_profile="official", device=0):
    """Measure an immutable package and require complete numerical comparisons."""
    root, output = Path(root).resolve(), Path(output).resolve()
    source = verify_recorded_build(root, expected_source)
    resource_check("cpu")
    binary, _ = build_runner(root, root)
    fixtures, configurations = [], []
    for case in specs:
        fixture = materialize_mock_fixture(case, DEFAULT_FIXTURE_CACHE, input_profile)
        fixtures.append(fixture)
        configurations.append({**case.to_manifest(), "workload_signature": workload_signature(case, input_profile)})
    fixtures *= rounds
    configurations *= rounds
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    command = ["msprof", "op", "--aic-metrics=" + metrics, "--warm-up=5",
               "--launch-count=" + str(len(fixtures)), "--output=" + str(output / "profiler"),
               str(binary), "--device", str(device), "--length-api", "tensor", "--continue-comparison-errors",
               "--rtol", str(RTOL), "--atol", str(ATOL)]
    for fixture in fixtures:
        command.extend(("--fixture", str(fixture)))
    env = dict(os.environ, ASCEND_CUSTOM_OPP_PATH=str(root / "tmp/vendors/custom"),
               SLI_EXPECTED_OPAPI=str(root / "libcust_opapi.so"))
    env["LD_LIBRARY_PATH"] = str(root) + ":" + str(_toolkit_root() / "lib64") + ":" + env.get("LD_LIBRARY_PATH", "")
    manifest = {"source_identity": source, "data_version": DATA_VERSION, "input_profile": input_profile,
                "suite_file": content_record(CONFIG, str(CONFIG)), "configurations": configurations,
                "command": command, "rounds": rounds, "warmup": 5, "metrics": metrics,
                "timeout_seconds": timeout, "rtol": RTOL, "atol": ATOL,
                "runner": content_record(binary, str(binary)),
                "opapi": content_record(root / "libcust_opapi.so", str(root / "libcust_opapi.so")),
                "build_receipt": content_record(root / "tests/final_build_state.json", str(root / "tests/final_build_state.json")),
                "test_sources": [content_record(p, str(p)) for p in
                    (Path(__file__), ROOT / "tests/common/mock_cases.py", ROOT / "tests/common/reference.py",
                     ROOT / "tests/st/scripts/gen_data.py")],
                "fixtures": [content_record(p / ".complete", str(p / ".complete")) for p in fixtures]}
    timed_out = False
    with serialized_npu(), (output / "profiler.log").open("w") as log:
        start = time.monotonic()
        process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        try:
            returncode = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            _stop_process_group(process)
            returncode = process.returncode
        except BaseException:
            _stop_process_group(process)
            raise
    manifest.update(returncode=returncode, timed_out=timed_out, elapsed_seconds=time.monotonic() - start)
    path = output / "profiler.log"
    if path.stat().st_size > MAX_LOG_BYTES:
        log_text = ""
        profile_error = "runner log exceeds bounded parser limit"
    else:
        log_text = path.read_text(errors="replace")
        profile_error = None
    outcomes = {c.case_id: _comparison_outcome(log_text, c.case_id, rounds, returncode, timed_out) for c in specs}
    rows = []
    try:
        if profile_error is None:
            rows = read_profile(output, configurations, path)
    except (RuntimeError, KeyError, ValueError, OSError) as error:
        profile_error = str(error)[:2000]
    for case in specs:
        local = outcomes[case.case_id]
        local["duration_us"] = [row["task_duration_us"] for row in rows if row["case_id"] == case.case_id]
        if local["status"] == "PASS" and (profile_error or len(local["duration_us"]) != rounds):
            local.update(status="INFRA", reason=profile_error or "missing_timing_rounds")
    manifest["profile_error"] = profile_error
    manifest["log"] = content_record(path, str(path))
    manifest["profile_csvs"] = [content_record(p, str(p)) for p in sorted(output.glob("profiler/OPPROF_*/*/*/*.csv"))]
    for fixture in set(fixtures):
        records = json.loads((fixture / ".complete").read_text())
        for name, expected_hash in records.items():
            if content_record(fixture / name, name)["sha256"] != expected_hash:
                raise RuntimeError("Mock fixture changed during execution: " + str(fixture / name))
    verify_recorded_build(root, source)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    (output / "summary.json").write_text(json.dumps(rows, indent=2) + "\n")
    (output / "outcomes.json").write_text(json.dumps(outcomes, indent=2) + "\n")
    return outcomes, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_FINAL_BUILD)
    parser.add_argument("--source-commit", help="Exact local commit for a historical package; never fetches")
    parser.add_argument("--rounds", type=int, choices=range(1, 6), default=3)
    parser.add_argument("--metrics", choices=("BasicInfo", "PipeUtilization"), default="PipeUtilization")
    parser.add_argument("--input-profile", choices=("official", "cancellation"), default="official")
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--device", type=int, default=0)
    args = parser.parse_args()
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    suite = load_mock_suite()
    specs = [CaseSpec(**item["spec"]) for item in suite["cases"]]
    if args.case:
        if set(args.case) - {c.case_id for c in specs}:
            parser.error("unknown Mock case")
        specs = [c for c in specs if c.case_id in args.case]
    elif args.input_profile == "cancellation":
        specs = [c for c in specs if c.core_dtype == "float32"]
    source = commit_source(args.source_commit) if args.source_commit else None
    root = args.build_dir if source else existing_final_build_dir(args.build_dir)
    output = DEFAULT_RESULTS / ("mock-profile-" + time.strftime("%Y%m%d-%H%M%S") + f"-{os.getpid()}")
    outcomes, _ = profile_cases(root, specs, output, rounds=args.rounds, metrics=args.metrics,
                                expected_source=source, timeout=args.timeout,
                                input_profile=args.input_profile, device=args.device)
    print("profile evidence:", output)
    for name, result in outcomes.items():
        print(name, "status=" + result["status"], "task_duration_us=", result["duration_us"])
    if any(value["status"] != "PASS" for value in outcomes.values()):
        raise SystemExit("Mock execution, comparison, or profiling failed; inspect recorded evidence")


if __name__ == "__main__":
    main()
