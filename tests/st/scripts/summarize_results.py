"""Summarize one completed runner invocation, retaining its exact configurations."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics

OUTPUTS = ("dQueryIndex", "dKeyIndex", "dWeights", "loss")


def summarize(run_dir):
    manifest = json.loads((run_dir / "manifest.json").read_text())
    records = manifest["artifacts"] + manifest["fixtures"]
    for record in records:
        path = Path(record["path"])
        if path.stat().st_size != record["size"]:
            raise ValueError(f"recorded file size changed: {path}")
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
        if digest.hexdigest() != record["sha256"]:
            raise ValueError(f"recorded file contents changed: {path}")
    cases = {}
    for record in manifest["fixtures"]:
        if Path(record["path"]).name == "case.json":
            config = json.loads(Path(record["path"]).read_text())
            cases[config["case_id"]] = {
                "configuration": config, "outputs": {}, "device_us": [],
                "zero_key_checks": [], "completion": {},
                "log_path": str(run_dir / "runner.log"),
            }
    if not cases:
        raise ValueError("manifest has no case configurations")
    identity_ok = False
    for line in (run_dir / "runner.log").read_text().splitlines():
        fields = dict(word.split("=", 1) for word in line.split() if "=" in word)
        if "loaded_opapi" in fields:
            identity_ok = fields.get("identity") == "PASS"
        if "case" not in fields:
            continue
        case = cases[fields["case"]]
        if "output" in fields:
            case["outputs"].setdefault(fields["output"], []).append({
                "repetition": int(fields["repetition"]), "status": fields["status"],
                "max_abs": float(fields["max_abs"]), "max_rel": float(fields["max_rel"]),
            })
        if "kernel_device_us" in fields:
            case["device_us"].append(float(fields["kernel_device_us"]))
        if "unselected_key_rows_zero" in fields:
            case["zero_key_checks"].append({"status": fields["unselected_key_rows_zero"],
                                             "checked_rows": int(fields["checked_rows"])})
        if "network_deterministic" in fields:
            case["network_deterministic"] = int(fields["network_deterministic"])
        if "comparison" in fields:
            case["completion"] = fields
    passed = manifest.get("returncode") == 0 and identity_ok
    for case in cases.values():
        completion = case["completion"]
        repetitions = case["configuration"]["repetitions"]
        ok = completion.get("status") == "PASS" and len(case["device_us"]) == repetitions
        if completion.get("comparison") != "launch_only":
            for name in OUTPUTS:
                values = case["outputs"].get(name, [])
                ok &= (len(values) == repetitions and
                       {v["repetition"] for v in values} == set(range(repetitions)) and
                       all(v["status"] == "PASS" for v in values))
            ok &= len(case["zero_key_checks"]) == repetitions
            ok &= all(v["status"] == "PASS" for v in case["zero_key_checks"])
        ok &= case.get("network_deterministic") == int(case["configuration"]["deterministic"])
        case["status"] = "PASS" if ok else "FAIL"
        case["median_device_us"] = statistics.median(case["device_us"]) if case["device_us"] else None
        # Runner compares output bytes to repetition zero and fails completion
        # on any difference; the conclusion is justified only after completion.
        case["repeat_consistency"] = "PASS" if ok and repetitions > 1 else "not_requested"
        passed &= ok
    return {"status": "PASS" if passed else "FAIL", "case_count": len(cases),
            "source": manifest["source"], "manifest_path": str(run_dir / "manifest.json"),
            "elapsed_seconds": manifest.get("elapsed_seconds"),
            "artifact_hashes_verified": len(records), "loaded_opapi_verified": identity_ok,
            "accuracy_policy": "tests/README.md: local dtype-specific atol + rtol * abs(reference)",
            "timing_policy": "correctness runs; first samples include launch overhead; no speedup claim",
            "cases": cases}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    report = summarize(args.run_dir.resolve())
    output = args.run_dir / "summary.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"{report['status']} cases={report['case_count']} report={output}")
    if report["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
