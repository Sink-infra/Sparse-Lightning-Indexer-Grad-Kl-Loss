"""Audit declared coverage; hardware execution is tracked separately."""
import argparse
import json
from itertools import product
from pathlib import Path
from tests.common.cases import CASE_GROUPS, CORE_DTYPES, GENERIC_N1, GENERIC_NIDX1, SUPPORTED_K


CASES = CASE_GROUPS["coverage"]


def audit():
    issues = []
    for case in CASES:
        heads = case.n1 in GENERIC_N1 and case.nidx1 in GENERIC_NIDX1
        if not (case.core_dtype in CORE_DTYPES and case.weights_dtype in (case.core_dtype, "float32")
                and (case.d_query, case.d_query_index, case.d_rope) == (512, 128, 64)
                and heads and case.k in SUPPORTED_K and 1 <= case.b <= 256
                and 0 <= case.s1 <= 8192 and max(1, case.s1) <= case.s2 <= 524288
                and case.layout == "BSND" and case.indices_layout == "BSNK"
                and case.statistics_layout == "BNSG" and case.actual_seq_dtype == "int64"
                and not case.omit_lengths):
            issues.append(f"Non-specification diagnostic case in formal matrix: {case.case_id}")
    dtype_cases = [c for c in CASES if "dtype_matrix" in c.tags]
    expected = {(core, w, qr, kr) for core in CORE_DTYPES
                for w, qr, kr in product((core, "float32"), CORE_DTYPES, CORE_DTYPES)}
    observed = {(c.core_dtype, c.weights_dtype, c.query_rope_dtype, c.key_rope_dtype) for c in dtype_cases}
    if observed != expected:
        issues.append("dtype/RoPE Cartesian product is incomplete")
    if not set(product(GENERIC_N1, GENERIC_NIDX1)) <= {(c.n1, c.nidx1) for c in CASES}:
        issues.append("head combinations are incomplete")
    if not set(SUPPORTED_K) <= {c.k for c in CASES}:
        issues.append("sparse sizes are incomplete")
    for field, bounds in (("b", {1, 256}), ("s1", {0, 1}), ("s2", {1, 524288})):
        if not bounds <= {getattr(c, field) for c in CASES}:
            issues.append(f"{field} boundary missing")
    if len({c.case_id for c in CASES}) != len(CASES):
        issues.append("duplicate case id")
    return {"status": "FAIL" if issues else "PASS", "issues": issues,
            "case_count": len(CASES), "dtype_combinations": len(observed),
            "stress_descriptors": sum(c.tier == "stress" for c in CASES),
            "nonzero_rope_assumption_cases": sum(c.rope_profile != "zero" for c in CASES),
            "hardware_coverage": "not inferred from declarations"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    report = audit()
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if report["issues"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
