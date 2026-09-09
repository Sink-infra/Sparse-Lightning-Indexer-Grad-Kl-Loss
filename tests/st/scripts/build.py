"""Focused Ascend build entry point with one incremental and one final tree.

Examples:
  python3 -m tests.st.scripts.build --target bootstrap
  python3 -m tests.st.scripts.build --target tiling
  python3 -m tests.st.scripts.build --target schema
  python3 -m tests.st.scripts.build --target kernel --representative-case baseline_f16
  python3 -m tests.st.scripts.build --target final

``bootstrap`` is the only development target allowed to populate all generated
artifacts in the incremental tree. Host targets never fall back to it. ``final``
requires an absent or empty final tree and performs one all-template build.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from tests.st.scripts.run_tests import (
    StepFailure,
    build_full,
    build_host,
    configure,
    build_representative_kernel,
    clean_final_build_dir,
    incremental_build_dir,
    kernel_cache_dir,
    resolve_jobs,
)


TARGETS = ("bootstrap", "tiling", "schema", "host", "kernel", "final")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--target", required=True, choices=TARGETS)
    parser.add_argument("--jobs", type=int, help="bounded parallelism; env override or 28")
    parser.add_argument("--build-dir", type=Path, help="single reusable incremental tree")
    parser.add_argument("--final-build-dir", type=Path, help="absent or empty final tree")
    parser.add_argument("--kernel-cache", type=Path, help="content-addressed Kernel cache")
    parser.add_argument("--representative-case", default="baseline_f16")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args.jobs, os.environ)
    incremental = incremental_build_dir(args.build_dir)
    kernel_cache = kernel_cache_dir(args.kernel_cache)
    if args.target == "bootstrap":
        elapsed = configure(incremental, jobs)
        elapsed += build_host(incremental, jobs)
        active = incremental
    elif args.target in ("tiling", "schema", "host"):
        component = "all" if args.target == "host" else args.target
        elapsed = build_host(incremental, jobs, component)
        active = incremental
    elif args.target == "kernel":
        elapsed, _ = build_representative_kernel(
            incremental, jobs, args.representative_case, cache_dir=kernel_cache
        )
        active = incremental
    else:
        active = clean_final_build_dir(args.final_build_dir)
        elapsed = build_full(active, jobs)
    print(f"PASS target={args.target} jobs={jobs} build_dir={active} elapsed={elapsed:.3f}s")


if __name__ == "__main__":
    try:
        main()
    except (StepFailure, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(error.returncode if isinstance(error, StepFailure) and error.returncode else 2)
