"""Preserve a verified historical package before reusing the fixed final directory."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

from tests.st.scripts.kernel_cache import content_record
from tests.st.scripts.run_tests import DEFAULT_FINAL_BUILD, ROOT, _git_commit

CATALOG = DEFAULT_FINAL_BUILD.parents[1] / "artifacts/relocations.json"


def commit_source(commit):
    paths = subprocess.check_output(["git", "ls-tree", "-r", "--name-only", commit, "--", "code"],
                                    cwd=ROOT, text=True).splitlines()
    if len(paths) != 7:
        raise ValueError("Unexpected operator source inventory in commit")
    digest = hashlib.sha256(b"source-without-git")
    for path in sorted(paths):
        digest.update(path.encode())
        digest.update(subprocess.check_output(["git", "show", commit + ":" + path], cwd=ROOT))
    return "source-sha256:" + digest.hexdigest()


def resolve_artifact(path, source):
    path = Path(path)
    if CATALOG.is_file():
        for relocation in json.loads(CATALOG.read_text()):
            old = Path(relocation["original_prefix"])
            if relocation["source"] == source and path.is_relative_to(old):
                return Path(relocation["archive_prefix"]) / path.relative_to(old)
    return path


def verify_recorded_build(build_dir, expected_source=None):
    """Validate the loaded package against its own receipt, including archives."""
    receipt = json.loads((build_dir / "tests/final_build_state.json").read_text())
    source = receipt["source"]
    if expected_source is not None and source != expected_source:
        raise ValueError("Recorded build source does not match the requested version")
    for record in receipt["artifacts"]:
        path = resolve_artifact(record["path"], source)
        if not path.is_relative_to(build_dir) or content_record(path, record["path"]) != record:
            raise ValueError("Recorded package artifact mismatch: " + str(path))
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    identity = parser.add_mutually_exclusive_group(required=True)
    identity.add_argument("--commit")
    identity.add_argument("--candidate", action="store_true",
                          help="Archive the actual uncommitted source and its package before a local revision")
    parser.add_argument("--build-dir", type=Path, default=DEFAULT_FINAL_BUILD)
    args = parser.parse_args()
    source = _git_commit() if args.candidate else commit_source(args.commit)
    receipt_path = args.build_dir / "tests/final_build_state.json"
    receipt = json.loads(receipt_path.read_text())
    if receipt["source"] != source:
        raise ValueError("Package does not correspond to the requested commit")
    for record in receipt["artifacts"]:
        if content_record(Path(record["path"]), record["path"]) != record:
            raise ValueError("Historical package changed before archival")
    identity_name = "candidate-" + source.removeprefix("source-sha256:") if args.candidate else args.commit
    destination = CATALOG.parent / identity_name / "final_package"
    if destination.exists():
        raise ValueError("Archive destination exists; no overwrite permitted")
    relocations = json.loads(CATALOG.read_text()) if CATALOG.is_file() else []
    record = {"commit": args.commit, "source": source,
              "original_prefix": str(args.build_dir), "archive_prefix": str(destination),
              "receipt_sha256": hashlib.sha256(receipt_path.read_bytes()).hexdigest(),
              "created_at": datetime.now(timezone.utc).isoformat()}
    destination.parent.mkdir(parents=True)
    if args.candidate:
        snapshot = destination.parent / "operator_source"
        snapshot.mkdir()
        record["source_snapshot"] = str(snapshot)
        record["source_artifacts"] = []
        for path in sorted((ROOT / "code").rglob("*")):
            if not path.is_file():
                continue
            target = snapshot / path.relative_to(ROOT)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
            record["source_artifacts"].append(content_record(target, str(target)))
    args.build_dir.rename(destination)
    CATALOG.write_text(json.dumps([*relocations, record], indent=2) + "\n")
    print("archived verified historical build:", destination)
    print("fixed final build directory is now absent:", args.build_dir)


if __name__ == "__main__":
    main()
