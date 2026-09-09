"""Strict content-addressed storage for representative Ascend C kernels.

Only compiler outputs are cached.  Every reuse verifies the complete identity
manifest and the size/hash of each ``.o`` and JSON artifact.  Invalid entries
are moved to a recoverable rejected area instead of being reused or overwritten.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple


SCHEMA_VERSION = 1
ARTIFACT_SUFFIXES = (".o", ".json")
MAX_ARTIFACT_FILES = 256
MAX_TREE_DEPTH = 8


class CacheValidationError(RuntimeError):
    """Raised when a cache entry exists but cannot be trusted."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def content_record(path: Path, label: str) -> Dict[str, Any]:
    if not path.is_file() or path.is_symlink():
        raise CacheValidationError(f"cache identity input is not a regular file: {path}")
    return {"path": label, "size": path.stat().st_size, "sha256": sha256_file(path)}


def bounded_artifact_records(root: Path, *, max_files: int = MAX_ARTIFACT_FILES,
                             max_depth: int = MAX_TREE_DEPTH) -> List[Dict[str, Any]]:
    """Hash a bounded artifact tree without following symlinks."""

    if not root.is_dir() or root.is_symlink():
        raise CacheValidationError(f"artifact root is not a regular directory: {root}")
    pending: List[Tuple[Path, int]] = [(root, 0)]
    files: List[Path] = []
    while pending:
        directory, depth = pending.pop()
        if depth > max_depth:
            raise CacheValidationError(f"artifact tree exceeds depth {max_depth}: {root}")
        with os.scandir(directory) as entries:
            for entry in entries:
                path = Path(entry.path)
                if entry.is_symlink():
                    raise CacheValidationError(f"symlink is forbidden in artifact tree: {path}")
                if entry.is_dir(follow_symlinks=False):
                    pending.append((path, depth + 1))
                elif entry.is_file(follow_symlinks=False) and path.suffix in ARTIFACT_SUFFIXES:
                    files.append(path)
                    if len(files) > max_files:
                        raise CacheValidationError(
                            f"artifact tree exceeds {max_files} cacheable files: {root}"
                        )
    records = [content_record(path, path.relative_to(root).as_posix()) for path in sorted(files)]
    if not any(record["path"].endswith(".o") for record in records):
        raise CacheValidationError(f"artifact tree contains no Kernel object: {root}")
    if not any(record["path"].endswith(".json") for record in records):
        raise CacheValidationError(f"artifact tree contains no Kernel JSON: {root}")
    return records


def artifact_tree_digest(root: Path) -> str:
    records = bounded_artifact_records(root)
    return sha256_bytes(canonical_json(records).encode("utf-8"))


class KernelArtifactCache:
    """Immutable, content-addressed Kernel artifact cache."""

    def __init__(self, root: Path):
        self.root = root
        self.entries = root / "entries"
        self.locks = root / "locks"
        self.rejected = root / "rejected"

    @staticmethod
    def key_for(identity: Mapping[str, Any]) -> str:
        return sha256_bytes(canonical_json(identity).encode("utf-8"))

    def entry_path(self, key: str) -> Path:
        if len(key) != 64 or any(character not in "0123456789abcdef" for character in key):
            raise ValueError(f"invalid Kernel cache key: {key}")
        return self.entries / key[:2] / key

    @contextlib.contextmanager
    def lock(self, key: str) -> Iterator[None]:
        self.locks.mkdir(parents=True, exist_ok=True)
        lock_path = self.locks / f"{key}.lock"
        with lock_path.open("w", encoding="utf-8") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _read_manifest(self, entry: Path) -> Dict[str, Any]:
        path = entry / "manifest.json"
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as error:
            raise CacheValidationError(f"unreadable Kernel cache manifest: {path}: {error}") from error
        if not isinstance(value, dict):
            raise CacheValidationError(f"Kernel cache manifest is not an object: {path}")
        return value

    def validate(self, entry: Path, identity: Optional[Mapping[str, Any]] = None,
                 key: str = "") -> Dict[str, Any]:
        if not entry.is_dir() or entry.is_symlink():
            raise CacheValidationError(f"Kernel cache entry is not a regular directory: {entry}")
        manifest = self._read_manifest(entry)
        expected_key = key or self.key_for(manifest.get("identity", {}))
        if manifest.get("schema_version") != SCHEMA_VERSION:
            raise CacheValidationError(f"unsupported Kernel cache schema: {entry}")
        if manifest.get("cache_key") != expected_key or entry.name != expected_key:
            raise CacheValidationError(f"Kernel cache key/manifest mismatch: {entry}")
        if identity is not None and manifest.get("identity") != dict(identity):
            raise CacheValidationError(f"Kernel cache identity mismatch: {entry}")
        if self.key_for(manifest.get("identity", {})) != expected_key:
            raise CacheValidationError(f"Kernel cache identity digest mismatch: {entry}")
        records = manifest.get("artifacts")
        if not isinstance(records, list) or not records:
            raise CacheValidationError(f"Kernel cache manifest has no artifacts: {entry}")
        actual = bounded_artifact_records(entry / "artifacts")
        if actual != records:
            raise CacheValidationError(f"Kernel cache artifact hash/size mismatch: {entry}")
        return manifest

    def lookup(self, identity: Mapping[str, Any]) -> Optional[Tuple[Path, Dict[str, Any]]]:
        key = self.key_for(identity)
        entry = self.entry_path(key)
        if not entry.exists():
            return None
        return entry, self.validate(entry, identity, key)

    def lookup_key(self, key: str) -> Tuple[Path, Dict[str, Any]]:
        entry = self.entry_path(key)
        if not entry.exists():
            raise CacheValidationError(f"Kernel cache key does not exist: {key}")
        return entry, self.validate(entry, key=key)

    def quarantine(self, entry: Path, reason: str) -> Path:
        self.rejected.mkdir(parents=True, exist_ok=True)
        suffix = f"{time.time_ns()}-{os.getpid()}"
        destination = self.rejected / f"{entry.name}-{suffix}"
        entry.rename(destination)
        (destination / "rejection.txt").write_text(reason.rstrip() + "\n", encoding="utf-8")
        return destination

    def publish(self, identity: Mapping[str, Any], binary_root: Path,
                provenance: Mapping[str, Any]) -> Tuple[Path, Dict[str, Any]]:
        key = self.key_for(identity)
        destination = self.entry_path(key)
        if destination.exists():
            return destination, self.validate(destination, identity, key)
        destination.parent.mkdir(parents=True, exist_ok=True)
        temp_parent = self.root / "staging"
        temp_parent.mkdir(parents=True, exist_ok=True)
        temporary = Path(tempfile.mkdtemp(prefix=f"{key}.", dir=temp_parent))
        try:
            artifact_root = temporary / "artifacts"
            source_records = bounded_artifact_records(binary_root)
            for record in source_records:
                relative = Path(record["path"])
                source = binary_root / relative
                target = artifact_root / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
            records = bounded_artifact_records(artifact_root)
            manifest = {
                "schema_version": SCHEMA_VERSION,
                "cache_key": key,
                "identity": dict(identity),
                "artifacts": records,
                "provenance": dict(provenance),
            }
            manifest_path = temporary / "manifest.json"
            manifest_path.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            with manifest_path.open("rb") as stream:
                os.fsync(stream.fileno())
            temporary.rename(destination)
            return destination, self.validate(destination, identity, key)
        finally:
            if temporary.exists():
                shutil.rmtree(temporary)

    def materialize(self, entry: Path, destination: Path) -> None:
        manifest = self.validate(entry)
        staging = destination.parent / f".{destination.name}.staging-{os.getpid()}"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        for record in manifest["artifacts"]:
            relative = Path(record["path"])
            source = entry / "artifacts" / relative
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            try:
                os.link(source, target)
            except OSError:
                shutil.copy2(source, target)
        if destination.exists():
            shutil.rmtree(destination)
        staging.rename(destination)


def identity_files(paths: Iterable[Tuple[Path, str]]) -> List[Dict[str, Any]]:
    return [content_record(path, label) for path, label in paths]
