"""Layered build and validation entry point for operator development.

The default quick pipeline gates static/contract checks, Host compilation,
Workspace queries, one representative Kernel, smoke, and directed correctness.
Final mode adds one fixed clean all-template tree and the complete 910B matrix.
Every layer can run independently; no CMake source file is rewritten.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from tests.common.cases import accepted_cases, get_case
from tests.st.scripts.gen_data import (
    DEFAULT_MAX_BYTES,
    case_data_mode,
    materialize_runner_fixture,
)
from tests.st.scripts.kernel_cache import (
    CacheValidationError,
    KernelArtifactCache,
    content_record,
    identity_files,
    sha256_file,
)


ROOT = Path(__file__).resolve().parents[3]
CODE_DIR = ROOT / "code"
RUNNER_SOURCE = ROOT / "tests/st/runner/sparse_lightning_indexer_grad_kl_loss_runner.cpp"
DEFAULT_WORK_ROOT = Path(os.environ.get("SLI_WORK_ROOT", "/tmp/sparse_lightning_indexer_grad_kl_loss_new"))
DEFAULT_INCREMENTAL_BUILD = DEFAULT_WORK_ROOT / "build/incremental"
DEFAULT_FIXTURE_CACHE = DEFAULT_WORK_ROOT / "fixtures"
DEFAULT_KERNEL_CACHE = DEFAULT_WORK_ROOT / "kernel-cache"
DEFAULT_RESULTS = DEFAULT_WORK_ROOT / "results"
DEFAULT_FINAL_BUILD = DEFAULT_WORK_ROOT / "build/final"
FULL_TARGET = "custom"
HOST_COMPONENTS = ("tiling", "schema", "all")
LAYERS = (
    "static", "host", "workspace", "kernel", "smoke", "correctness",
    "full-build", "full-test", "full", "pipeline",
)
DTYPE_CASES = (
    "uniform_distributions",
    "baseline_f16",
    "weights_fp32_with_f16",
    "baseline_bf16",
    "weights_fp32_with_bf16",
)


class StepFailure(RuntimeError):
    def __init__(self, message: str, returncode: Optional[int] = None):
        super().__init__(message)
        self.returncode = returncode


def resource_check(kind: str) -> None:
    snapshot = subprocess.run(("ps", "-eo", "pid,comm,pcpu", "--sort=-pcpu"),
                              text=True, capture_output=True, check=True).stdout.splitlines()
    relevant = [line for line in snapshot[1:] if any(
        word in line.split()[1:2] for word in ("runner", "ccec", "bisheng", "cc1plus", "cc1"))]
    print(f"resource check {kind}: load={os.getloadavg()} active={relevant[:12]}", flush=True)
    if kind == "cpu":
        compiler_load = sum(float(line.split()[-1]) for line in relevant
                            if line.split()[1] != "runner")
        if compiler_load > 200:
            raise StepFailure("another compilation is active; retry after it finishes")
    else:
        snapshot = subprocess.run(("npu-smi", "info"), text=True,
                                  capture_output=True, check=True).stdout
        process_rows = [line for line in snapshot.splitlines()
                        if re.search(r"\|\s*\d+\s+\d+\s*\|\s*\d+\s*\|", line)]
        print(f"NPU process check: {process_rows or 'idle'}", flush=True)
        if process_rows:
            raise StepFailure("another NPU process is active; no device work was started")


def _render_command(command: Sequence[str], max_chars: int = 2048) -> str:
    rendered = shlex.join(str(part) for part in command)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]} ... [{len(command)} arguments; command truncated]"


def resolve_jobs(command_line_jobs: Optional[int], environment: Mapping[str, str]) -> int:
    """Resolve bounded build parallelism, preserving an explicit override."""

    raw = command_line_jobs
    if raw is None:
        configured = environment.get("CMAKE_BUILD_PARALLEL_LEVEL")
        raw = int(configured) if configured else 28
    if not 24 <= raw <= 28:
        raise ValueError("build parallelism must be between 24 and 28")
    return raw


def _run(command: Sequence[str], *, env: Optional[Mapping[str, str]] = None,
         cwd: Path = ROOT) -> float:
    rendered = _render_command(command)
    print(f"+ {rendered}", flush=True)
    started = time.monotonic()
    completed = subprocess.run(command, cwd=cwd, env=None if env is None else dict(env), check=False)
    elapsed = time.monotonic() - started
    if completed.returncode != 0:
        raise StepFailure(f"command failed ({completed.returncode}): {rendered}", completed.returncode)
    print(f"completed in {elapsed:.3f}s: {rendered}", flush=True)
    return elapsed


def _bounded_log_excerpt(path: Path, failed: bool, max_bytes: int = 256 * 1024,
                         max_lines: int = 80) -> str:
    """Read only the tail of a build log and retain bounded error context."""

    with path.open("rb") as stream:
        size = stream.seek(0, os.SEEK_END)
        stream.seek(max(0, size - max_bytes))
        text = stream.read(max_bytes).decode("utf-8", errors="replace")
    lines = text.splitlines()
    if not failed:
        return "\n".join(lines[-min(max_lines, 12):])
    pattern = re.compile(r"error:|fatal:|failed|no rule to make target|undefined reference", re.IGNORECASE)
    selected = set()
    for index, line in enumerate(lines):
        if pattern.search(line):
            selected.update(range(max(0, index - 3), min(len(lines), index + 4)))
    if not selected:
        return "\n".join(lines[-max_lines:])
    ordered = sorted(selected)[-max_lines:]
    return "\n".join(lines[index] for index in ordered)


def _run_build(command: Sequence[str], build_dir: Path, label: str, jobs: int,
               cwd: Path = ROOT) -> float:
    """Run one build command with bounded console output and a reusable log."""

    resource_check("cpu")
    log_dir = build_dir / "tests/logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{label}.log"
    rendered = _render_command(command)
    print(f"+ {rendered}  [log: {log_path}]", flush=True)
    started = time.monotonic()
    with log_path.open("wb") as stream:
        completed = subprocess.run(
            command, cwd=cwd, env=_build_environment(jobs), stdout=stream,
            stderr=subprocess.STDOUT, check=False,
        )
    elapsed = time.monotonic() - started
    excerpt = _bounded_log_excerpt(log_path, failed=completed.returncode != 0)
    if excerpt:
        print(excerpt, flush=True)
    if completed.returncode != 0:
        raise StepFailure(
            f"build command failed ({completed.returncode}); bounded context above, full log: {log_path}",
            completed.returncode,
        )
    print(f"completed in {elapsed:.3f}s: {label}", flush=True)
    return elapsed


def _capture(command: Sequence[str]) -> str:
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if completed.returncode != 0:
        raise StepFailure(f"command failed ({completed.returncode}): {shlex.join(command)}\n{completed.stdout}")
    return completed.stdout


def static_checks() -> float:
    started = time.monotonic()
    baseline = json.loads((ROOT / "tests/common/source_baseline.json").read_text())
    current = {str(p.relative_to(ROOT)) for p in CODE_DIR.rglob("*") if p.is_file()}
    if current != set(baseline):
        raise StepFailure("code/ inventory differs from the original seven files")
    forbidden = re.compile(r"\b(?:printf|cout|cerr|puts|write|syscall|dlopen|dlsym|asm|freopen)\b")
    for relative in current:
        path = ROOT / relative
        if path.name == "CMakeLists.txt":
            if sha256_file(path) != baseline[relative]:
                raise StepFailure(f"CMakeLists changes are forbidden: {relative}")
        elif forbidden.search(path.read_text()):
            raise StepFailure(f"forbidden diagnostic/system construct in {relative}")
    if (ROOT / ".git").exists():
        _run(("git", "diff", "--check"))
    print("static constraints: PASS (original CMake hashes and code inventory; forbidden APIs absent)", flush=True)
    return time.monotonic() - started


def python_checks() -> Dict[str, float]:
    return {
        "python_ut": _run(("python3", "-m", "unittest", "discover", "-s", "tests/ut", "-v")),
        "coverage_audit": _run(("python3", "-m", "tests.st.scripts.audit_coverage")),
    }


def _build_environment(jobs: int) -> Dict[str, str]:
    environment = dict(os.environ)
    environment["CMAKE_BUILD_PARALLEL_LEVEL"] = str(jobs)
    environment["OMP_NUM_THREADS"] = "1"
    environment["OPENBLAS_NUM_THREADS"] = "1"
    return environment


def configure(build_dir: Path, jobs: int) -> float:
    build_dir.mkdir(parents=True, exist_ok=True)
    return _run_build(("cmake", "-S", str(CODE_DIR), "-B", str(build_dir)),
                      build_dir, "configure", jobs)


def _kernel_binary_snapshot(build_dir: Path) -> Tuple[Tuple[str, int, int], ...]:
    kernel_dir = (
        build_dir / "op_kernel/ascendc_kernels/binary/ascend910b"
        / "sparse_lightning_indexer_grad_kl_loss"
    )
    if not kernel_dir.is_dir():
        return ()
    return tuple(
        sorted((path.name, path.stat().st_mtime_ns, path.stat().st_size)
               for path in kernel_dir.glob("*.o"))
    )


def _require_schema_bootstrap(build_dir: Path) -> None:
    required = (
        build_dir / "autogen/aic-ascend910b-ops-info.ini",
        build_dir / "autogen/aclnn_sparse_lightning_indexer_grad_kl_loss.h",
    )
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise StepFailure(
            "Schema/ACLNN fast build needs one explicitly bootstrapped incremental tree; "
            "run `python3 -m tests.st.scripts.build --target bootstrap` once. Missing: "
            + ", ".join(missing)
        )


def build_host(build_dir: Path, jobs: int, component: str = "all") -> float:
    if component not in HOST_COMPONENTS:
        raise ValueError(f"unknown Host component: {component}")
    if not (build_dir / "CMakeCache.txt").is_file():
        configure(build_dir, jobs)
    if component in ("schema", "all"):
        _require_schema_bootstrap(build_dir)
        configure(build_dir, jobs)

    before = _kernel_binary_snapshot(build_dir)
    elapsed = 0.0
    if component in ("tiling", "all"):
        elapsed += _run_build(
            ("cmake", "--build", str(build_dir), "--target", "cust_optiling",
             "--parallel", str(jobs)), build_dir, "host-cust-optiling", jobs
        )
        elapsed += _run_build(
            ("cmake", "--build", str(build_dir), "--target",
             "custom_ascendc_cust_optiling/fast", "--parallel", str(jobs)),
            build_dir, "host-link-optiling", jobs,
        )
    if component in ("schema", "all"):
        _, schema_time = build_host_only_opapi(build_dir, jobs)
        elapsed += schema_time
        elapsed += _run_build(
            ("cmake", "--build", str(build_dir), "--target",
             "ascendc_kernels_ops_info_gen_ascend910b", "--parallel", str(jobs)),
            build_dir, "host-op-info", jobs,
        )
    after = _kernel_binary_snapshot(build_dir)
    if before != after:
        raise StepFailure("Host-only build modified Kernel binaries; refusing hidden template rebuild")
    if component in ("tiling", "all"):
        _write_host_tiling_state(build_dir)
    print(f"Host {component} targets: PASS; Kernel binary snapshot unchanged ({len(after)} files)")
    return elapsed


def _host_tiling_state_identity(build_dir: Path) -> Dict[str, object]:
    generated_config = build_dir / "autogen/aic-ascend910b-ops-info.ini"
    generated_json = (
        build_dir / "op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b"
        / "aic-ascend910b-ops-info.json"
    )
    paths = (
        (CODE_DIR / "op_host/sparse_lightning_indexer_grad_kl_loss.cpp",
         "code/op_host/sparse_lightning_indexer_grad_kl_loss.cpp"),
        (CODE_DIR / "op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h",
         "code/op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h"),
        (CODE_DIR / "op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h",
         "code/op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h"),
        (CODE_DIR / "CMakeLists.txt", "code/CMakeLists.txt"),
        (CODE_DIR / "op_host/CMakeLists.txt", "code/op_host/CMakeLists.txt"),
        (CODE_DIR / "op_kernel/CMakeLists.txt", "code/op_kernel/CMakeLists.txt"),
        (generated_config, "autogen/aic-ascend910b-ops-info.ini"),
        (generated_json, "op_info/aic-ascend910b-ops-info.json"),
    )
    toolkit = _toolkit_root()
    host_compiler = Path(shutil.which("c++") or "/usr/bin/c++").resolve()
    if not host_compiler.is_file():
        raise StepFailure(f"Host C++ compiler is missing: {host_compiler}")
    return {
        "schema_version": 1,
        "inputs": identity_files(paths),
        "cann_version": content_record(
            (toolkit / "compiler/version.info").resolve(), "compiler/version.info"
        ),
        "host_compiler": {
            "path": str(host_compiler),
            "sha256": sha256_file(host_compiler),
            "version": _capture_version((str(host_compiler), "--version")),
        },
        "cmake": _cmake_generation_options(build_dir),
    }


def _host_tiling_state_path(build_dir: Path) -> Path:
    return build_dir / "tests/host_tiling_state.json"


def _write_host_tiling_state(build_dir: Path) -> None:
    library = build_dir / "op_host/libcustom_ascendc_cust_optiling.so"
    state = {
        "identity": _host_tiling_state_identity(build_dir),
        "library": content_record(library, "libcustom_ascendc_cust_optiling.so"),
    }
    path = _host_tiling_state_path(build_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )


def _host_tiling_state_current(build_dir: Path) -> bool:
    path = _host_tiling_state_path(build_dir)
    library = build_dir / "op_host/libcustom_ascendc_cust_optiling.so"
    if not path.is_file() or not library.is_file():
        return False
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
        return (
            state.get("identity") == _host_tiling_state_identity(build_dir)
            and state.get("library") == content_record(
                library, "libcustom_ascendc_cust_optiling.so"
            )
        )
    except (CacheValidationError, KeyError, OSError, json.JSONDecodeError):
        return False


def ensure_host_tiling_for_kernel(build_dir: Path, jobs: int) -> float:
    if _host_tiling_state_current(build_dir):
        print("Host/Tiling identity HIT; verified source, generation, CANN and library hashes", flush=True)
        return 0.0
    elapsed = build_host(build_dir, jobs, "all")
    if not _host_tiling_state_current(build_dir):
        raise StepFailure("Host/Tiling state failed post-build identity verification")
    return elapsed


def build_full(build_dir: Path, jobs: int) -> float:
    source_before = _git_commit()
    if not (build_dir / "CMakeCache.txt").is_file():
        configure(build_dir, jobs)
    elapsed = _run_build(
        ("cmake", "--build", str(build_dir), "--target", FULL_TARGET, "--parallel", str(jobs)),
        build_dir, "full-templates", jobs,
    )
    if _git_commit() != source_before:
        raise StepFailure("operator sources changed during the final build")
    _write_final_build_state(build_dir)
    return elapsed


def _final_build_identity(build_dir: Path) -> Dict[str, object]:
    package = build_dir / "tmp/vendors/custom"
    paths = [build_dir / "libcust_opapi.so",
             build_dir / "op_host/libcustom_ascendc_cust_optiling.so",
             build_dir / "op_host/CMakeFiles/cust_optiling.dir/sparse_lightning_indexer_grad_kl_loss.cpp.o"]
    paths.extend(p for p in package.rglob("*")
                 if p.is_file() and p.suffix in (".o", ".json", ".so"))
    if not package.is_dir() or len(paths) > 256:
        raise StepFailure("missing or unexpectedly large final operator package")
    return {"source": _git_commit(), "host": _host_tiling_state_identity(build_dir),
            "artifacts": [content_record(p, str(p)) for p in sorted(paths)]}


def _write_final_build_state(build_dir: Path) -> None:
    state = build_dir / "tests/final_build_state.json"
    state.parent.mkdir(parents=True, exist_ok=True)
    state.write_text(json.dumps(_final_build_identity(build_dir), indent=2) + "\n")


def _toolkit_root() -> Path:
    for variable in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        if os.environ.get(variable):
            path = Path(os.environ[variable]).resolve()
            if path.is_dir():
                return path
    compiler = shutil.which("bisheng")
    if compiler:
        path = Path(compiler).parent.parent
        if (path / "compiler/tikcpp").is_dir() and (path / "include").is_dir():
            return path
    raise StepFailure("ASCEND_TOOLKIT_HOME or ASCEND_HOME_PATH must identify the CANN toolkit")


def _capture_version(command: Sequence[str]) -> str:
    completed = subprocess.run(
        command, cwd=ROOT, check=False, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    if completed.returncode != 0:
        raise StepFailure(
            f"cannot identify compiler version ({completed.returncode}): "
            f"{_render_command(command)}\n{completed.stdout[-2048:]}"
        )
    return completed.stdout.strip()


def _cmake_generation_options(build_dir: Path) -> Dict[str, str]:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        raise StepFailure(f"Kernel cache identity requires configured CMake cache: {cache}")
    selected: Dict[str, str] = {}
    pattern = re.compile(
        r"^(ASCEND_COMPUTE_UNIT|CMAKE_BUILD_TYPE|CMAKE_(?:ASC|C|CXX)_FLAGS(?:_[A-Z]+)?):[^=]*=(.*)$"
    )
    for line in cache.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.match(line)
        if match:
            selected[match.group(1)] = match.group(2)
    return selected


def _compiler_identity(toolkit: Path, compile_script: Path) -> Dict[str, object]:
    compiler = shutil.which("bisheng") or str(toolkit / "bin/bisheng")
    compiler_path = Path(compiler).resolve()
    version_info = (toolkit / "compiler/version.info").resolve()
    if not compiler_path.is_file():
        raise StepFailure(f"BiSheng compiler is missing: {compiler_path}")
    if not version_info.is_file():
        raise StepFailure(f"CANN compiler version metadata is missing: {version_info}")
    return {
        "cann_version_info": content_record(version_info, "compiler/version.info"),
        "bisheng": {
            "path": str(compiler_path),
            "sha256": sha256_file(compiler_path),
            "version": _capture_version((str(compiler_path), "--version")),
        },
        "compile_driver": content_record(compile_script, "ascendc_compile_kernel.py"),
        "python": sys.version,
    }


def _git_commit() -> str:
    # Build identity follows the actual files, including uncommitted changes.
    # Adding Git metadata must not invalidate binaries built from these files.
    paths = sorted(p for p in CODE_DIR.rglob("*") if p.is_file())
    return "source-sha256:" + _content_digest(paths, "source-without-git")


def _dtype_column(config_text: str, case_id: str) -> int:
    case = get_case(case_id)
    wanted = (
        case.input_dtypes()["query"],
        case.input_dtypes()["weights"],
        case.input_dtypes()["queryRope"],
        case.input_dtypes()["keyRope"],
    )
    columns: Dict[str, List[str]] = {}
    for line in config_text.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key in ("input0.dtype", "input4.dtype", "input8.dtype", "input9.dtype"):
            columns[key] = value.split(",")
    if len(columns) != 4:
        raise StepFailure("generated operator config does not expose the expected dtype columns")
    width = len(columns["input0.dtype"])
    for index in range(width):
        observed = (
            columns["input0.dtype"][index], columns["input4.dtype"][index],
            columns["input8.dtype"][index], columns["input9.dtype"][index],
        )
        if observed == wanted:
            return index
    raise StepFailure(f"representative case {case_id} has no generated kernel dtype signature")


def _single_dtype_config(config_text: str, column: int) -> str:
    rendered: List[str] = []
    for line in config_text.splitlines():
        if "=" not in line:
            rendered.append(line)
            continue
        key, value = line.split("=", 1)
        values = value.split(",")
        if len(values) > 1 and (key.endswith(".dtype") or key.endswith(".format")):
            if column >= len(values):
                raise StepFailure(f"generated config column mismatch at {key}")
            value = values[column]
        rendered.append(f"{key}={value}")
    return "\n".join(rendered) + "\n"


def _representative_overlay(binary_root: Path, destination: Path) -> Path:
    if destination.exists():
        shutil.rmtree(destination)
    kernel_root = destination / "op_impl/ai_core/tbe/kernel"
    source_kernel = binary_root / "ascend910b/sparse_lightning_indexer_grad_kl_loss"
    source_config = binary_root / "config/ascend910b"
    if not source_kernel.is_dir() or not source_config.is_dir():
        raise StepFailure("representative CANN compilation did not produce a runtime kernel package")
    target_kernel = kernel_root / "ascend910b/sparse_lightning_indexer_grad_kl_loss"
    target_config = kernel_root / "config/ascend910b"
    target_kernel.mkdir(parents=True, exist_ok=True)
    target_config.mkdir(parents=True, exist_ok=True)
    for source in source_kernel.iterdir():
        if source.suffix in (".o", ".json"):
            shutil.copy2(source, target_kernel / source.name)
    for source in source_config.glob("*.json"):
        shutil.copy2(source, target_config / source.name)
    return destination


def _content_digest(paths: Sequence[Path], identity: str) -> str:
    digest = hashlib.sha256(identity.encode("utf-8"))
    for path in paths:
        digest.update(str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path.name).encode("utf-8"))
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _kernel_cache_identity(build_dir: Path, case_id: str, toolkit: Path,
                           compile_script: Path, quick_config: Path,
                           generated_config: Path, generated_json: Path,
                           option_files: Sequence[Path]) -> Dict[str, object]:
    case = get_case(case_id)
    tiling_library = build_dir / "op_host/libcustom_ascendc_cust_optiling.so"
    if not tiling_library.is_file():
        raise StepFailure(f"Kernel compile identity requires the Tiling library: {tiling_library}")
    source_mapping = build_dir / "autogen/ascendc_kernels_custom_source_files.ini"
    generated_inputs = [
        (generated_config, "autogen/aic-ascend910b-ops-info.ini"),
        (quick_config, "filtered/aic-ascend910b-ops-info.ini"),
        (generated_json, "op_info/aic-ascend910b-ops-info.json"),
    ]
    if source_mapping.is_file():
        generated_inputs.append((source_mapping, "autogen/ascendc_kernels_custom_source_files.ini"))
    generated_inputs.extend((path, f"options/{path.name}") for path in option_files)
    dtype = case.input_dtypes()
    return {
        "schema_version": 1,
        "operator": "SparseLightningIndexerGradKlLoss",
        "soc": {"compute_unit": "ascend910b"},
        "dtype_tuple": {
            "query": dtype["query"],
            "weights": dtype["weights"],
            "query_rope": dtype["queryRope"],
            "key_rope": dtype["keyRope"],
        },
        "kernel_inputs": identity_files((
            (CODE_DIR / "op_kernel/sparse_lightning_indexer_grad_kl_loss.cpp",
             "code/op_kernel/sparse_lightning_indexer_grad_kl_loss.cpp"),
        )),
        "host_tiling_abi": {
            "sources": identity_files((
                (CODE_DIR / "op_host/sparse_lightning_indexer_grad_kl_loss.cpp",
                 "code/op_host/sparse_lightning_indexer_grad_kl_loss.cpp"),
                (CODE_DIR / "op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h",
                 "code/op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h"),
                (CODE_DIR / "op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h",
                 "code/op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h"),
            )),
            "compiled_library": content_record(tiling_library, "libcustom_ascendc_cust_optiling.so"),
        },
        "generated_configuration": identity_files(generated_inputs),
        "compiler": _compiler_identity(toolkit, compile_script),
        "compile_options": {
            "compile_options": "",
            "debug_config": "",
            "enable_binary": True,
            "target_name": "ascendc_kernels",
            "build_tool": "make",
            "cmake": _cmake_generation_options(build_dir),
        },
    }


def build_representative_kernel(build_dir: Path, jobs: int, case_id: str,
                                prepare_host: bool = True,
                                cache_dir: Optional[Path] = None) -> Tuple[float, Optional[Path]]:
    """Compile or strictly restore one generated dtype signature."""

    started = time.monotonic()
    if prepare_host:
        ensure_host_tiling_for_kernel(build_dir, jobs)
    else:
        _require_schema_bootstrap(build_dir)
    generated_config = build_dir / "autogen/aic-ascend910b-ops-info.ini"
    generated_json = build_dir / "op_kernel/ascendc_kernels/tbe/op_info_cfg/ai_core/ascend910b/aic-ascend910b-ops-info.json"
    if not generated_json.is_file():
        _run_build(
            ("cmake", "--build", str(build_dir), "--target",
             "ascendc_kernels_ops_info_gen_ascend910b", "--parallel", str(jobs)),
            build_dir, "kernel-op-info", jobs,
        )
    config_text = generated_config.read_text(encoding="utf-8")
    column = _dtype_column(config_text, case_id)
    quick_root = build_dir / "tests/representative" / case_id
    quick_root.mkdir(parents=True, exist_ok=True)
    quick_config = quick_root / "aic-ascend910b-ops-info.ini"
    filtered_config = _single_dtype_config(config_text, column)
    if not quick_config.is_file() or quick_config.read_text(encoding="utf-8") != filtered_config:
        quick_config.write_text(filtered_config, encoding="utf-8")
    opc_options = quick_root / "custom_opc_options.ini"
    compile_options = quick_root / "custom_compile_options.ini"
    dtype_code = {"float16": 1, "bfloat16": 27, "float32": 0}
    case_dtype = get_case(case_id).input_dtypes()
    template_key = sum(dtype_code[case_dtype[name]] << (8 * index)
                       for index, name in enumerate(("query", "weights", "queryRope", "keyRope")))
    template_keys = [template_key]
    key_header = CODE_DIR / "op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h"
    if "ASCENDC_TPL_BOOL_DECL(VECTOR_DOT" in key_header.read_text():
        # A representative dtype includes both algorithms; old archived source
        # can still be rebuilt with its original dtype-only key contract.
        template_keys.append(template_key | (1 << 32))
    debug_option = "@--op_debug_config=dump_cce" if os.environ.get("SLI_KERNEL_DEBUG") == "1" else ""
    keys_option = ";".join(map(str, template_keys))
    opc_options.write_text(f"SparseLightningIndexerGradKlLoss@@--tiling_key={keys_option}{debug_option}\n")
    compile_options.touch()

    toolkit = _toolkit_root()
    compile_script = toolkit / "compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util/ascendc_compile_kernel.py"
    if not compile_script.is_file():
        compile_script = toolkit / "tools/op_project_templates/ascendc/common/util/ascendc_compile_kernel.py"
    binary_root = quick_root / "binary"
    overlay_root = quick_root / "opp/vendors/custom"
    completion = quick_root / ".complete"
    manifest_copy = quick_root / "kernel-cache-manifest.json"
    identity = _kernel_cache_identity(
        build_dir, case_id, toolkit, compile_script, quick_config,
        generated_config, generated_json, (opc_options, compile_options),
    )
    cache = KernelArtifactCache(cache_dir or DEFAULT_KERNEL_CACHE)
    key = cache.key_for(identity)
    with cache.lock(key):
        try:
            cached = cache.lookup(identity)
        except CacheValidationError as error:
            entry = cache.entry_path(key)
            rejected = cache.quarantine(entry, str(error))
            print(f"kernel cache REJECT key={key[:16]} quarantined={rejected}: {error}", flush=True)
            cached = None
        if cached is not None:
            entry, manifest = cached
            cache.materialize(entry, binary_root)
            overlay = _representative_overlay(binary_root, overlay_root)
            manifest_copy.write_text(
                json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
            completion.write_text(key + "\n", encoding="utf-8")
            (quick_root / ".kernel-inputs.sha256").unlink(missing_ok=True)
            elapsed = time.monotonic() - started
            print(
                f"kernel cache HIT key={key[:16]} case={case_id} "
                f"artifacts={len(manifest['artifacts'])} verified=sha256 elapsed={elapsed:.3f}s",
                flush=True,
            )
            return elapsed, overlay

        print(f"kernel cache MISS key={key[:16]} case={case_id}", flush=True)
        dynamic_dir = quick_root / "dynamic"
        for generated in (binary_root, dynamic_dir, overlay_root):
            if generated.is_dir():
                shutil.rmtree(generated)
            elif generated.exists():
                generated.unlink()
        completion.unlink(missing_ok=True)
        manifest_copy.unlink(missing_ok=True)
        dynamic_dir.mkdir(parents=True, exist_ok=True)
        command = (
            "python3", str(compile_script),
            "--op-type=SparseLightningIndexerGradKlLoss",
            f"--src-file={CODE_DIR / 'op_kernel/sparse_lightning_indexer_grad_kl_loss.cpp'}",
            "--compute-unit=ascend910b", "--compile-options=", "--debug-config=",
            f"--config-ini={quick_config}",
            f"--tiling-lib={build_dir / 'op_host/libcustom_ascendc_cust_optiling.so'}",
            f"--output-path={binary_root}", f"--dynamic-dir={dynamic_dir}",
            "--enable-binary=TRUE", f"--json-file={generated_json}",
            "--target-name=ascendc_kernels", f"--auto-gen-path={build_dir / 'autogen'}",
            "--build-tool=make",
        )
        _run_build(command, build_dir, f"kernel-{case_id}", jobs, cwd=quick_root)
        config_tool_root = toolkit / "compiler/tikcpp/ascendc_kernel_cmake/fwk_modules/util"
        if not config_tool_root.is_dir():
            config_tool_root = toolkit / "tools/op_project_templates/ascendc/common/util"
        soc_binary_root = binary_root / "ascend910b"
        _run(("python3", str(config_tool_root / "insert_simplified_keys.py"),
              "-p", str(soc_binary_root)), cwd=quick_root)
        _run(("python3", str(config_tool_root / "ascendc_ops_config.py"),
              "-p", str(soc_binary_root), "-s", "ascend910b"), cwd=quick_root)
        generated_runtime_config = binary_root / "config/ascend910b"
        generated_runtime_config.mkdir(parents=True, exist_ok=True)
        for config_file in soc_binary_root.glob("*.json"):
            shutil.move(str(config_file), generated_runtime_config / config_file.name)
        entry, manifest = cache.publish(
            identity, binary_root,
            {"git_commit": _git_commit(), "representative_case": case_id},
        )
        cache.materialize(entry, binary_root)
        overlay = _representative_overlay(binary_root, overlay_root)
        manifest_copy.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        completion.write_text(key + "\n", encoding="utf-8")
        (quick_root / ".kernel-inputs.sha256").unlink(missing_ok=True)
    elapsed = time.monotonic() - started
    print(
        f"kernel cache STORE key={key[:16]} case={case_id} "
        f"artifacts={len(manifest['artifacts'])} verified=sha256 elapsed={elapsed:.3f}s",
        flush=True,
    )
    return elapsed, overlay


def _runner_binary(build_dir: Path) -> Path:
    return build_dir / "tests/bin/sparse_lightning_indexer_grad_kl_loss_runner"


def build_host_only_opapi(build_dir: Path, jobs: int) -> Tuple[Path, float]:
    """Build the ACLNN entry without embedding a stale all-template archive."""

    toolkit = _toolkit_root()
    generated_source = build_dir / "autogen/aclnn_sparse_lightning_indexer_grad_kl_loss.cpp"
    host_object = (
        build_dir / "op_host/CMakeFiles/cust_optiling.dir"
        / "sparse_lightning_indexer_grad_kl_loss.cpp.o"
    )
    dependencies = (generated_source, host_object)
    missing = [str(path) for path in dependencies if not path.is_file()]
    if missing:
        raise StepFailure(
            "Host-only ACLNN library requires generated Schema and compiled Tiling objects. Missing: "
            + ", ".join(missing)
        )

    output_dir = build_dir / "tests/host_only_opapi"
    output_dir.mkdir(parents=True, exist_ok=True)
    generated_object = output_dir / "aclnn_sparse_lightning_indexer_grad_kl_loss.cpp.o"
    library = output_dir / "libcust_opapi.so"
    input_state = output_dir / ".inputs.sha256"
    dependency_digest = _content_digest(dependencies, "host-only-opapi-v1")
    if (library.is_file() and input_state.is_file()
            and input_state.read_text(encoding="utf-8").strip() == dependency_digest):
        print(f"reused Host-only ACLNN library: {library}", flush=True)
        return output_dir, 0.0

    generated_object.unlink(missing_ok=True)
    library.unlink(missing_ok=True)
    elapsed = _run_build(
        (
            "c++", "-std=c++17", "-D_GLIBCXX_USE_CXX11_ABI=0", "-DLOG_CPP",
            "-fPIC", "-fvisibility=hidden", "-fvisibility-inlines-hidden",
            "-fstack-protector-strong", "-I", str(toolkit / "include"),
            "-c", str(generated_source), "-o", str(generated_object),
        ),
        build_dir, "host-only-opapi-compile", jobs,
    )
    elapsed += _run_build(
        (
            "c++", "-fPIC", "-shared", "-Wl,-z,relro", "-Wl,-z,now",
            "-Wl,-z,noexecstack", "-Wl,-soname,libcust_opapi.so",
            "-o", str(library), str(host_object), str(generated_object),
            "-L", str(toolkit / "lib64"), "-lascendcl", "-lnnopbase",
            "-lexe_graph", "-lregister", "-ltiling_api",
        ),
        build_dir, "host-only-opapi-link", jobs,
    )
    input_state.write_text(dependency_digest + "\n", encoding="utf-8")
    return output_dir, elapsed


def build_runner(build_dir: Path, opapi_dir: Path) -> Tuple[Path, float]:
    toolkit = _toolkit_root()
    header = build_dir / "autogen/aclnn_sparse_lightning_indexer_grad_kl_loss.h"
    opapi = opapi_dir / "libcust_opapi.so"
    if not header.is_file() or not opapi.is_file():
        raise StepFailure("runner requires built ACLNN header and Host-only libcust_opapi.so")
    inputs = (RUNNER_SOURCE, header, opapi)
    array_api = "aclnnSparseLightningIndexerGradKlLossTensorGetWorkspaceSize" in header.read_text()
    dependency_digest = _content_digest(inputs, "aclnn-runner-v2:" + str(array_api))
    base_binary = _runner_binary(build_dir)
    binary = base_binary.parent / dependency_digest[:16] / base_binary.name
    input_state = binary.parent / ".runner-inputs.sha256"
    if (binary.is_file() and input_state.is_file()
            and input_state.read_text(encoding="utf-8").strip() == dependency_digest):
        print(f"reused runner: {binary}", flush=True)
        return binary, 0.0
    binary.parent.mkdir(parents=True, exist_ok=True)
    elapsed = _run(
        (
            "c++", "-std=c++17", "-O2", f"-DSLI_LENGTH_ARRAY_API={int(array_api)}", str(RUNNER_SOURCE),
            "-I", str(build_dir / "autogen"), "-I", str(toolkit / "include"),
            "-L", str(opapi_dir), "-L", str(toolkit / "lib64"),
            f"-Wl,-rpath,{opapi_dir}", f"-Wl,-rpath,{toolkit / 'lib64'}",
            "-lcust_opapi", "-lascendcl", "-lnnopbase", "-o", str(binary),
        )
    )
    input_state.write_text(dependency_digest + "\n", encoding="utf-8")
    return binary, elapsed


def selected_case_ids(suite: str, directed: Iterable[str], core_dtype: str = "",
                      weights_dtype: str = "", data_mode: str = "") -> Tuple[str, ...]:
    """Resolve a bounded named suite and optional fixture contract filters."""

    suites = {
        "custom": (),
        "dtype": DTYPE_CASES,
        "coverage": tuple(case.case_id for case in accepted_cases("coverage")),
        "correctness": tuple(case.case_id for case in accepted_cases("correctness")),
        "bounded": tuple(
            case.case_id for case in accepted_cases()
            if "platform_a2_a3" in case.tags and case.estimated_input_bytes() <= DEFAULT_MAX_BYTES
        ),
        "full": tuple(
            case.case_id for case in accepted_cases() if "platform_a2_a3" in case.tags
        ),
    }
    if suite not in suites:
        raise ValueError(f"unknown case suite: {suite}")
    identifiers = tuple(dict.fromkeys((*suites[suite], *directed)))
    if suite in ("coverage", "correctness"):
        outside = set(identifiers) - set(suites[suite])
        if outside:
            raise ValueError(f"Cases outside {suite} suite: {sorted(outside)}")
    selected = []
    for case_id in identifiers:
        case = get_case(case_id)
        if core_dtype and case.core_dtype != core_dtype:
            continue
        if weights_dtype and case.weights_dtype != weights_dtype:
            continue
        if data_mode and case_data_mode(case) != data_mode:
            continue
        selected.append(case_id)
    if not selected:
        raise StepFailure("case selection is empty after applying dtype/data-mode filters")
    return tuple(selected)


def materialize_fixtures(case_ids: Sequence[str], cache_dir: Path,
                         allow_large: bool) -> Tuple[List[Path], Dict[str, float]]:
    max_bytes = (1 << 63) - 1 if allow_large else DEFAULT_MAX_BYTES
    paths: List[Path] = []
    generated = 0
    started = time.monotonic()
    for case_id in case_ids:
        path, created = materialize_runner_fixture(get_case(case_id), cache_dir, max_bytes)
        paths.append(path)
        generated += int(created)
        print(f"{'generated' if created else 'reused'} fixture: {case_id}", flush=True)
    return paths, {
        "fixture_prepare": time.monotonic() - started,
        "fixture_generated": float(generated),
        "fixture_reused": float(len(paths) - generated),
    }


def _latest_runtime_log(pid: Optional[int] = None) -> Optional[Path]:
    candidates: List[Path] = []
    roots = []
    if os.environ.get("ASCEND_PROCESS_LOG_PATH"):
        roots.append(Path(os.environ["ASCEND_PROCESS_LOG_PATH"]))
    roots.append(Path.home() / "ascend/log/plog")
    roots.append(Path.home() / "ascend/log/run/plog")
    roots.append(Path.home() / "ascend/log/debug/plog")
    if pid is not None:
        candidates = [path for root in roots for path in root.glob(f"plog-{pid}_*.log")]
        return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None
    # ACL log roots can contain years of process logs. Inspect a bounded depth
    # and entry count; never recursively stream the complete tree after a crash.
    pending = [(root, 0) for root in roots if root.is_dir()]
    inspected = 0
    while pending and inspected < 512:
        directory, depth = pending.pop()
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    inspected += 1
                    if inspected > 512:
                        break
                    try:
                        if entry.is_file(follow_symlinks=False):
                            candidates.append(Path(entry.path))
                        elif depth < 3 and entry.is_dir(follow_symlinks=False):
                            pending.append((Path(entry.path), depth + 1))
                    except OSError:
                        continue
        except OSError:
            continue
    return max(candidates, key=lambda path: path.stat().st_mtime) if candidates else None


def collect_latest_runtime_log_once(results_dir: Path, pid: Optional[int] = None) -> Optional[Path]:
    latest = _latest_runtime_log(pid)
    if latest is None:
        print("no ACL runtime log found", flush=True)
        return None
    results_dir.mkdir(parents=True, exist_ok=True)
    destination = results_dir / "latest_runtime.log"
    shutil.copy2(latest, destination)
    print(f"collected latest ACL runtime log once: {latest} -> {destination}", flush=True)
    return destination


@contextlib.contextmanager
def serialized_npu():
    # Shared with the existing local operator projects, not the build cache.
    lock_path = Path("/tmp/sparse_lightning_indexer_grad_kl_loss/npu.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        resource_check("npu")
        yield


def run_npu(binary: Path, fixtures: Sequence[Path], build_dir: Path, opapi_dir: Path,
            overlay: Optional[Path], rtol: float, atol: float, device: int,
            results_dir: Path, workspace_only: bool = False,
            core_dtype: str = "", weights_dtype: str = "", data_mode: str = "",
            length_api: str = "", continue_comparison_errors: bool = False,
            capture_workspace_errors: bool = False) -> float:
    environment = dict(os.environ)
    library_paths = [str(opapi_dir), str(_toolkit_root() / "lib64")]
    if environment.get("LD_LIBRARY_PATH"):
        library_paths.append(environment["LD_LIBRARY_PATH"])
    environment["LD_LIBRARY_PATH"] = ":".join(library_paths)
    custom_opp = []
    if overlay is not None:
        if not overlay.is_dir():
            raise StepFailure(f"representative OPP is missing: {overlay}")
        custom_opp.append(str(overlay))
    built_opp = build_dir / "tmp/vendors/custom"
    if built_opp.is_dir():
        custom_opp.append(str(built_opp))
    if not custom_opp:
        raise StepFailure(f"no operator package for this build: {build_dir}")
    environment["ASCEND_CUSTOM_OPP_PATH"] = ":".join(custom_opp)
    operator_source = _git_commit()
    if (build_dir / "tests/final_build_state.json").is_file() and opapi_dir == build_dir:
        from tests.st.scripts.archive_build import verify_recorded_build
        operator_source = verify_recorded_build(build_dir)
    command = [str(binary), "--device", str(device)]
    if length_api:
        command.extend(("--length-api", length_api))
    if continue_comparison_errors:
        command.append("--continue-comparison-errors")
    if capture_workspace_errors:
        command.append("--capture-workspace-errors")
    if workspace_only:
        command.append("--workspace-only")
    else:
        command.extend(("--rtol", str(rtol), "--atol", str(atol)))
    for option, value in (
        ("--core-dtype", core_dtype),
        ("--weights-dtype", weights_dtype),
        ("--data-mode", data_mode),
    ):
        if value:
            command.extend((option, value))
    for path in fixtures:
        command.extend(("--fixture", str(path)))
    with serialized_npu():
        results_dir.mkdir(parents=True, exist_ok=True)
        run_dir = results_dir / (time.strftime("run-%Y%m%d-%H%M%S") + f"-{os.getpid()}")
        run_dir.mkdir()
        environment["SLI_EXPECTED_OPAPI"] = str((opapi_dir / "libcust_opapi.so").resolve())
        artifacts = [binary, opapi_dir / "libcust_opapi.so",
                     build_dir / "op_host/libcustom_ascendc_cust_optiling.so"]
        for root in custom_opp:
            artifacts.extend(p for p in Path(root).rglob("*")
                             if p.is_file() and p.suffix in (".o", ".json", ".so"))
        manifest = {"command": command, "source": operator_source,
                    "opp_path": custom_opp, "rtol": rtol, "atol": atol,
                    "fixtures": [content_record(p / name, str(p / name)) for p in fixtures
                                 for name in ("case.conf", "case.json", ".complete")],
                    "artifacts": [content_record(p, str(p)) for p in sorted(set(artifacts))]}
        manifest_path = run_dir / "manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
        print(f"NPU evidence: {run_dir}", flush=True)
        started = time.monotonic()
        try:
            with (run_dir / "runner.log").open("w") as log:
                process = subprocess.Popen(command, env=environment, cwd=ROOT,
                                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                manifest["runner_pid"] = process.pid
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end="", flush=True)
                status = process.wait()
            manifest["returncode"] = status
            manifest["elapsed_seconds"] = time.monotonic() - started
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            if status:
                raise StepFailure(f"NPU validation failed; evidence: {run_dir}", status)
            return manifest["elapsed_seconds"]
        except StepFailure as error:
            if error.returncode != 1:
                collect_latest_runtime_log_once(run_dir, process.pid)
            raise


def incremental_build_dir(requested: Optional[Path]) -> Path:
    return requested or Path(os.environ.get("SLI_INCREMENTAL_BUILD", DEFAULT_INCREMENTAL_BUILD))


def kernel_cache_dir(requested: Optional[Path]) -> Path:
    return requested or Path(os.environ.get("SLI_KERNEL_CACHE", DEFAULT_KERNEL_CACHE))


def clean_final_build_dir(requested: Optional[Path]) -> Path:
    path = requested or Path(os.environ.get("SLI_FINAL_BUILD", DEFAULT_FINAL_BUILD))
    if path.exists() and any(path.iterdir()):
        raise StepFailure(
            f"final build directory is not clean: {path}; move or remove it explicitly before final validation"
        )
    path.mkdir(parents=True, exist_ok=True)
    return path


def existing_final_build_dir(requested: Optional[Path]) -> Path:
    path = requested or Path(os.environ.get("SLI_FINAL_BUILD", DEFAULT_FINAL_BUILD))
    required = (path / "CMakeCache.txt", path / "libcust_opapi.so")
    if not all(item.is_file() for item in required):
        raise StepFailure(
            f"final build is missing or incomplete: {path}; run the full-build layer first"
        )
    receipt = path / "tests/final_build_state.json"
    if not receipt.is_file() or json.loads(receipt.read_text()) != _final_build_identity(path):
        raise StepFailure("final build source/toolchain/artifact identity does not match")
    return path


def _existing_representative_overlay(build_dir: Path, case_id: str) -> Path:
    root = build_dir / "tests/representative" / case_id
    overlay = root / "opp/vendors/custom"
    current = False
    try:
        identity = json.loads((root / "kernel-cache-manifest.json").read_text())["identity"]
        kernel_name = "op_kernel/sparse_lightning_indexer_grad_kl_loss.cpp"
        host_names = (
            "op_host/sparse_lightning_indexer_grad_kl_loss.cpp",
            "op_kernel/sparse_lightning_indexer_grad_kl_loss_tiling.h",
            "op_kernel/tiling_key_sparse_lightning_indexer_grad_kl_loss.h",
        )
        current = (
            identity["kernel_inputs"] == identity_files(((CODE_DIR / kernel_name, "code/" + kernel_name),))
            and identity["host_tiling_abi"]["sources"] == identity_files(
                (CODE_DIR / name, "code/" + name) for name in host_names)
            and identity["host_tiling_abi"]["compiled_library"] == content_record(
                build_dir / "op_host/libcustom_ascendc_cust_optiling.so",
                "libcustom_ascendc_cust_optiling.so")
        )
    except (OSError, ValueError, KeyError, TypeError):
        pass
    if not current or not (root / ".complete").is_file() or not overlay.is_dir():
        raise StepFailure(
            "representative Kernel is not current; run "
            f"`python3 -m tests.st.scripts.run_tests --layer kernel --representative-case {case_id}` first"
        )
    return overlay


def _default_suite(layer: str) -> str:
    return {
        "workspace": "dtype",
        "smoke": "custom",
        "correctness": "correctness",
        "full": "full",
    }.get(layer, "custom")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layer", choices=LAYERS, default="pipeline",
        help="independent layer, or fail-fast pipeline (default: pipeline)",
    )
    parser.add_argument(
        "--stage", choices=("quick", "final"), default="quick",
        help="pipeline depth; final adds one clean all-template build and full NPU matrix",
    )
    parser.add_argument("--host-component", choices=HOST_COMPONENTS, default="all")
    parser.add_argument("--jobs", type=int, help="build parallelism; default env override or 28")
    parser.add_argument("--build-dir", type=Path, help="single reusable incremental build directory")
    parser.add_argument("--final-build-dir", type=Path, help="absent or empty directory used only by final layers")
    parser.add_argument(
        "--kernel-cache", type=Path,
        help="strict content-addressed Kernel cache; env SLI_KERNEL_CACHE or work root",
    )
    parser.add_argument("--suite", choices=("auto", "custom", "dtype", "coverage", "correctness", "bounded", "full"),
                        default="auto", help="named parameterized case suite")
    parser.add_argument("--case", action="append", default=[], help="additional case id; repeatable")
    parser.add_argument("--representative-case", default="baseline_f16", help="dtype used for quick kernel compilation")
    parser.add_argument("--core-dtype", choices=("float16", "bfloat16"), default="")
    parser.add_argument("--weights-dtype", choices=("float16", "bfloat16", "float32"), default="")
    parser.add_argument("--data-mode", choices=("empty", "zero", "nonzero"), default="")
    parser.add_argument("--fixture-cache", type=Path, default=DEFAULT_FIXTURE_CACHE)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rtol", type=float, help="configured NPU relative tolerance")
    parser.add_argument("--atol", type=float, help="configured NPU absolute tolerance")
    parser.add_argument("--allow-large", action="store_true", help="explicitly allow stress fixture materialization")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs = resolve_jobs(args.jobs, os.environ)
    build_dir = incremental_build_dir(args.build_dir)
    kernel_cache = kernel_cache_dir(args.kernel_cache)
    timings: Dict[str, float] = {}

    def run_static_layer() -> None:
        timings["static"] = static_checks()
        timings.update(python_checks())

    def execute_cases(layer: str, active_build: Path, overlay: Optional[Path],
                      suite_override: str = "", directed: Optional[Sequence[str]] = None,
                      use_filters: bool = True) -> None:
        suite = suite_override or (_default_suite(layer) if args.suite == "auto" else args.suite)
        selected_directed = args.case if directed is None else directed
        identifiers = selected_case_ids(
            suite, selected_directed,
            args.core_dtype if use_filters else "",
            args.weights_dtype if use_filters else "",
            args.data_mode if use_filters else "",
        )
        fixtures, fixture_timings = materialize_fixtures(
            identifiers, args.fixture_cache, args.allow_large
        )
        timings.update(fixture_timings)
        if layer == "full":
            opapi_dir, opapi_elapsed = active_build, 0.0
        else:
            opapi_dir, opapi_elapsed = build_host_only_opapi(active_build, jobs)
        timings["host_only_opapi"] = timings.get("host_only_opapi", 0.0) + opapi_elapsed
        runner, runner_elapsed = build_runner(active_build, opapi_dir)
        timings["runner_build"] = timings.get("runner_build", 0.0) + runner_elapsed
        workspace_only = layer == "workspace"
        runtime_opapi_dir = opapi_dir
        if not workspace_only and (args.rtol is None or args.atol is None):
            raise StepFailure(f"{layer} NPU validation requires explicit --rtol and --atol")
        timings[layer] = run_npu(
            runner, fixtures, active_build, runtime_opapi_dir, overlay,
            args.rtol or 0.0, args.atol or 0.0,
            args.device, args.results_dir, workspace_only,
            args.core_dtype if use_filters else "",
            args.weights_dtype if use_filters else "",
            args.data_mode if use_filters else "",
        )
        print(f"{layer}: PASS cases={len(fixtures)} suite={suite}", flush=True)

    if args.layer == "static":
        run_static_layer()
    elif args.layer == "host":
        timings["host"] = build_host(build_dir, jobs, args.host_component)
    elif args.layer == "workspace":
        overlay = _existing_representative_overlay(build_dir, args.representative_case)
        execute_cases("workspace", build_dir, overlay)
    elif args.layer == "kernel":
        timings["kernel"], _ = build_representative_kernel(
            build_dir, jobs, args.representative_case, cache_dir=kernel_cache
        )
    elif args.layer in ("smoke", "correctness"):
        timings["kernel"], overlay = build_representative_kernel(
            build_dir, jobs, args.representative_case, cache_dir=kernel_cache
        )
        execute_cases(
            args.layer, build_dir, overlay,
        )
    elif args.layer == "full-build":
        final_dir = clean_final_build_dir(args.final_build_dir)
        timings["full_build"] = build_full(final_dir, jobs)
    elif args.layer == "full-test":
        final_dir = existing_final_build_dir(args.final_build_dir)
        suite = _default_suite("full") if args.suite == "auto" else args.suite
        if suite == "full" and not args.allow_large:
            raise StepFailure("full NPU matrix requires explicit --allow-large")
        execute_cases("full", final_dir, None)
    elif args.layer == "full":
        suite = _default_suite("full") if args.suite == "auto" else args.suite
        if suite == "full" and not args.allow_large:
            raise StepFailure("full NPU matrix requires explicit --allow-large")
        final_dir = clean_final_build_dir(args.final_build_dir)
        timings["full_build"] = build_full(final_dir, jobs)
        execute_cases("full", final_dir, None)
    else:
        # Deliberately sequential: every layer is a gate and device work never
        # starts after a failed static, contract, Host, or Workspace check.
        if args.suite != "auto" or args.core_dtype or args.weights_dtype or args.data_mode:
            raise StepFailure(
                "pipeline uses its fixed fail-fast suites; dtype/data filters are for independent layers"
            )
        run_static_layer()
        timings["host"] = build_host(build_dir, jobs, args.host_component)
        # The static Host contract test queries Tiling/workspace before Kernel
        # compilation. ACLNN additionally needs generated binary registration.
        timings["kernel"], overlay = build_representative_kernel(
            build_dir, jobs, args.representative_case, prepare_host=False,
            cache_dir=kernel_cache,
        )
        execute_cases("workspace", build_dir, overlay, "custom",
                      ("query_empty_special_case", "baseline_f16"), False)
        execute_cases("smoke", build_dir, overlay, "custom",
                      ("query_empty_special_case", "baseline_f16"), False)
        execute_cases("correctness", build_dir, overlay, "correctness", args.case, False)
        if args.stage == "final":
            if not args.allow_large:
                raise StepFailure("final pipeline requires explicit --allow-large")
            final_dir = clean_final_build_dir(args.final_build_dir)
            timings["full_build"] = build_full(final_dir, jobs)
            execute_cases("full", final_dir, None, "full", (), False)

    print(
        f"PASS layer={args.layer} stage={args.stage} jobs={jobs} "
        f"incremental_build={build_dir} timings={timings}",
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("ERROR: validation interrupted; the runner is not retried", file=sys.stderr)
        raise SystemExit(130)
    except (StepFailure, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        raise SystemExit(error.returncode if isinstance(error, StepFailure) and error.returncode else 2)
