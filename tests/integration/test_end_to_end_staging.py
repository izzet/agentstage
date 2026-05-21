"""Layer-3 end-to-end integration test.

The whole stager stack on a synthetic 5-file workload, without an LLM.
This is the closest analog of E5 we can run pre-Day-5:

  - Stager runs in-process, gets a DataHint, starts copying cold -> hot
  - A subprocess opens the cold paths via LD_PRELOAD shim
  - The shim redirects opens that have hot copies; falls through otherwise
  - We assert: detected files were served from hot; undetected from cold

If this passes, the stager + shim contract is fully wired and we can
trust the Day-7 manual smoke (T32 with real LLM) to work or fail for
reasons unrelated to the stager itself.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

import pytest

from agentstage.stager import DataHint, Stager


SHIM_SO = (
    Path(__file__).parent.parent.parent
    / "src" / "agentstage" / "stager" / "shim" / "libagentstage_shim.so"
)


@pytest.fixture(scope="module")
def shim_lib() -> Path:
    """Ensure the shim is built before integration tests run."""
    shim_dir = SHIM_SO.parent
    src = shim_dir / "agentstage_shim.c"
    if not SHIM_SO.exists() or SHIM_SO.stat().st_mtime < src.stat().st_mtime:
        subprocess.run(["make", "-C", str(shim_dir)], check=True, capture_output=True)
    return SHIM_SO


def _make_synthetic_workload(cold_dir: Path) -> list[tuple[Path, bytes]]:
    """Create 5 files with deterministic content + known sizes."""
    specs = [
        ("file1_1MB.bin", 1024 * 1024),
        ("file2_5MB.bin", 5 * 1024 * 1024),
        ("file3_10MB.bin", 10 * 1024 * 1024),
        ("file4_25MB.bin", 25 * 1024 * 1024),
        ("file5_50MB.bin", 50 * 1024 * 1024),
    ]
    files = []
    for name, size in specs:
        path = cold_dir / name
        # Reproducible pseudo-random content keyed by name
        seed = int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
        rng = bytes((seed + i) & 0xFF for i in range(size))
        path.write_bytes(rng)
        files.append((path, rng))
    return files


def _agent_subprocess_script(targets: list[Path], summary_path: Path) -> str:
    """The "agent" script: opens each target, records read latency + hash."""
    return f"""
import hashlib, json, sys, time
from pathlib import Path

results = []
for target in {[str(t) for t in targets]!r}:
    t0 = time.monotonic_ns()
    with open(target, 'rb') as f:
        data = f.read()
    elapsed_ms = (time.monotonic_ns() - t0) / 1e6
    h = hashlib.sha256(data).hexdigest()
    results.append({{
        "target": target,
        "elapsed_ms": elapsed_ms,
        "size_bytes": len(data),
        "sha256": h,
    }})

Path({str(summary_path)!r}).write_text(json.dumps(results))
"""


@pytest.mark.parametrize("with_dftracer", [False, True], ids=["no_dftracer", "with_dftracer"])
def test_end_to_end_synthetic_5_file_workload(shim_lib, tmp_path, with_dftracer):
    """Five-file mini-workload, parametrized to run with and without DFTracer
    in the LD_PRELOAD chain:

      - Stager pre-stages files 1, 2, 3 (NOT 4, 5)
      - Subprocess opens all 5 via LD_PRELOAD shim
      - Files 1-3 must come from hot (fast); 4-5 from cold
      - Byte identity preserved on all 5

    With dftracer in the chain, additionally verifies the integration
    still works (no crashes, redirect still functions). DFTracer's own
    trace correctness is tested in tests/test_dftracer_chain.py.
    """
    if with_dftracer:
        from tests.test_dftracer_chain import _find_dftracer_preload
        dftracer = _find_dftracer_preload()
        if dftracer is None:
            pytest.skip(
                "DFTracer .so not found; set AGENTSTAGE_DFTRACER_PRELOAD "
                "or build external/libs/dftracer/"
            )
    else:
        dftracer = None

    cold_dir = tmp_path / "cold"
    hot_dir = tmp_path / "hot"
    cold_dir.mkdir()
    hot_dir.mkdir()

    files = _make_synthetic_workload(cold_dir)
    cold_paths = [p for p, _ in files]
    expected_hashes = {str(p): hashlib.sha256(b).hexdigest() for p, b in files}

    # 1. Stager pre-stages the first 3 files
    stager = Stager(
        hot_root=hot_dir,
        cold_roots=[cold_dir],
        max_workers=3,
        capacity_bytes=256 * 1024 * 1024,
    )
    hint = DataHint(
        detected_files=tuple(str(p) for p in cold_paths[:3]),
        tier=1,
        fired_at_ms=0.0,
        rule_id="integration_test",
    )
    futures = stager.prefetch(hint)
    for f in futures:
        f.result(timeout=30)

    # Sanity: 3 hot files should exist now, 2 should not
    for p in cold_paths[:3]:
        assert stager.is_staged(p), f"{p} should be staged"
    for p in cold_paths[3:]:
        assert not stager.is_staged(p), f"{p} should NOT be staged"

    # 2. Subprocess opens all 5 cold paths via shim
    summary_path = tmp_path / "agent_results.json"
    script = _agent_subprocess_script(cold_paths, summary_path)

    env = os.environ.copy()
    # LD_PRELOAD chain: dftracer first if present (logs intent),
    # then our shim (redirects).
    if dftracer is not None:
        env["LD_PRELOAD"] = f"{dftracer}:{shim_lib}"
        # Minimal dftracer config so it loads without crashing
        env["DFTRACER_ENABLE"] = "1"
        env["DFTRACER_LOG_FILE"] = str(tmp_path / "dftracer_trace")
        env["DFTRACER_DATA_DIR"] = str(cold_dir)
        env["DFTRACER_DISABLE_IO"] = "0"
        env["DFTRACER_INIT"] = "PRELOAD"
    else:
        env["LD_PRELOAD"] = str(shim_lib)
    env["AGENTSTAGE_HOT_ROOT"] = str(hot_dir)
    env["AGENTSTAGE_COLD_ROOTS"] = str(cold_dir)
    env["AGENTSTAGE_RETRY_SPIN_MS"] = "5"
    env["AGENTSTAGE_SHIM_LOG"] = str(tmp_path / "shim.log")

    r = subprocess.run(
        ["python3", "-c", script],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, f"agent subprocess failed:\n{r.stderr}"

    # 3. Inspect the agent's read results
    agent_results = json.loads(summary_path.read_text())
    by_target = {r["target"]: r for r in agent_results}

    # Byte-identity: every read must match the cold source
    for target, expected_hash in expected_hashes.items():
        got = by_target[target]["sha256"]
        assert got == expected_hash, (
            f"read hash differs from source for {target}: "
            f"expected {expected_hash}, got {got}"
        )

    # 4. Inspect the shim log: hits for files 1-3, misses for 4-5
    shim_log = (tmp_path / "shim.log").read_text()
    print(f"\nshim log:\n{shim_log}")

    for p in cold_paths[:3]:
        # Each staged file should have at least one HIT entry
        line = f"HIT  {p} -> "
        assert line in shim_log, (
            f"expected HIT for {p} but log shows:\n{shim_log}"
        )

    for p in cold_paths[3:]:
        # Unstaged files should show MISS
        line = f"MISS {p}"
        assert line in shim_log, (
            f"expected MISS for {p} but log shows:\n{shim_log}"
        )

    # 5. Check the staging report from the Stager side
    summary = stager.report.summary()
    print(f"\nstager report: {summary}")
    assert summary["n_staged"] == 3
    assert summary["total_bytes_staged"] == (1 + 5 + 10) * 1024 * 1024

    stager.shutdown(wait=True)


def test_end_to_end_latency_signal(shim_lib, tmp_path):
    """Sanity-check that staged files read faster than cold files in
    the same subprocess. This is the synthetic version of E5's headline
    measurement — much smaller in scale, but the directional signal must
    be correct: staged file read < cold file read."""
    cold_dir = tmp_path / "cold"
    hot_dir = tmp_path / "hot"
    cold_dir.mkdir()
    hot_dir.mkdir()

    # Two files: one staged, one not. Both large enough that I/O dominates.
    staged_file = cold_dir / "staged.bin"
    cold_file = cold_dir / "cold.bin"
    payload = b"x" * (10 * 1024 * 1024)
    staged_file.write_bytes(payload)
    cold_file.write_bytes(payload)

    stager = Stager(
        hot_root=hot_dir,
        cold_roots=[cold_dir],
        max_workers=1,
    )
    futures = stager.prefetch(DataHint(
        detected_files=(str(staged_file),),
        tier=1,
        fired_at_ms=0.0,
        rule_id="latency_test",
    ))
    for f in futures:
        f.result(timeout=30)

    # Drop page cache on the cold file so it's truly cold
    try:
        from agentiobench.utils.cache import _resident_pages
        fd = os.open(str(cold_file), os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
    except ImportError:
        pass

    # Subprocess opens both, measures elapsed
    summary_path = tmp_path / "results.json"
    script = _agent_subprocess_script(
        [staged_file, cold_file], summary_path,
    )

    env = os.environ.copy()
    env["LD_PRELOAD"] = str(shim_lib)
    env["AGENTSTAGE_HOT_ROOT"] = str(hot_dir)
    env["AGENTSTAGE_COLD_ROOTS"] = str(cold_dir)
    env["AGENTSTAGE_RETRY_SPIN_MS"] = "5"

    r = subprocess.run(
        ["python3", "-c", script],
        env=env, capture_output=True, text=True, timeout=60,
    )
    assert r.returncode == 0, r.stderr

    results = {r["target"]: r for r in json.loads(summary_path.read_text())}
    staged_ms = results[str(staged_file)]["elapsed_ms"]
    cold_ms = results[str(cold_file)]["elapsed_ms"]

    print(f"\nstaged: {staged_ms:.3f} ms, cold: {cold_ms:.3f} ms")

    # Direction: staged must be faster than cold (both 10 MB)
    # On tmpfs vs warm XFS the gap can be small, so just require <
    assert staged_ms < cold_ms, (
        f"expected staged read to be faster than cold; "
        f"got staged={staged_ms:.3f}ms, cold={cold_ms:.3f}ms"
    )

    stager.shutdown(wait=True)
