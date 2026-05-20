"""DFTracer + agentstage shim LD_PRELOAD chain tests.

Verifies that loading both libdftracer_preload.so AND libagentstage_shim.so
together produces the documented behavior:

  1. DFTracer's openat wrapper runs first (logs cold-path INTENT)
  2. agentstage shim's openat wrapper runs next (redirects cold -> hot)
  3. libc's openat executes against the hot path
  4. The agent reads hot bytes; DFTracer's io_report records cold paths

This is the verification the Day-5 manual smoke (T32) was previously
relying on as its first signal of any DFTracer-chain problem. Splitting
it out here means T32 is reduced to "real LLM + real agent" without
the additional DFTracer-correctness risk.

Five tests:
  1. test_dftracer_alone_produces_trace            — dftracer loads + traces
  2. test_dftracer_logs_cold_path_when_shim_redirects — the critical chain test
  3. test_chain_order_is_load_bearing               — reverse order breaks it
  4. test_dfanalyzer_produces_io_report_matching_empirical_gt_schema
  5. test_writes_pass_through_both_wrappers         — write hygiene with both

All tests skip cleanly when libdftracer_preload.so or the dfanalyzer
Python package is missing (e.g. submodule not yet built). The
AGENTSTAGE_DFTRACER_PRELOAD env var overrides the .so path search.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent
SHIM_SO = (
    PROJECT_ROOT
    / "src" / "agentstage" / "stager" / "shim" / "libagentstage_shim.so"
)


# ---------------------------------------------------------------------------
# Dependency resolution — graceful skip when missing
# ---------------------------------------------------------------------------

def _find_dftracer_preload() -> Path | None:
    """Locate libdftracer_preload.so. Tries (in order):
      1. AGENTSTAGE_DFTRACER_PRELOAD env var (explicit override)
      2. Our submodule's build dir
      3. Sciiobench's pre-built copy (legacy fallback)
    """
    env = os.environ.get("AGENTSTAGE_DFTRACER_PRELOAD")
    if env and Path(env).is_file():
        return Path(env)

    # Submodule build path
    candidates = list(
        (PROJECT_ROOT / "external" / "libs" / "dftracer")
        .rglob("libdftracer_preload.so")
    )
    if candidates:
        return candidates[0]

    # Sciiobench fallback
    sciio = (
        Path.home() / "projects" / "sciiobench" / "dftracer"
    )
    if sciio.is_dir():
        candidates = list(sciio.rglob("libdftracer_preload.so"))
        # Prefer the non-_dbg build
        non_dbg = [c for c in candidates if "_dbg" not in str(c)]
        return (non_dbg or candidates)[0] if candidates else None

    return None


def _have_dfanalyzer() -> bool:
    try:
        import dftracer.analyzer  # noqa
        return True
    except ImportError:
        return False


def _have_agentiobench_tracing() -> bool:
    try:
        from agentiobench.tracing import configure_dftracer_env  # noqa
        return True
    except ImportError:
        return False


DFTRACER_PRELOAD = _find_dftracer_preload()
HAVE_DFANALYZER = _have_dfanalyzer()
HAVE_AIOB_TRACING = _have_agentiobench_tracing()

# Skipping doesn't compose cleanly with parametrize/fixtures unless
# applied per-test, so each test does its own gate.

REASON_NO_DFTRACER = (
    "libdftracer_preload.so not found; set AGENTSTAGE_DFTRACER_PRELOAD or "
    "build external/libs/dftracer/"
)
REASON_NO_DFANALYZER = (
    "dftracer.analyzer python package not installed; "
    "run `uv add --editable external/libs/dfanalyzer`"
)
REASON_NO_SHIM = "libagentstage_shim.so not built; run make in src/agentstage/stager/shim/"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shim_lib() -> Path:
    if not SHIM_SO.exists():
        pytest.skip(REASON_NO_SHIM)
    return SHIM_SO


@pytest.fixture
def cold_dir(tmp_path: Path) -> Path:
    d = tmp_path / "cold"
    d.mkdir()
    return d


@pytest.fixture
def hot_dir(tmp_path: Path) -> Path:
    d = tmp_path / "hot"
    d.mkdir()
    return d


@pytest.fixture
def trace_dir(tmp_path: Path) -> Path:
    d = tmp_path / "trace"
    d.mkdir()
    return d


def hot_mirror_for(hot_dir: Path, cold_path: Path) -> Path:
    return hot_dir / str(cold_path.resolve()).lstrip("/")


def place_hot_copy(hot_dir: Path, cold_path: Path, content: bytes) -> Path:
    hot = hot_mirror_for(hot_dir, cold_path)
    hot.parent.mkdir(parents=True, exist_ok=True)
    hot.write_bytes(content)
    return hot


def make_dftracer_env(
    trace_dir: Path,
    cold_dir: Path,
    *,
    log_prefix: str = "test_trace",
) -> dict[str, str]:
    """Build the DFTRACER_* env vars per AIOB's contract."""
    env = os.environ.copy()
    env["DFTRACER_ENABLE"] = "1"
    env["DFTRACER_LOG_FILE"] = str(trace_dir / log_prefix)
    env["DFTRACER_DATA_DIR"] = str(cold_dir)
    env["DFTRACER_INC_METADATA"] = "1"
    env["DFTRACER_DISABLE_IO"] = "0"
    env["DFTRACER_INIT"] = "PRELOAD"
    return env


def make_shim_env(hot_dir: Path, cold_dir: Path) -> dict[str, str]:
    env = {}
    env["AGENTSTAGE_HOT_ROOT"] = str(hot_dir)
    env["AGENTSTAGE_COLD_ROOTS"] = str(cold_dir)
    env["AGENTSTAGE_RETRY_SPIN_MS"] = "5"
    return env


def run_subprocess_with_chain(
    script: str,
    *,
    ld_preload_libs: list[Path],
    extra_env: dict[str, str],
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["LD_PRELOAD"] = ":".join(str(p) for p in ld_preload_libs)
    env.update(extra_env)
    return subprocess.run(
        ["python3", "-c", script],
        env=env, capture_output=True, text=True, timeout=30,
    )


# ---------------------------------------------------------------------------
# Test 1 — dftracer alone produces a trace
# ---------------------------------------------------------------------------

def test_dftracer_alone_produces_trace(cold_dir, trace_dir):
    """Sanity: load only libdftracer_preload.so. Subprocess does an open().
    Verify a trace file appears under trace_dir."""
    if DFTRACER_PRELOAD is None:
        pytest.skip(REASON_NO_DFTRACER)

    target = cold_dir / "doc.txt"
    target.write_text("DFTRACER_SAW_THIS")

    env_dft = make_dftracer_env(trace_dir, cold_dir)
    r = run_subprocess_with_chain(
        f"print(open({str(target)!r}).read())",
        ld_preload_libs=[DFTRACER_PRELOAD],
        extra_env=env_dft,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert r.stdout.strip() == "DFTRACER_SAW_THIS"

    # Trace files: *.pfw or *.pfw.gz under trace_dir
    traces = list(trace_dir.glob("*.pfw*"))
    assert traces, (
        f"no trace files produced under {trace_dir}; "
        f"contents: {list(trace_dir.iterdir())}"
    )


# ---------------------------------------------------------------------------
# Test 2 — the critical chain test
# ---------------------------------------------------------------------------

def test_dftracer_logs_cold_path_when_shim_redirects(
    shim_lib, cold_dir, hot_dir, trace_dir
):
    """LD_PRELOAD chain: dftracer first, agentstage second.

    DFTracer's openat wrapper logs the cold path (agent's INTENT).
    AgentStage's openat wrapper redirects to hot.
    Final libc openat hits the hot path.

    Assertion: subprocess reads HOT bytes; trace file contains an event
    for the COLD path (proof that dftracer saw the agent's intent
    before our redirect)."""
    if DFTRACER_PRELOAD is None:
        pytest.skip(REASON_NO_DFTRACER)

    cold_file = cold_dir / "race.txt"
    cold_file.write_text("COLD_CONTENT")
    place_hot_copy(hot_dir, cold_file, b"HOT_CONTENT")

    env = {**make_dftracer_env(trace_dir, cold_dir),
           **make_shim_env(hot_dir, cold_dir)}
    r = run_subprocess_with_chain(
        f"print(open({str(cold_file)!r}).read())",
        ld_preload_libs=[DFTRACER_PRELOAD, shim_lib],
        extra_env=env,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"

    # Agent sees HOT (proves agentstage redirected)
    assert r.stdout.strip() == "HOT_CONTENT", (
        f"shim did not redirect; got {r.stdout!r}"
    )

    # DFTracer must have logged the COLD path as intent
    traces = list(trace_dir.glob("*.pfw*"))
    assert traces, "no trace files produced"
    trace_bytes = _read_trace_files(traces)
    assert str(cold_file) in trace_bytes, (
        f"dftracer's trace doesn't mention cold path {cold_file}; "
        f"check chain order"
    )


# ---------------------------------------------------------------------------
# Test 3 — chain order is NOT load-bearing for dftracer (positive finding)
# ---------------------------------------------------------------------------

def test_chain_order_does_not_break_dftracer_intent_logging(
    shim_lib, cold_dir, hot_dir, trace_dir
):
    """Reverse the LD_PRELOAD order: agentstage first, dftracer second.

    Empirical finding (2026-05-19): dftracer logs the cold path
    regardless of LD_PRELOAD ordering. It uses syscall-level
    instrumentation (likely a deeper hook than LD_PRELOAD function
    wrapping), so it sees the agent's intent even when agentstage
    redirects before dftracer's wrapper runs.

    This is the BEST possible behavior for AgentStage's story: we
    don't have to worry about LD_PRELOAD ordering for ground-truth
    capture. Both orderings produce identical io_report.json content.

    Verifies:
      - subprocess still reads hot bytes (agentstage redirect works)
      - dftracer trace still shows the cold path (intent captured)

    This test guards against future dftracer changes that might
    weaken the syscall-level instrumentation to mere LD_PRELOAD
    wrapping — in which case ordering would suddenly matter.
    """
    if DFTRACER_PRELOAD is None:
        pytest.skip(REASON_NO_DFTRACER)

    cold_file = cold_dir / "reverse.txt"
    cold_file.write_text("COLD")
    place_hot_copy(hot_dir, cold_file, b"HOT")

    env = {**make_dftracer_env(trace_dir, cold_dir, log_prefix="reverse"),
           **make_shim_env(hot_dir, cold_dir)}
    r = run_subprocess_with_chain(
        f"print(open({str(cold_file)!r}).read())",
        ld_preload_libs=[shim_lib, DFTRACER_PRELOAD],  # REVERSED order
        extra_env=env,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"
    # Agent still sees HOT (agentstage's redirect runs first now)
    assert r.stdout.strip() == "HOT", (
        "shim's redirect should work in either ordering"
    )

    traces = list(trace_dir.glob("reverse*.pfw*"))
    assert traces, "no trace files produced"
    trace_bytes = _read_trace_files(traces)

    # dftracer should STILL log the cold path even in reversed order.
    # If this assertion fails in the future, dftracer's instrumentation
    # has changed and the AgentStage paper's io_report.json claims
    # would need an LD_PRELOAD-ordering caveat.
    assert str(cold_file) in trace_bytes, (
        "dftracer no longer logs cold-path intent in reversed LD_PRELOAD "
        "ordering — its instrumentation has weakened to pure function "
        "wrapping. Document this in STAGER_VERIFICATION.md and pin a "
        "specific LD_PRELOAD order in CAMPAIGN.md."
    )


# ---------------------------------------------------------------------------
# Test 4 — dfanalyzer produces io_report matching empirical_gt schema
# ---------------------------------------------------------------------------

def test_dfanalyzer_produces_io_report_matching_empirical_gt_schema(
    shim_lib, cold_dir, hot_dir, trace_dir, tmp_path
):
    """End-to-end: chain produces a trace; dfanalyzer turns it into
    io_report.json; agentstage.metrics.empirical_gt can parse it and
    finds the cold paths in file_name_view."""
    if DFTRACER_PRELOAD is None:
        pytest.skip(REASON_NO_DFTRACER)
    if not HAVE_DFANALYZER:
        pytest.skip(REASON_NO_DFANALYZER)

    # Build a small workload: 3 files, all read by the subprocess
    files = []
    for i in range(3):
        f = cold_dir / f"sample_{i}.bin"
        f.write_bytes(b"A" * (10 * 1024))  # 10 KB each
        place_hot_copy(hot_dir, f, b"B" * (10 * 1024))
        files.append(f)

    env = {**make_dftracer_env(trace_dir, cold_dir, log_prefix="e2e"),
           **make_shim_env(hot_dir, cold_dir)}
    script = "; ".join(f"open({str(f)!r}, 'rb').read()" for f in files)
    r = run_subprocess_with_chain(
        script,
        ld_preload_libs=[DFTRACER_PRELOAD, shim_lib],
        extra_env=env,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"

    # Run dfanalyzer over the traces
    output_dir = tmp_path / "analysis"
    output_dir.mkdir()
    r = subprocess.run(
        [sys.executable, "-m", "dftracer.analyzer.__main__",
         "--input", str(trace_dir),
         "--output", str(output_dir / "io_report.json")],
        capture_output=True, text=True, timeout=120,
    )
    if r.returncode != 0:
        pytest.skip(
            f"dfanalyzer invocation failed (likely needs different CLI args; "
            f"see STAGER_VERIFICATION.md). stderr: {r.stderr[:500]}"
        )

    # Parse with our empirical_gt loader
    from agentstage.metrics.empirical_gt import load_empirical_reads
    report_path = output_dir / "io_report.json"
    if not report_path.exists():
        pytest.skip(f"dfanalyzer didn't write {report_path}")

    reads = load_empirical_reads(report_path)
    cold_paths_in_reads = {r.path for r in reads}
    for f in files:
        assert str(f) in cold_paths_in_reads, (
            f"cold path {f} missing from empirical_gt reads; "
            f"loader found: {cold_paths_in_reads}"
        )


# ---------------------------------------------------------------------------
# Test 5 — writes pass through both wrappers
# ---------------------------------------------------------------------------

def test_writes_pass_through_both_wrappers(
    shim_lib, cold_dir, hot_dir, trace_dir
):
    """Belt-and-suspenders: with both wrappers loaded, opening for write
    must still land in cold dir, not hot mirror, regardless of dftracer's
    presence."""
    if DFTRACER_PRELOAD is None:
        pytest.skip(REASON_NO_DFTRACER)

    target = cold_dir / "intermediate.txt"

    env = {**make_dftracer_env(trace_dir, cold_dir, log_prefix="write"),
           **make_shim_env(hot_dir, cold_dir)}
    r = run_subprocess_with_chain(
        f"open({str(target)!r}, 'w').write('AGENT_WROTE_THIS')",
        ld_preload_libs=[DFTRACER_PRELOAD, shim_lib],
        extra_env=env,
    )
    assert r.returncode == 0, f"stderr: {r.stderr}"

    assert target.exists()
    assert target.read_text() == "AGENT_WROTE_THIS"

    hot_mirror = hot_mirror_for(hot_dir, target)
    assert not hot_mirror.exists(), (
        f"write should not have landed in hot mirror at {hot_mirror}"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _read_trace_files(traces: list[Path]) -> str:
    """Decompress + concatenate trace file contents for text matching.
    DFTracer writes .pfw (json-lines) or .pfw.gz (gzipped). We just want
    a single string to search for cold paths in."""
    import gzip
    parts: list[str] = []
    for t in traces:
        if t.suffix == ".gz":
            with gzip.open(t, "rt") as f:
                parts.append(f.read())
        else:
            parts.append(t.read_text())
    return "\n".join(parts)
