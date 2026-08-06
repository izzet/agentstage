"""Layer-2 LD_PRELOAD shim tests.

Build the shim, spawn subprocesses with LD_PRELOAD set, verify the shim
intercepts the right syscalls and redirects correctly. Equivalent to the
pure-C harness; chosen pytest-Python over raw C
for ease of authorship + maintenance.

Covered:
  - opens under managed cold roots redirect to hot when hot exists
  - opens fall through to cold when hot is missing (ENOENT)
  - retry-spin catches a late-arriving rename
  - writes (O_WRONLY/O_CREAT) always pass through to cold
  - stats are redirected so file sizes match hot copies
  - DFTracer load-order compatibility (smoke: shim doesn't break when
    another LD_PRELOAD lib precedes it)
"""

from __future__ import annotations

import os
import shutil
import subprocess
import threading
import time
from pathlib import Path

import pytest


SHIM_DIR = (
    Path(__file__).parent.parent
    / "src" / "agentstage" / "stager" / "shim"
)
SHIM_SO = SHIM_DIR / "libagentstage_shim.so"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def shim_lib() -> Path:
    """Build the shim once per module."""
    if not SHIM_SO.exists() or SHIM_SO.stat().st_mtime < (SHIM_DIR / "agentstage_shim.c").stat().st_mtime:
        subprocess.run(["make", "-C", str(SHIM_DIR)], check=True, capture_output=True)
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


def shim_env(shim_lib: Path, hot_dir: Path, cold_dir: Path, *,
             retry_spin_ms: int = 20, log_path: Path | None = None,
             extra_ld_preload: str = "") -> dict[str, str]:
    """Build the env dict for spawning a subprocess with the shim loaded."""
    env = os.environ.copy()
    preload = str(shim_lib)
    if extra_ld_preload:
        preload = f"{extra_ld_preload}:{preload}"
    env["LD_PRELOAD"] = preload
    env["AGENTSTAGE_HOT_ROOT"] = str(hot_dir)
    env["AGENTSTAGE_COLD_ROOTS"] = str(cold_dir)
    env["AGENTSTAGE_RETRY_SPIN_MS"] = str(retry_spin_ms)
    if log_path is not None:
        env["AGENTSTAGE_SHIM_LOG"] = str(log_path)
    return env


def run_python(env: dict, script: str) -> subprocess.CompletedProcess:
    """Run a one-liner Python script in a subprocess with the given env."""
    return subprocess.run(
        ["python3", "-c", script],
        env=env, capture_output=True, text=True, timeout=30,
    )


def run_cat(env: dict, path: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["cat", str(path)], env=env, capture_output=True, text=True, timeout=10,
    )


def hot_mirror_for(hot_dir: Path, cold_path: Path) -> Path:
    """Mirror cold_path under hot_dir using absolute-path mirroring."""
    abs_cold = str(cold_path.resolve())
    return hot_dir / abs_cold.lstrip("/")


def place_hot_copy(hot_dir: Path, cold_path: Path, content: bytes) -> Path:
    hot = hot_mirror_for(hot_dir, cold_path)
    hot.parent.mkdir(parents=True, exist_ok=True)
    hot.write_bytes(content)
    return hot


# ---------------------------------------------------------------------------
# T1: opens under cold root redirect to hot
# ---------------------------------------------------------------------------

def test_open_redirects_to_hot(shim_lib, cold_dir, hot_dir, tmp_path):
    cold_file = cold_dir / "doc.txt"
    cold_file.write_text("COLD")
    place_hot_copy(hot_dir, cold_file, b"HOT")

    log = tmp_path / "shim.log"
    env = shim_env(shim_lib, hot_dir, cold_dir, log_path=log)
    r = run_python(env, f"print(open({str(cold_file)!r}).read())")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert r.stdout.strip() == "HOT", f"got {r.stdout!r}"

    log_text = log.read_text() if log.exists() else ""
    assert "HIT" in log_text, f"expected HIT in log, got:\n{log_text}"


def test_cat_also_redirects(shim_lib, cold_dir, hot_dir):
    """`cat` uses open() not openat() — verifies our open alias works."""
    cold_file = cold_dir / "doc.txt"
    cold_file.write_text("COLD")
    place_hot_copy(hot_dir, cold_file, b"HOT_FROM_CAT")

    env = shim_env(shim_lib, hot_dir, cold_dir)
    r = run_cat(env, cold_file)
    assert r.returncode == 0
    assert r.stdout == "HOT_FROM_CAT"


# ---------------------------------------------------------------------------
# T2: fall through to cold when hot is missing
# ---------------------------------------------------------------------------

def test_falls_through_when_hot_missing(shim_lib, cold_dir, hot_dir):
    """No hot copy → shim returns cold fd after retry spin."""
    cold_file = cold_dir / "only_cold.txt"
    cold_file.write_text("COLD_ONLY")

    env = shim_env(shim_lib, hot_dir, cold_dir, retry_spin_ms=5)
    r = run_python(env, f"print(open({str(cold_file)!r}).read())")
    assert r.returncode == 0
    assert r.stdout.strip() == "COLD_ONLY"


def test_unmanaged_cold_root_passes_through(shim_lib, cold_dir, hot_dir, tmp_path):
    """Files outside the managed cold roots are never touched by the shim."""
    other = tmp_path / "other"
    other.mkdir()
    target = other / "outside.txt"
    target.write_text("OUTSIDE")

    # Even though hot has a file at the mirror, shim shouldn't redirect
    # because /tmp/other isn't in AGENTSTAGE_COLD_ROOTS
    place_hot_copy(hot_dir, target, b"SHOULD_NOT_BE_SEEN")

    env = shim_env(shim_lib, hot_dir, cold_dir)
    r = run_python(env, f"print(open({str(target)!r}).read())")
    assert r.returncode == 0
    assert r.stdout.strip() == "OUTSIDE"


# ---------------------------------------------------------------------------
# T3: retry-spin catches a late-arriving rename
# ---------------------------------------------------------------------------

def test_retry_spin_catches_late_rename(shim_lib, cold_dir, hot_dir, tmp_path):
    """Background thread renames hot file into place 10 ms after openat starts.
    Shim's retry-spin should catch it before falling through."""
    cold_file = cold_dir / "race.txt"
    cold_file.write_text("COLD_RACE")
    hot_path = hot_mirror_for(hot_dir, cold_file)
    hot_path.parent.mkdir(parents=True, exist_ok=True)
    hot_tmp = hot_path.with_suffix(".tmp")
    hot_tmp.write_bytes(b"HOT_LATE")

    # Use a small Python script that spawns a "renamer" thread, then opens
    script = f"""
import os, threading, time, sys

def renamer():
    time.sleep(0.010)  # 10 ms — within the 50 ms retry-spin budget
    os.rename({str(hot_tmp)!r}, {str(hot_path)!r})

t = threading.Thread(target=renamer)
t.start()
content = open({str(cold_file)!r}).read()
t.join()
sys.stdout.write(content)
"""
    env = shim_env(shim_lib, hot_dir, cold_dir, retry_spin_ms=50)
    r = run_python(env, script)
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert r.stdout == "HOT_LATE", (
        f"retry-spin failed to catch rename; got {r.stdout!r}"
    )


def test_retry_spin_falls_through_after_budget(shim_lib, cold_dir, hot_dir):
    """If hot never appears, shim falls through to cold within ~budget ms."""
    cold_file = cold_dir / "never_hot.txt"
    cold_file.write_text("COLD_FALLBACK")

    env = shim_env(shim_lib, hot_dir, cold_dir, retry_spin_ms=10)
    t0 = time.monotonic()
    r = run_python(env, f"print(open({str(cold_file)!r}).read())")
    elapsed_ms = (time.monotonic() - t0) * 1000
    assert r.returncode == 0
    assert r.stdout.strip() == "COLD_FALLBACK"
    # Loose upper bound: python startup is ~50 ms; even with a few retry
    # spins per openat the total should comfortably stay under 1 second.
    assert elapsed_ms < 2000, f"took {elapsed_ms:.0f} ms — retry-spin may be unbounded"


# ---------------------------------------------------------------------------
# T4: writes pass through to cold
# ---------------------------------------------------------------------------

def test_writes_pass_through_to_cold(shim_lib, cold_dir, hot_dir):
    """Opening a file under cold_dir with O_WRONLY|O_CREAT must land in cold,
    NOT in the hot mirror."""
    target = cold_dir / "agent_output.txt"

    env = shim_env(shim_lib, hot_dir, cold_dir)
    script = f"""
with open({str(target)!r}, 'w') as f:
    f.write('AGENT_WROTE_THIS')
"""
    r = run_python(env, script)
    assert r.returncode == 0, f"stderr: {r.stderr}"

    # The file should exist in cold_dir, NOT in the hot mirror
    assert target.exists()
    assert target.read_text() == "AGENT_WROTE_THIS"

    hot_mirror = hot_mirror_for(hot_dir, target)
    assert not hot_mirror.exists(), (
        f"write should not have landed in hot mirror at {hot_mirror}"
    )


# ---------------------------------------------------------------------------
# T5: stats redirect — file size matches hot copy
# ---------------------------------------------------------------------------

def test_stat_returns_hot_size_when_redirected(shim_lib, cold_dir, hot_dir):
    """os.stat() on a cold path should return the HOT file's size when
    the file is staged. This matters because agents allocate buffers
    based on the stat result; mismatch would corrupt reads."""
    cold_file = cold_dir / "stat.bin"
    cold_file.write_bytes(b"x" * 100)
    place_hot_copy(hot_dir, cold_file, b"y" * 200)

    env = shim_env(shim_lib, hot_dir, cold_dir)
    script = f"""
import os
st = os.stat({str(cold_file)!r})
print(st.st_size)
"""
    r = run_python(env, script)
    assert r.returncode == 0, f"stderr: {r.stderr}"
    # If shim redirects stat, we should see 200 (hot); if not, we see 100 (cold)
    assert r.stdout.strip() == "200", (
        f"stat returned {r.stdout!r}; expected hot file size 200"
    )


# ---------------------------------------------------------------------------
# T6: DFTracer load-order compatibility (smoke)
# ---------------------------------------------------------------------------

def test_shim_works_with_dftracer_before_it(shim_lib, cold_dir, hot_dir):
    """LD_PRELOAD chain: dftracer + agentstage shim together.

    Verifies our shim still works when dftracer is in the chain. The
    cold/hot redirect must function correctly regardless of dftracer's
    presence. The detailed chain semantics live in
    tests/test_dftracer_chain.py; this is the spot-check that our shim's
    own test suite has dftracer-compatibility coverage.
    """
    # Reuse the chain test's dftracer discovery
    from tests.test_dftracer_chain import _find_dftracer_preload
    dftracer = _find_dftracer_preload()
    if dftracer is None:
        pytest.skip(
            "libdftracer_preload.so not found; set "
            "AGENTSTAGE_DFTRACER_PRELOAD or build external/libs/dftracer/"
        )

    cold_file = cold_dir / "chained.txt"
    cold_file.write_text("COLD_CHAINED")
    place_hot_copy(hot_dir, cold_file, b"HOT_CHAINED")

    env = shim_env(shim_lib, hot_dir, cold_dir, extra_ld_preload=str(dftracer))
    # Minimal dftracer config so it doesn't crash on missing env vars
    env["DFTRACER_ENABLE"] = "0"  # we don't care about traces here, just chain compat

    r = run_python(env, f"print(open({str(cold_file)!r}).read())")
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert r.stdout.strip() == "HOT_CHAINED", (
        "shim didn't redirect when dftracer was loaded first; chain broken"
    )


# ---------------------------------------------------------------------------
# T7: AGENTSTAGE_SHIM_DISABLE=1 makes the shim a no-op
# ---------------------------------------------------------------------------

def test_shim_disable_makes_passthrough(shim_lib, cold_dir, hot_dir):
    """When AGENTSTAGE_SHIM_DISABLE=1, shim is loaded but does nothing."""
    cold_file = cold_dir / "disabled.txt"
    cold_file.write_text("COLD_DISABLED")
    place_hot_copy(hot_dir, cold_file, b"HOT_DISABLED")

    env = shim_env(shim_lib, hot_dir, cold_dir)
    env["AGENTSTAGE_SHIM_DISABLE"] = "1"
    r = run_python(env, f"print(open({str(cold_file)!r}).read())")
    assert r.returncode == 0
    assert r.stdout.strip() == "COLD_DISABLED"


# ---------------------------------------------------------------------------
# T8: byte-identity check end-to-end
# ---------------------------------------------------------------------------

def test_redirect_preserves_byte_identity(shim_lib, cold_dir, hot_dir):
    """Important correctness invariant: after the shim redirects, the
    bytes the agent reads must equal the hot file's bytes exactly.
    Verifies sha256(read_through_shim) == sha256(direct_read_of_hot)."""
    import hashlib

    cold_file = cold_dir / "bigfile.bin"
    cold_file.write_bytes(b"\x00" * (4 * 1024 * 1024))  # 4 MB of zeros

    hot_content = bytes(range(256)) * (16 * 1024)  # ~4 MB structured pattern
    hot_path = place_hot_copy(hot_dir, cold_file, hot_content)
    expected_hash = hashlib.sha256(hot_path.read_bytes()).hexdigest()

    env = shim_env(shim_lib, hot_dir, cold_dir)
    script = f"""
import hashlib
data = open({str(cold_file)!r}, 'rb').read()
print(hashlib.sha256(data).hexdigest())
"""
    r = run_python(env, script)
    assert r.returncode == 0, f"stderr: {r.stderr}"
    assert r.stdout.strip() == expected_hash, "bytes diverge from hot file"
