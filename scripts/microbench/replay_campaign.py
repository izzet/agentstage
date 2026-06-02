"""Trajectory-controlled replay campaign.

Replay a curated list of baseline sessions in both cold and staged
modes. Each cell uses the baseline rep with the highest rc=0 shell
time (i.e., the most mechanism-actionable I/O in the recorded
trajectory).

Sessions are selected to:
  - Match prompt_mode (hinted) for cross-mode comparability
  - Have at least 30 seconds of rc=0 shell time (mechanism scope)
  - Have ≤ 1 shell timeout (kills introduce trajectory bias)
  - Belong to one of the 3 paper-named Curated tasks
    (aiob_104/107/110) — skip aiob_103

Run:
    uv run python scripts/microbench/replay_campaign.py \
        --out outputs/replay/campaign.json
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REPLAY_SCRIPT = REPO / "scripts" / "microbench" / "replay_session.py"

_RC_RE = re.compile(r"run_shell_command \(rc=(-?\d+), ([0-9.]+)s\)")


def model_key(m: str) -> str | None:
    m = (m or "").lower()
    if "claude-haiku" in m: return "haiku"
    if "claude-sonnet" in m: return "sonnet"
    if "gemini-2.5-flash" in m or "gemini-flash" in m: return "flash"
    if "qwen" in m: return "qwen3"
    return None


def session_stats(sf: Path) -> dict:
    try:
        s = json.loads(sf.read_text())
    except Exception:
        return {}
    n_kills = 0
    sum_rc0 = 0.0
    for t in s.get("per_turn", []):
        tn = t["turn"]
        tdir = sf.parent / "turns" / f"turn_{tn:02d}"
        tr = tdir / "tool_result.jsonl"
        if not tr.exists(): continue
        for line in tr.read_text().splitlines():
            try: d = json.loads(line)
            except: continue
            m = _RC_RE.search(d.get("content", ""))
            if m:
                rc, el = int(m.group(1)), float(m.group(2))
                if rc == -9: n_kills += 1
                if rc == 0: sum_rc0 += el
    return {
        "elapsed_s": s.get("session_elapsed_s"),
        "submitted": s.get("submitted"),
        "prompt_mode": s.get("prompt_mode"),
        "n_kills": n_kills,
        "rc0_shell_s": sum_rc0,
        "task": s.get("task"),
        "model": s.get("model"),
    }


def select_best_baseline(bench: str, mdl: str, task: str,
                          min_rc0: float = 30.0,
                          max_kills: int = 1) -> Path | None:
    """Pick the baseline rep for (bench, mdl, task) with most rc=0
    shell time, matching prompt_mode=hinted, submitted, ≤ max_kills."""
    candidates = []
    base = REPO / "outputs" / f"{bench}_mt"
    if not base.is_dir():
        return None
    for sf in base.rglob("summary.json"):
        if "_smoke" in str(sf) or "_archive" in str(sf): continue
        try: s = json.loads(sf.read_text())
        except: continue
        if s.get("task") != task: continue
        if s.get("mode") != "baseline": continue
        if model_key(s.get("model")) != mdl: continue
        if s.get("prompt_mode") != "hinted": continue
        st = session_stats(sf)
        if not st.get("submitted"): continue
        if st["n_kills"] > max_kills: continue
        if st["rc0_shell_s"] < min_rc0: continue
        candidates.append((sf, st["rc0_shell_s"]))
    if not candidates:
        return None
    candidates.sort(key=lambda x: -x[1])
    return candidates[0][0]


def run_replay(session: Path, mode: str, out_path: Path,
               hot_root: Path) -> dict:
    """Invoke replay_session.py via subprocess; return its JSON result.
    `session` may be a summary.json path or its parent dir — we pass
    the parent to replay_session.py which expects a session dir."""
    session_dir = session.parent if session.is_file() else session
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Sanity: ensure /dev/shm has enough room (>20 GB) before staging.
    # A full tmpfs silently breaks the Stager's copy and our measurement.
    import shutil as _sh
    free = _sh.disk_usage("/dev/shm").free
    if mode == "staged" and free < 20 * 1024**3:
        return {"error": f"/dev/shm only has {free/1e9:.1f} GB free; need >=20 GB"}
    proc = subprocess.run(
        ["uv", "run", "python", str(REPLAY_SCRIPT),
         "--session", str(session_dir),
         "--mode", mode,
         "--out", str(out_path),
         "--hot-root", str(hot_root)],
        cwd=str(REPO),
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        return {"error": "replay failed", "rc": proc.returncode,
                "stderr_tail": proc.stderr[-2000:]}
    if not out_path.exists():
        return {"error": "no output file", "stderr_tail": proc.stderr[-2000:]}
    return json.loads(out_path.read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/replay/campaign.json")
    ap.add_argument("--hot-root", default="/dev/shm/agentstage_replay")
    ap.add_argument("--include-qwen3", action="store_true",
                    help="Also replay 1 Qwen3 cell (vLLM caveat applies)")
    args = ap.parse_args()

    # Cell roster — re-running cells with bad fidelity (DSBench/MLE)
    # + 1 aiob control. With full eviction (no 200-cap) we should see
    # accurate read_bytes and a true mechanism speedup.
    cells = [
        ("aiob", "haiku",  "aiob_110"),  # control — already 39 files only
        ("dsbench", "haiku", "ventilator-pressure-prediction"),
        ("mlebench", "haiku", "dogs-vs-cats-redux-kernels-edition"),
        ("mlebench", "sonnet", "new-york-city-taxi-fare-prediction"),
    ]
    if args.include_qwen3:
        cells.append(("mlebench", "qwen3", "dogs-vs-cats-redux-kernels-edition"))

    out_dir = Path(args.out).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== REPLAY CAMPAIGN ({len(cells)} cells) ===", flush=True)
    print("Selecting best baseline rep per cell...", flush=True)
    selected: list[tuple] = []
    for bench, mdl, task in cells:
        # for Qwen3 allow more kills (but still need ≥30s rc=0)
        max_k = 2 if mdl == "qwen3" else 1
        sf = select_best_baseline(bench, mdl, task, min_rc0=30.0,
                                    max_kills=max_k)
        if sf is None:
            print(f"  SKIP {bench}/{mdl}/{task}: no qualifying baseline", flush=True)
            continue
        st = session_stats(sf)
        print(f"  {bench:9s}/{mdl:7s}/{task[:30]:30s} → "
              f"{sf.parent.name} (elapsed={st['elapsed_s']:.0f}s, "
              f"rc0={st['rc0_shell_s']:.0f}s, kills={st['n_kills']})",
              flush=True)
        selected.append((bench, mdl, task, sf, st))

    print(f"\nReplay schedule: {len(selected)} cells x 2 modes "
          f"= {2*len(selected)} runs. Est. wall: "
          f"{sum(s[4]['elapsed_s'] for s in selected)*2/60:.0f} min", flush=True)
    print(flush=True)

    results = []
    t0 = time.monotonic()
    for i, (bench, mdl, task, sf, st) in enumerate(selected):
        cell_key = f"{bench}_{mdl}_{task}".replace("-","_").replace(" ","_")
        print(f"[{i+1}/{len(selected)}] {bench}/{mdl}/{task} "
              f"(rep {sf.parent.name})", flush=True)
        cell_record = {
            "bench": bench, "model": mdl, "task": task,
            "rep_dir": str(sf.parent),
            "original_elapsed_s": st["elapsed_s"],
            "original_rc0_shell_s": st["rc0_shell_s"],
            "original_n_kills": st["n_kills"],
            "submitted": st["submitted"],
        }
        for mode in ("cold", "staged"):
            mode_t0 = time.monotonic()
            out_path = out_dir / f"{cell_key}_{mode}.json"
            print(f"  [{mode}] running...", flush=True)
            r = run_replay(sf, mode, out_path, Path(args.hot_root))
            elapsed = time.monotonic() - mode_t0
            if "error" in r:
                print(f"    ERROR: {r['error']} — {r.get('stderr_tail','')[-300:]}", flush=True)
                cell_record[f"{mode}_error"] = r["error"]
            else:
                print(f"    elapsed={r['session_elapsed_s']:.1f}s "
                      f"(wallclock {elapsed:.0f}s), "
                      f"fidelity={r.get('fidelity_vs_baseline_minus_kills', '?')}", flush=True)
                cell_record[f"{mode}_replay_s"] = r["session_elapsed_s"]
                cell_record[f"{mode}_fidelity"] = r.get("fidelity_vs_baseline_minus_kills")
                if mode == "staged":
                    cell_record["n_prefetched"] = r.get("n_prefetched_files", 0)
        # Mechanism speedup
        if "cold_replay_s" in cell_record and "staged_replay_s" in cell_record:
            cell_record["mechanism_speedup"] = round(
                cell_record["cold_replay_s"] / cell_record["staged_replay_s"], 3)
            print(f"  mechanism speedup: {cell_record['mechanism_speedup']}×", flush=True)
        else:
            cell_record["mechanism_speedup"] = None
        results.append(cell_record)
        # Save after each cell so we don't lose data if the campaign crashes
        Path(args.out).write_text(json.dumps(results, indent=2))
        print(flush=True)

    print(f"=== DONE in {(time.monotonic()-t0)/60:.1f} min ===", flush=True)
    print(f"Results: {args.out}", flush=True)
    print(flush=True)
    print(f"{'cell':50s}  {'cold':>7s}  {'staged':>7s}  {'mech':>6s}  {'fid_cold':>8s}")
    for r in results:
        if r.get("mechanism_speedup") is None:
            continue
        cell = f"{r['bench']}/{r['model']}/{r['task'][:25]}"
        print(f"{cell[:50]:50s}  {r['cold_replay_s']:>7.1f}  "
              f"{r['staged_replay_s']:>7.1f}  "
              f"{r['mechanism_speedup']:>5.2f}×  "
              f"{r['cold_fidelity']:>7.2f}×")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
