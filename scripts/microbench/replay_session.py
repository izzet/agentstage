"""1-1 trajectory replay of a baseline session.

Replays a recorded baseline session's events (thinking + text deltas
at their recorded t_ms timestamps, exact tool calls in order) under
either cold or staged I/O conditions, and measures total wall-clock
time. By holding the agent's trajectory fixed and varying only the
I/O path, we eliminate LLM-side stochasticity and isolate AgentStage's
mechanism speedup.

Methodology:
  - Thinking/text deltas are grouped by block_id and replayed at the
    LAST delta's recorded t_ms (= original block-completion time).
  - For each turn we compute implied_streaming_s = duration_s -
    sum(rc=0 shell_elapsed_s). After replaying recorded deltas, we
    sleep out the remaining streaming time. This captures LLM-emit
    time for tool_use blocks (whose deltas we never logged).
  - For each tool_use: rc=-9 (timed-out) shells are SKIPPED
    symmetrically in both modes (agent-side errors, not mechanism).
    Other tools are executed via the same make_tool_executor as the
    live runner.
  - Staged mode: SessionDetector consumes the replayed thinking/text
    blocks → AutoRuleGenerator ruleset fires → Stager prefetches.
    Shell tools run under LD_PRELOAD pointing at /dev/shm.

Output: per-(session, mode) elapsed + per-turn breakdown.

Usage:
    uv run python scripts/microbench/replay_session.py \
        --session outputs/aiob_mt/.../aiob_104_baseline_r1 \
        --mode cold \
        --out outputs/replay/aiob_104_baseline_r1_cold.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

# Load .env so any inherited env vars in the live runner exist here too.
def _load_dotenv(p: Path) -> None:
    if not p.is_file():
        return
    import os
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        k, _, v = line.partition("=")
        if k and v and k.strip() not in os.environ:
            import os as _os
            _os.environ[k.strip()] = v.strip().strip('"').strip("'")

_load_dotenv(REPO / ".env")
if os.environ.get("SCIIOBENCH_ROOT"):
    _load_dotenv(Path(os.environ["SCIIOBENCH_ROOT"]) / ".env")

from agentstage.detector.auto_rules import AutoRuleGenerator  # noqa: E402
from agentstage.detector.engine import StreamBlock  # noqa: E402
from agentstage.detector.session import SessionDetector  # noqa: E402
from agentstage.stager import DataHint, Stager  # noqa: E402
from agentstage.workloads.aiob import (  # noqa: E402
    load_aiob_103, load_aiob_104, load_aiob_107, load_aiob_110,
    load_aiob_201, load_aiob_202, load_aiob_203, load_aiob_204, load_aiob_205,
)
from agentstage.workloads.kramabench import (  # noqa: E402
    load_kb_astronomy_inventory, load_kb_wildfire_inventory,
)
from agentstage.workloads.dsbench import (  # noqa: E402
    load_dsbench_task, load_dsbench_integrity_manifest,
    load_dsbench_integrity_single,
)
from agentstage.workloads.mlebench import (  # noqa: E402
    load_mle_competition, load_mle_competition_dispatch,
    load_mle_integrity_manifest, load_mle_dogsvcats_integrity,
    load_mle_dogsvcats_thumbhash, load_mle_histopath_thumbhash,
)

# Each multiturn runner has its own make_tool_executor with bench-specific
# logic (sandboxing, symlink layout). Import all three.
sys.path.insert(0, str(REPO / "scripts" / "microbench"))
from aiob_multiturn import (  # noqa: E402
    make_tool_executor as aiob_executor,
    resolve_logical, STAGER_BUCKET_CAP, evict,
)
from dsbench_multiturn import make_tool_executor as dsbench_executor  # noqa: E402
from mlebench_multiturn import make_tool_executor as mlebench_executor  # noqa: E402


def load_workload_for_task(task_id: str, bench: str):
    """Return the loaded workload + the correct make_tool_executor fn
    for the given bench (aiob | dsbench | mlebench)."""
    aiob_loaders = {
        "aiob_103": load_aiob_103, "aiob_104": load_aiob_104,
        "aiob_107": load_aiob_107, "aiob_110": load_aiob_110,
        "aiob_201": load_aiob_201, "aiob_202": load_aiob_202,
        "aiob_203": load_aiob_203, "aiob_204": load_aiob_204,
        "aiob_205": load_aiob_205,
        # KramaBench / DSBench / MLE-bench integrity tasks via AIOB
        # runner (interface-compatible workload types).
        "kb_astronomy_inventory": load_kb_astronomy_inventory,
        "kb_wildfire_inventory": load_kb_wildfire_inventory,
        "dsb_integrity_manifest": load_dsbench_integrity_manifest,
        "dsb_integrity_single": load_dsbench_integrity_single,
        "mle_integrity_manifest": load_mle_integrity_manifest,
        "mle_dogsvcats_integrity": load_mle_dogsvcats_integrity,
        "mle_dogsvcats_thumbhash": load_mle_dogsvcats_thumbhash,
        "mle_histopath_thumbhash": load_mle_histopath_thumbhash,
    }
    if bench == "aiob" and task_id in aiob_loaders:
        return aiob_loaders[task_id](), aiob_executor
    if bench == "dsbench":
        return load_dsbench_task(task_id), dsbench_executor
    if bench == "mlebench":
        return load_mle_competition_dispatch(task_id), mlebench_executor
    raise ValueError(f"Unknown task '{task_id}' for bench '{bench}'")

_RC_ELAPSED_RE = re.compile(r"run_shell_command \(rc=(-?\d+), ([0-9.]+)s\)")


# ---------------------------------------------------------------------------
# Trajectory loader
# ---------------------------------------------------------------------------

def load_trajectory(session_dir: Path) -> dict:
    """Parse a recorded session into a replayable trajectory."""
    summary = json.loads((session_dir / "summary.json").read_text())
    turns = []
    for t in summary["per_turn"]:
        tn = t["turn"]
        tdir = session_dir / "turns" / f"turn_{tn:02d}"

        # Deltas grouped by block_id
        thinking_blocks: dict[int, list[dict]] = defaultdict(list)
        text_blocks: dict[int, list[dict]] = defaultdict(list)
        for fn, target in (("thinking.jsonl", thinking_blocks),
                            ("text.jsonl", text_blocks)):
            p = tdir / fn
            if not p.exists():
                continue
            for line in p.read_text(errors="replace").splitlines():
                if not line.strip():
                    continue
                d = json.loads(line)
                target[d["block"]].append(d)
        # Build block entries: (t_stop_ms, block_type, joined_text)
        blocks = []
        for bid, deltas in thinking_blocks.items():
            deltas.sort(key=lambda x: x["t_ms"])
            t_stop = deltas[-1]["t_ms"]
            text = "".join(d["delta"] for d in deltas)
            blocks.append({"t_stop_ms": t_stop, "type": "thinking",
                           "text": text, "block_id": bid})
        for bid, deltas in text_blocks.items():
            deltas.sort(key=lambda x: x["t_ms"])
            t_stop = deltas[-1]["t_ms"]
            text = "".join(d["delta"] for d in deltas)
            blocks.append({"t_stop_ms": t_stop, "type": "text",
                           "text": text, "block_id": bid})
        blocks.sort(key=lambda b: b["t_stop_ms"])

        # Tool calls + results
        tool_uses = []
        if (tdir / "tool_use.jsonl").exists():
            for line in (tdir / "tool_use.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                tool_uses.append(json.loads(line))
        tool_results = []
        if (tdir / "tool_result.jsonl").exists():
            for line in (tdir / "tool_result.jsonl").read_text().splitlines():
                if not line.strip():
                    continue
                tool_results.append(json.loads(line))

        # Index tool_results by tool_use_id for fast lookup
        result_by_id = {r["tool_use_id"]: r for r in tool_results}

        # For each tool_use, parse rc + recorded elapsed (shells only)
        parsed_tools = []
        sum_all_shell_elapsed_s = 0.0
        sum_killed_shell_elapsed_s = 0.0
        for tu in tool_uses:
            r = result_by_id.get(tu.get("id"))
            content = (r or {}).get("content", "")
            m = _RC_ELAPSED_RE.search(content)
            rc, recorded_elapsed = (None, 0.0)
            if m:
                rc = int(m.group(1))
                recorded_elapsed = float(m.group(2))
            parsed_tools.append({
                "name": tu["name"],
                "parsed_input": tu.get("parsed_input", {}),
                "tool_use_id": tu.get("id"),
                "recorded_rc": rc,
                "recorded_elapsed_s": recorded_elapsed,
            })
            sum_all_shell_elapsed_s += recorded_elapsed
            if rc == -9:
                sum_killed_shell_elapsed_s += recorded_elapsed

        # Implied LLM streaming time = duration - sum(ALL shell elapsed
        # including kills). We sleep this much, then execute each shell
        # (skipping kills entirely). This means replay total ≈ original
        # minus killed-shell time, the "successful-work" baseline.
        duration_s = t.get("duration_s") or 0.0
        last_delta_ms = max((b["t_stop_ms"] for b in blocks), default=0.0)
        implied_stream_s = max(
            duration_s - sum_all_shell_elapsed_s,
            last_delta_ms / 1000.0,
        )

        turns.append({
            "turn": tn,
            "duration_s": duration_s,
            "blocks": blocks,
            "tools": parsed_tools,
            "implied_stream_s": implied_stream_s,
            "last_delta_ms": last_delta_ms,
            "sum_all_shell_elapsed_s": sum_all_shell_elapsed_s,
            "sum_killed_shell_elapsed_s": sum_killed_shell_elapsed_s,
        })

    return {
        "summary": summary,
        "turns": turns,
        "session_dir": session_dir,
        "n_kills": sum(1 for t in turns for c in t["tools"]
                       if c["recorded_rc"] == -9),
        "killed_shell_s": sum(c["recorded_elapsed_s"] for t in turns
                              for c in t["tools"]
                              if c["recorded_rc"] == -9),
    }


# ---------------------------------------------------------------------------
# Replay execution
# ---------------------------------------------------------------------------

def _dispatch_prefetches(*, new_acts, stager, prefix_map, turn,
                          source, dispatched, n_total_holder):
    """Mirror of aiob_multiturn._dispatch_prefetches."""
    if stager is None:
        return
    seen_phys: set[str] = set()
    # On large workloads (e.g. dogvscats: 22500 files), the is_file() check
    # below becomes the bottleneck because each Path.is_file() is a network
    # metadata RPC on OrangeFS (1-3ms). For 22500 files that's 30-60s blocking
    # the main thread. Stager handles missing files gracefully (failed future)
    # so we can skip the pre-check; trust the workload's workspace_prior.
    # Env var SKIP_PREFETCH_ISFILE_CHECK=0 restores the old behavior.
    skip_isfile = os.environ.get("SKIP_PREFETCH_ISFILE_CHECK", "1") == "1"
    for act in new_acts:
        phys_files = [resolve_logical(p, prefix_map) for p in act.detected_files]
        if skip_isfile:
            phys_files = [p for p in phys_files if p not in seen_phys]
        else:
            phys_files = [p for p in phys_files
                          if Path(p).is_file() and p not in seen_phys]
        if len(phys_files) > STAGER_BUCKET_CAP:
            phys_files = phys_files[:STAGER_BUCKET_CAP]
        for p in phys_files:
            seen_phys.add(p)
        if not phys_files:
            continue
        hint = DataHint(
            detected_files=tuple(phys_files),
            tier=1 if len(phys_files) <= 10 else 3,
            fired_at_ms=act.fired_at_ms or 0.0,
            rule_id=f"turn{turn}:{act.rule_name}:{source}",
        )
        stager.prefetch(hint)
        dispatched.append({"rule": act.rule_name,
                           "n_files": len(phys_files),
                           "source": source})
        n_total_holder[0] += len(phys_files)


def replay(traj: dict, *, workload, executor_fn, mode: str,
           out_dir: Path, hot_root: Path) -> dict:
    """Replay a recorded baseline session under cold or staged I/O.
    Returns timing + per-turn breakdown."""
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    cold_root_anc = str(Path(data_phys_root).parent.parent)

    auto_rs = AutoRuleGenerator(
        workload_id=workload.task_id,
        task_instruction=workload.task.task_inst,
        workspace_prior_keys=tuple(workload.workspace_prior.keys()),
    ).generate()
    session_detector = SessionDetector(
        prior=workload.workspace_prior, ruleset=auto_rs,
    )

    stager = None
    if mode == "staged":
        if hot_root.exists():
            shutil.rmtree(hot_root)
        hot_root.mkdir(parents=True, exist_ok=True)
        _overflow = os.environ.get("AGENTSTAGE_HOT_OVERFLOW") or None
        _cap_gb = os.environ.get("AGENTSTAGE_HOT_PRIMARY_CAP_GB")
        _primary_cap = int(float(_cap_gb) * 1024**3) if _cap_gb else None
        _stager_workers = int(os.environ.get("STAGER_MAX_WORKERS", "4"))
        stager = Stager(
            hot_root=hot_root, cold_roots=[Path(cold_root_anc)],
            max_workers=_stager_workers, capacity_bytes=64 * 1024**3,
            hot_overflow_root=_overflow,
            hot_primary_capacity_bytes=_primary_cap,
        )

    workspace_dir = out_dir / "agent_workspace"
    if workspace_dir.exists():
        shutil.rmtree(workspace_dir)
    workspace_dir.mkdir(parents=True, exist_ok=True)

    shell_io_log: list[dict] = []
    _sh_to = int(os.environ.get("AGENTSTAGE_SHELL_TIMEOUT", "300"))
    execute = executor_fn(
        workload, workspace_dir, mode=mode, hot_root=hot_root,
        cold_root=cold_root_anc, shell_timeout=_sh_to,
        io_log=shell_io_log,
    )

    n_prefetched_holder = [0]
    per_turn: list[dict] = []

    # CRITICAL: evict OS page cache for workload files before measurement.
    # Without this, the cold-mode "baseline" reads from warm page cache
    # (especially after a prior staged run in the same campaign), masking
    # the cold-tier I/O penalty entirely. Matches live-runner methodology
    # but WITHOUT the 200-file cap — for many-file workloads (e.g.
    # dogs-vs-cats's 50k images) the cap means most reads silently hit
    # warm OS cache from prior runs and read_bytes under-reports.
    evict_targets: list[Path] = []
    for p in workload.all_workspace_paths:
        phys = resolve_logical(p, prefix_map)
        if Path(phys).is_file():
            evict_targets.append(Path(phys))
    print(f"[replay] evicting {len(evict_targets)} workload files...",
          flush=True)
    t_evict = time.monotonic()
    ev = evict(evict_targets, verify=False)
    print(f"[replay] eviction done in {time.monotonic()-t_evict:.1f}s, "
          f"{ev.get('files',0)} files / {ev.get('bytes',0)/1e9:.1f} GB",
          flush=True)
    evict_summary = {
        "n_files_evicted": ev.get("files", 0),
        "bytes_evicted": ev.get("bytes", 0),
    }

    t_session_start = time.monotonic()
    # Cap per-turn replayed stream time. Recorded sessions occasionally contain
    # multi-thousand-second "streaming" stalls from the OSS model's vLLM stream
    # hanging mid-turn (not real reasoning). Reproducing those as time.sleep()
    # makes a cell take ~the stalled wall time and dilutes the speedup toward 1.
    # Real reasoning turns are seconds; cap at 180s (env-overridable).
    _max_stream = float(os.environ.get("AGENTSTAGE_MAX_TURN_STREAM_S", "180"))
    for turn in traj["turns"]:
        t_turn_start = time.monotonic()

        # 1. Replay thinking + text blocks at recorded t_stop pace
        blocks_for_detector: list[StreamBlock] = []
        fired_rules_this_turn: list[str] = []
        dispatched_this_turn: list[dict] = []
        for b in turn["blocks"]:
            target_t = t_turn_start + min(b["t_stop_ms"] / 1000.0, _max_stream)
            now = time.monotonic()
            if target_t > now:
                time.sleep(target_t - now)
            sb = StreamBlock(
                type=b["type"], t_first=0, t_stop=int(b["t_stop_ms"]),
                text=b["text"], chunks=1, turn=turn["turn"],
            )
            new_acts = session_detector.feed_blocks([sb])
            fired_rules_this_turn.extend(a.rule_name for a in new_acts)
            if mode == "staged":
                _dispatch_prefetches(
                    new_acts=new_acts, stager=stager,
                    prefix_map=prefix_map, turn=turn["turn"],
                    source="thinking",
                    dispatched=dispatched_this_turn,
                    n_total_holder=n_prefetched_holder,
                )

        # 2. Sleep out remaining LLM streaming time (covers unsaved
        #    tool_use deltas). Implied = duration - rc=0 shell elapsed.
        end_of_stream = t_turn_start + min(turn["implied_stream_s"], _max_stream)
        now = time.monotonic()
        if end_of_stream > now:
            time.sleep(end_of_stream - now)

        # 3. Execute recorded tool calls in order. Skip rc=-9 (kills)
        #    symmetrically. Execute everything else (real shells in
        #    both modes; non-shell tools are deterministic via the
        #    executor).
        skipped_kills = 0
        executed_shells = 0
        replayed_outs = []
        tool_result_blocks_for_detector: list[StreamBlock] = []
        for c in turn["tools"]:
            if c["recorded_rc"] == -9:
                skipped_kills += 1
                continue
            out = execute(c["name"], c["parsed_input"])
            replayed_outs.append({"name": c["name"], "out_len": len(out)})
            if c["name"] == "run_shell_command":
                executed_shells += 1
            tool_result_blocks_for_detector.append(StreamBlock(
                type="tool_result", t_first=0, t_stop=0,
                text=out, chunks=1, turn=turn["turn"],
            ))

        if tool_result_blocks_for_detector:
            tr_acts = session_detector.feed_tool_results(
                tool_result_blocks_for_detector)
            fired_rules_this_turn.extend(a.rule_name for a in tr_acts)
            if mode == "staged" and tr_acts:
                _dispatch_prefetches(
                    new_acts=tr_acts, stager=stager,
                    prefix_map=prefix_map, turn=turn["turn"],
                    source="tool_result",
                    dispatched=dispatched_this_turn,
                    n_total_holder=n_prefetched_holder,
                )

        # Important: SessionDetector advances turn counter via
        # feed_turn(), not feed_blocks(). Since we're feeding blocks
        # one at a time, we need to bump current_turn manually.
        session_detector.current_turn = turn["turn"] + 1

        turn_elapsed = time.monotonic() - t_turn_start
        per_turn.append({
            "turn": turn["turn"],
            "elapsed_s": round(turn_elapsed, 3),
            "original_duration_s": turn["duration_s"],
            "n_shells_executed": executed_shells,
            "n_kills_skipped": skipped_kills,
            "fired_rules": fired_rules_this_turn,
            "dispatched_prefetches": dispatched_this_turn,
        })

    session_elapsed = time.monotonic() - t_session_start

    if stager is not None:
        stager.shutdown(wait=True)

    # Aggregate shell I/O across all calls
    total_rchar = sum(c.get("rchar", 0) for c in shell_io_log)
    total_read_bytes = sum(c.get("read_bytes", 0) for c in shell_io_log)
    total_syscr = sum(c.get("syscr", 0) for c in shell_io_log)
    total_shell_elapsed = sum(c.get("elapsed_s", 0.0) for c in shell_io_log)
    return {
        "session_elapsed_s": round(session_elapsed, 3),
        "mode": mode,
        "n_turns": len(per_turn),
        "n_prefetched_files": n_prefetched_holder[0],
        "n_kills_in_original": traj["n_kills"],
        "killed_shell_s_in_original": round(traj["killed_shell_s"], 3),
        "original_session_elapsed_s": traj["summary"]["session_elapsed_s"],
        "implied_baseline_minus_kills_s": round(
            traj["summary"]["session_elapsed_s"] - traj["killed_shell_s"], 3),
        "evict": evict_summary,
        "shell_io_aggregate": {
            "n_shell_calls": len(shell_io_log),
            "total_rchar": total_rchar,
            "total_read_bytes": total_read_bytes,
            "total_syscr": total_syscr,
            "total_shell_elapsed_s": round(total_shell_elapsed, 3),
        },
        "shell_io_per_call": shell_io_log,
        "per_turn": per_turn,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", required=True,
                    help="Path to recorded baseline session dir")
    ap.add_argument("--mode", choices=("cold", "staged"), required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hot-root", default="/dev/shm/agentstage_replay")
    args = ap.parse_args()

    session_dir = Path(args.session)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    hot_root = Path(args.hot_root)

    summary = json.loads((session_dir / "summary.json").read_text())
    task_id = summary["task"]
    # Derive bench from session path
    bench = next((p.replace("_mt","") for p in session_dir.parts
                  if p.endswith("_mt")), None)
    if bench is None:
        raise RuntimeError(f"Cannot infer bench from session path: {session_dir}")
    workload, executor_fn = load_workload_for_task(task_id, bench)

    print(f"=== Replay {session_dir.name} mode={args.mode} ===", flush=True)
    print(f"  workload: {task_id}", flush=True)
    print(f"  original elapsed: {summary['session_elapsed_s']:.1f}s, "
          f"n_turns={summary['n_turns']}, "
          f"submitted={summary['submitted']}", flush=True)

    traj = load_trajectory(session_dir)
    print(f"  parsed {len(traj['turns'])} turns, "
          f"{traj['n_kills']} shell kills (would be skipped, "
          f"{traj['killed_shell_s']:.1f}s in original)",
          flush=True)

    out_dir = out_path.parent / out_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"  replay workspace: {out_dir}", flush=True)

    result = replay(traj, workload=workload, executor_fn=executor_fn,
                     mode=args.mode, out_dir=out_dir, hot_root=hot_root)

    fidelity = (result["session_elapsed_s"]
                / result["implied_baseline_minus_kills_s"])
    result["fidelity_vs_baseline_minus_kills"] = round(fidelity, 3)

    print(f"  REPLAY ELAPSED: {result['session_elapsed_s']:.1f}s "
          f"(original-minus-kills: {result['implied_baseline_minus_kills_s']:.1f}s, "
          f"fidelity={fidelity:.2f}x)",
          flush=True)
    if args.mode == "staged":
        print(f"  prefetched: {result['n_prefetched_files']} files",
              flush=True)

    out_path.write_text(json.dumps(result, indent=2))
    print(f"  wrote: {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
