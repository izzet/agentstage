"""H12 pathful-prompt wrapper around aiob_multiturn.py.

WHY THIS EXISTS (and is NOT a patch to aiob_multiturn.py):
  A campaign is running on another host off the SAME shared /mnt/common
  checkout. Editing aiob_multiturn.py in place would be read by that
  campaign's freshly-spawned cells. This wrapper is a brand-new file the
  campaign never imports, so aiob_multiturn.py stays byte-for-byte
  unchanged and the live campaign is untouched.

WHAT IT CHANGES:
  Adds prompt_mode ∈ {"pathful", "strict_pp"} — a third user-message
  branch that forces the agent to enumerate absolute file paths before
  each tool call. Everything else (run_session, eviction, scoring,
  summary.json layout) is reused verbatim from aiob_multiturn so the H12
  test pairs these runs with the curated hinted-baseline sweeps on
  (task, model, turn, seed).

The monkeypatch on base._build_prompts is picked up by run_session's
internal callers because they reference _build_prompts via module-global
lookup at call time.

Pathful-prompt ablation: campaign plan + assertions live with the H12 eval.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make the sibling aiob_multiturn module importable regardless of CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aiob_multiturn as base  # noqa: E402

_PATHFUL_MODES = {"pathful", "strict_pp"}
_orig_build_prompts = base._build_prompts


def _build_prompts_pathful(workload, prompt_mode, shell_timeout=300):
    """Pathful branch: force literal-path enumeration. For any other mode
    delegate to the original builder so hinted/sparse stay identical."""
    if prompt_mode not in _PATHFUL_MODES:
        return _orig_build_prompts(workload, prompt_mode, shell_timeout)

    log_root = workload.prefix_map[0][0].rstrip("/")
    ds_name = log_root[len("/data/"):]
    tid = workload.task_id

    user_msg = (
        f"Task: {tid}\n\n"
        f"{workload.task.task_inst}\n\n"
        f"The dataset is accessible under the relative path "
        f"'data/{ds_name}/' from your CWD (a symlink we set up for you).\n"
        f"Use list_dir / open_file to inspect.\n\n"
        f"BEFORE EACH TOOL CALL, you MUST enumerate the absolute file "
        f"paths you intend to read in your reasoning. List the FULL "
        f"absolute paths (e.g., /data/{ds_name}/raw/<file>.csv) — not "
        f"shorthand, not glob patterns, not directory names. If you "
        f"don't know the exact paths yet, name your best guesses and "
        f"refine after the next list_dir.\n\n"
        f"DO NOT use absolute paths like /data/ or /workspace/ in your "
        f"Python scripts — those are LOGICAL labels only the list_dir / "
        f"open_file tools understand. Always use relative paths in scripts."
    )

    # Reuse the exact system prompt the hinted branch produces; the
    # system_msg does not change for pathful (per design doc).
    _, system_msg = _orig_build_prompts(workload, "hinted", shell_timeout)
    return user_msg, system_msg


base._build_prompts = _build_prompts_pathful


def main() -> int:
    """Faithful copy of base.main() with --prompt-mode extended to accept
    the pathful variants. Reuses base.* for all heavy lifting."""
    import argparse
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        choices=list(base.TASK_LOADERS.keys()))
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--mode", choices=["baseline", "staged"], required=True)
    parser.add_argument("--prompt-mode",
                        choices=["hinted", "sparse", "pathful", "strict_pp"],
                        default="pathful")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_aiobmt")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--shell-timeout", type=int, default=300,
                        help="Per-shell-command subprocess timeout in seconds")
    args = parser.parse_args()

    if not base.SHIM.is_file():
        print(f"FATAL: shim missing at {base.SHIM}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    workload = base.TASK_LOADERS[args.task]()
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")

    EVICT_CAP = 200
    targets: list[Path] = []
    for p in workload.all_workspace_paths[:EVICT_CAP]:
        phys = base.resolve_logical(p, prefix_map)
        if Path(phys).is_file():
            targets.append(Path(phys))

    ev = base.evict(targets)
    print(f"E-042 AIOB agentic (PATHFUL) | task={args.task} model={args.model} "
          f"mode={args.mode} prompt={args.prompt_mode}")
    print(f"  data root: {data_phys_root}")
    print(f"  evicted: {ev.get('files',0)} files / {ev.get('bytes',0)/1024/1024:.1f} MB "
          f"(sample, capped at {EVICT_CAP})")
    print(f"  resident_frac_after_evict: {ev.get('resident_frac_sample','?')}")
    print()

    crash_info = None
    try:
        result = base.run_session(
            workload=workload, model=args.model, mode=args.mode,
            prompt_mode=args.prompt_mode, out_dir=args.out,
            hot_root=Path(args.hot_root), max_turns=args.max_turns,
            shell_timeout=args.shell_timeout,
        )
    except Exception as e:
        import traceback
        crash_info = {
            "exc_type": type(e).__name__,
            "exc_msg": str(e)[:1000],
            "traceback": traceback.format_exc()[:3000],
        }
        result = {
            "session_elapsed_s": None, "n_turns": None, "n_tool_uses": None,
            "files_opened_logical": [], "submitted": False,
            "submission_bytes": 0, "n_prefetched_files": 0, "per_turn": [],
            "crash": crash_info,
        }
    result["task"] = args.task
    result["mode"] = args.mode
    result["model"] = args.model
    result["prompt_mode"] = args.prompt_mode
    result["evict"] = ev
    result["experiment"] = "E-042"

    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    if result.get("session_elapsed_s") is not None:
        print(f"  session: {result['session_elapsed_s']}s | "
              f"turns={result['n_turns']} | tool_uses={result['n_tool_uses']} | "
              f"outputs={result.get('n_workspace_outputs', 0)}")
        if args.mode == "staged":
            print(f"  prefetched: {result['n_prefetched_files']} files")
    else:
        print(f"  CRASHED: {crash_info.get('exc_type')}")
    print(f"  wrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
