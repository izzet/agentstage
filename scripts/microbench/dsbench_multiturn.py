"""E-040 — DSBench full-agentic-loop with AgentStage integrated naturally.

The LLM agent receives the DSBench task description and a /data/<task>/
sandbox. Tools: list_dir, open_file, read_file, run_shell_command. The
agent reads the task, explores the data, writes a Python solution, and
runs it via run_shell_command. The session is timed end-to-end (first
LLM request → final assistant message) for two modes:

  --mode baseline : Stager + shim DISABLED. Subprocess runs naked.
  --mode staged   : SessionDetector + AutoRuleGenerator fire during
                    streaming → Stager.prefetch fires when rules
                    activate, copying cold files to /dev/shm. The
                    subprocess launched by run_shell_command runs with
                    LD_PRELOAD=libagentstage_shim.so so its reads redirect
                    to the hot tier.

This is the honest "whole-session" speedup measurement the user asked
for. The prefetch happens DURING the agent's reasoning turns, so the
I/O cost overlaps with reasoning slack.

Usage:
    python scripts/microbench/dsbench_multiturn.py \\
        --task santander-value-prediction-challenge \\
        --model claude-haiku-4-5 --mode staged \\
        --out outputs/dsbench_mt/santander_staged_h_1
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

# Load .env (ours + sciiobench fallback) for API keys
def _load_dotenv(p: Path) -> None:
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(Path(__file__).resolve().parents[2] / ".env")
_load_dotenv(Path("/mnt/common/iyildirim/projects/sciiobench/.env"))

from agentstage.detector.auto_rules import AutoRuleGenerator  # noqa: E402
from agentstage.detector.engine import StreamBlock  # noqa: E402
from agentstage.detector.session import SessionDetector  # noqa: E402
from agentstage.stager import DataHint, Stager  # noqa: E402
from agentstage.workloads.dsbench import (  # noqa: E402
    DSB_E2E_SLICE, DSBWorkload, load_dsbench_task,
)

REPO = Path(__file__).resolve().parents[2]
SHIM = (REPO / "src" / "agentstage" / "stager" / "shim"
        / "libagentstage_shim.so").resolve()


# ---------------------------------------------------------------------------
# Cold-cache eviction — same standard as E-030/E-039
# ---------------------------------------------------------------------------

def evict(paths: list[Path], *, verify: bool = True) -> dict:
    n_files = n_bytes = 0
    for p in paths:
        try:
            st = p.stat()
            fd = os.open(str(p), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                n_files += 1; n_bytes += st.st_size
            finally:
                os.close(fd)
        except OSError:
            pass
    os.sync()
    if not verify:
        return {"files": n_files, "bytes": n_bytes}
    try:
        from agentiobench.utils.cache import _resident_pages
    except ImportError:
        return {"files": n_files, "bytes": n_bytes}
    resident = total = 0
    for p in paths[:5]:
        try:
            r, t = _resident_pages(p)
            resident += r; total += t
        except OSError:
            continue
    return {"files": n_files, "bytes": n_bytes,
            "resident_pages_sample": resident,
            "total_pages_sample": total,
            "resident_frac_sample": resident / total if total else 0.0}


# ---------------------------------------------------------------------------
# Logical ↔ physical path resolution (DSBench-specific)
# ---------------------------------------------------------------------------

def resolve_logical(path: str, prefix_map) -> str:
    for lp, rp in prefix_map:
        if path.startswith(lp):
            return rp + path[len(lp):]
    return path


# ---------------------------------------------------------------------------
# Tool sandbox
# ---------------------------------------------------------------------------

TOOLS_SCHEMA = [
    {"name": "list_dir",
     "description": "List the immediate children of a directory under "
                    "/data/<task>/.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "open_file",
     "description": "Read the first ~4 KB of a file under /data/<task>/. "
                    "Use for previews; do NOT use to load full datasets.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "read_file",
     "description": "Alias for open_file. Reads the start of a file.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "run_shell_command",
     "description": "Run a shell command in the task workspace. Use this "
                    "to execute Python scripts you've written. The "
                    "workspace contains /data/<task>/ (input data) and "
                    "/workspace/ (writable area for scripts and outputs). "
                    "Returns stdout+stderr (truncated to 4 KB).",
     "input_schema": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}},
    {"name": "write_file",
     "description": "Write content to /workspace/<filename>. Use this to "
                    "save Python scripts before running them.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                       "content": {"type": "string"}},
                       "required": ["path", "content"]}},
]


def make_tool_executor(workload: DSBWorkload, workspace_dir: Path,
                       *, mode: str, hot_root: Path, cold_root: str):
    """Return execute_tool(name, args) -> str. Sandboxed to the task's
    /data/<task>/ logical root and /workspace/ for outputs."""
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    log_root = prefix_map[0][0].rstrip("/")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(path: str) -> tuple[str, bool]:
        """Returns (physical_path, allowed)."""
        if path.startswith("/data/") or path == "/data":
            phys = resolve_logical(path, prefix_map)
            if not phys.startswith(data_phys_root):
                # Resolution didn't catch — try the same-task fallback
                rel = path[len(log_root)+1:] if path.startswith(log_root) else ""
                phys = f"{data_phys_root}/{rel}" if rel else data_phys_root
            return phys, phys.startswith(data_phys_root)
        if path.startswith("/workspace/") or path == "/workspace":
            rel = path[len("/workspace"):].lstrip("/")
            phys = str(workspace_dir / rel) if rel else str(workspace_dir)
            return phys, True
        return path, False

    def execute(name: str, args: dict) -> str:
        if name == "list_dir":
            path = args.get("path", "")
            phys, ok = _resolve(path)
            if not ok:
                return (f"ERROR: list_dir({path!r}): outside allowed sandbox. "
                        f"Use /data/{workload.task.task_id}/ or /workspace/.")
            p = Path(phys)
            if not p.is_dir():
                return f"ERROR: list_dir: not a directory: {path}"
            entries = sorted(p.iterdir())
            display = path.rstrip("/")
            lines = [f"# Listing of {display} ({len(entries)} entries):"]
            for e in entries:
                kind = "FILE" if e.is_file() else "DIR "
                sz = e.stat().st_size if e.is_file() else 0
                full = f"{display}/{e.name}" + ("/" if e.is_dir() else "")
                lines.append(f"  {kind}  {full}  ({sz} bytes)" if e.is_file()
                             else f"  {kind}  {full}")
            return "\n".join(lines)
        elif name in ("open_file", "read_file"):
            path = args.get("path", "")
            phys, ok = _resolve(path)
            if not ok:
                return (f"ERROR: {name}({path!r}): outside sandbox.")
            p = Path(phys)
            if not p.is_file():
                return f"ERROR: {name}: file does not exist: {path}"
            size = p.stat().st_size
            with open(phys, "rb") as f:
                head = f.read(4096)
            try:
                txt = head.decode("utf-8")
                return f"# Contents of {path} (first {len(head)}/{size} bytes):\n{txt}"
            except UnicodeDecodeError:
                return f"# Binary file {path} (size {size} bytes). First 64 bytes hex:\n{head[:64].hex()}"
        elif name == "write_file":
            path = args.get("path", "")
            content = args.get("content", "")
            phys, ok = _resolve(path)
            if not ok or not phys.startswith(str(workspace_dir)):
                return f"ERROR: write_file: can only write to /workspace/"
            Path(phys).parent.mkdir(parents=True, exist_ok=True)
            Path(phys).write_text(content)
            return f"# Wrote {path} ({len(content)} bytes)"
        elif name == "run_shell_command":
            cmd = args.get("cmd", "")
            if not cmd:
                return "ERROR: run_shell_command: empty cmd"
            # Build env for subprocess. In staged mode, LD_PRELOAD points at
            # the shim and the AgentStage env vars are set so reads under
            # cold_root redirect to hot_root.
            env = os.environ.copy()
            for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
                        "UV_PROJECT_ENVIRONMENT"):
                env.pop(var, None)
            # Strip the venv prefix from PATH so `python` resolves to
            # /usr/bin/python3 (which has lightgbm/sklearn/xgb installed
            # via the user-local site-packages).
            path_parts = env.get("PATH", "/usr/bin:/bin").split(":")
            env["PATH"] = ":".join(p for p in path_parts
                                    if "/.venv/" not in p
                                    and "/agentstage/.venv" not in p)
            env["MPLBACKEND"] = "Agg"
            if mode == "staged":
                env["LD_PRELOAD"] = str(SHIM)
                env["AGENTSTAGE_HOT_ROOT"] = str(hot_root)
                env["AGENTSTAGE_COLD_ROOTS"] = cold_root
                env["AGENTSTAGE_RETRY_SPIN_MS"] = "20"
            else:
                env.pop("LD_PRELOAD", None)
                env["AGENTSTAGE_SHIM_DISABLE"] = "1"
            # Provide convenient symlinks /data/<task> and /workspace inside
            # the agent's cwd so the script can reference logical paths.
            # We create them under the workspace's cwd.
            agent_cwd = workspace_dir
            (agent_cwd / "data").mkdir(exist_ok=True)
            data_link = agent_cwd / "data" / workload.task.task_id
            if not data_link.exists():
                try:
                    data_link.symlink_to(data_phys_root)
                except FileExistsError:
                    pass
            t0 = time.monotonic()
            try:
                r = subprocess.run(["/bin/bash", "-c", cmd],
                                   cwd=str(agent_cwd), env=env,
                                   capture_output=True, text=True,
                                   timeout=180)
                rc = r.returncode
                stdout = r.stdout or ""
                stderr = r.stderr or ""
            except subprocess.TimeoutExpired as te:
                rc = -9
                stdout = (te.stdout.decode("utf-8", errors="replace")
                          if te.stdout else "")
                stderr = ((te.stderr.decode("utf-8", errors="replace")
                           if te.stderr else "")
                          + "\n[TIMEOUT after 180s — solution too slow; "
                          "use a faster baseline]")
            elapsed = time.monotonic() - t0
            out = stdout[-3000:]
            err = stderr[-1000:]
            return (f"# run_shell_command (rc={rc}, {elapsed:.2f}s):\n"
                    f"## stdout:\n{out}\n"
                    f"## stderr:\n{err}\n")
        return f"ERROR: unknown tool: {name}"

    return execute


# ---------------------------------------------------------------------------
# Anthropic streaming + agent loop with AgentStage integrated
# ---------------------------------------------------------------------------

def run_session(*, workload: DSBWorkload, model: str, mode: str,
                prompt_mode: str, out_dir: Path,
                hot_root: Path, max_turns: int = 12,
                thinking_budget: int = 4096) -> dict:
    """Run one full agentic session. Returns timing + outcome dict."""
    import anthropic

    azure_key = os.environ.get("AZURE_FOUNDRY_KEY", "")
    direct_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if azure_key:
        base = os.environ.get("AZURE_FOUNDRY_ANTHROPIC_URL",
            "https://izzet-2249-resource.openai.azure.com/anthropic/v1/messages")
        base = base.split("/v1/messages")[0]
        if not base.endswith("/anthropic"):
            base = base.rstrip("/") + "/anthropic"
        client = anthropic.Anthropic(api_key=azure_key, base_url=base)
    elif direct_key:
        client = anthropic.Anthropic(api_key=direct_key)
    else:
        raise RuntimeError("no Anthropic key")

    # AgentStage pieces — instantiated regardless of mode (so we capture
    # detector activations both ways), but only DISPATCHED in staged mode.
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    cold_root_anc = str(Path(data_phys_root).parent)  # ancestor for shim

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
        stager = Stager(
            hot_root=hot_root, cold_roots=[Path(cold_root_anc)],
            max_workers=4, capacity_bytes=64 * 1024**3,
        )

    workspace_dir = out_dir / "agent_workspace"
    execute = make_tool_executor(workload, workspace_dir,
                                  mode=mode, hot_root=hot_root,
                                  cold_root=cold_root_anc)

    turns_dir = out_dir / "turns"; turns_dir.mkdir(parents=True, exist_ok=True)

    # Build first user message
    if prompt_mode == "hinted":
        user_msg = (
            f"Task: {workload.task.task_id}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"The task data is at /data/{workload.task.task_id}/. "
            f"Specifically:\n"
            f"  - /data/{workload.task.task_id}/train.csv\n"
            f"  - /data/{workload.task.task_id}/test.csv\n"
            f"  - /data/{workload.task.task_id}/sample_submission.csv\n\n"
            f"Your working directory is /workspace/. Save the final "
            f"submission to /workspace/submission.csv. "
            f"Think step-by-step about which files you need to read."
        )
    else:  # sparse
        user_msg = (
            f"Task: {workload.task.task_id}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"Task data is under /data/. Use list_dir to discover what's "
            f"available. Working directory is /workspace/. Save the final "
            f"submission to /workspace/submission.csv."
        )

    tid = workload.task.task_id
    system_msg = (
        "You are a data-science agent solving a Kaggle-style modeling task.\n"
        "\n"
        "Workspace layout:\n"
        f"  data/{tid}/train.csv               — training data\n"
        f"  data/{tid}/test.csv                — test data\n"
        f"  data/{tid}/sample_submission.csv   — submission format\n"
        "  (these are accessible by relative path from your CWD; the\n"
        "   list_dir / open_file tools also accept the logical form\n"
        f"   /data/{tid}/<file>)\n"
        "  submission.csv                      — write your final output here\n"
        "\n"
        "Example Python script (relative paths resolved from CWD):\n"
        "  import pandas as pd\n"
        f"  train = pd.read_csv('data/{tid}/train.csv')\n"
        f"  test  = pd.read_csv('data/{tid}/test.csv')\n"
        "  # ... model ...\n"
        "  sub.to_csv('submission.csv', index=False)\n"
        "\n"
        "Python libraries available (DO NOT pip install — these are already there):\n"
        "  pandas, numpy, scipy, sklearn, lightgbm, openpyxl, matplotlib\n"
        "  Just `import lightgbm as lgb` etc — `pip install` will FAIL because\n"
        "  there is no network. If you see ModuleNotFoundError, check the\n"
        "  import name (e.g. 'sklearn' not 'scikit-learn').\n"
        "\n"
        "Workflow:\n"
        "  1. list_dir / open_file to inspect the data shape (preview only)\n"
        "  2. write_file to save your Python solution to /workspace/solution.py\n"
        "  3. run_shell_command 'python solution.py' to execute it\n"
        "  4. Iterate as needed, then say done.\n"
        "\n"
        "Solution-speed budget (IMPORTANT — shell commands time out at 180s):\n"
        "  Goal: a CORRECT submission, NOT the best accuracy.\n"
        "  Prefer FAST baselines:\n"
        "    - mean / median predictor on the target column\n"
        "    - or sklearn LinearRegression / LogisticRegression\n"
        "    - or LightGBM with n_estimators<=30, num_leaves<=31\n"
        "  Avoid: 500-tree boosts, hyperparameter search, full GBM on >1k cols.\n"
        "  If your script times out, REWRITE it simpler — do not retry."
    )

    messages: list[dict] = [{"role": "user", "content": user_msg}]

    # ── Session timing starts here ──
    session_start = time.monotonic()
    tool_use_count = 0
    files_opened: list[str] = []
    n_prefetched_total = 0
    per_turn: list[dict] = []

    for turn in range(max_turns):
        turn_dir = turns_dir / f"turn_{turn:02d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        thinking_log = (turn_dir / "thinking.jsonl").open("w")
        text_log = (turn_dir / "text.jsonl").open("w")
        tool_use_log = (turn_dir / "tool_use.jsonl").open("w")
        tool_result_log = (turn_dir / "tool_result.jsonl").open("w")
        turn_start = time.monotonic()
        stream_started_ms = turn_start * 1000

        body = {"model": model, "max_tokens": 8192, "temperature": 1.0,
                "messages": messages, "tools": TOOLS_SCHEMA,
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
                "system": system_msg}

        # Stream; on staged mode, run detector + stager live
        thinking_acc: dict[int, list[str]] = {}
        text_acc: dict[int, list[str]] = {}
        with client.messages.stream(**body) as stream:
            for event in stream:
                t_ms = time.monotonic() * 1000 - stream_started_ms
                etype = getattr(event, "type", None)
                if etype != "content_block_delta":
                    continue
                idx = event.index
                d = event.delta
                dt = getattr(d, "type", None)
                if dt == "thinking_delta":
                    piece = getattr(d, "thinking", "")
                    if piece:
                        thinking_acc.setdefault(idx, []).append(piece)
                        thinking_log.write(json.dumps({
                            "t_ms": round(t_ms, 1), "block": idx,
                            "delta": piece,
                        }) + "\n")
                elif dt == "text_delta":
                    piece = getattr(d, "text", "")
                    if piece:
                        text_acc.setdefault(idx, []).append(piece)
                        text_log.write(json.dumps({
                            "t_ms": round(t_ms, 1), "block": idx,
                            "delta": piece,
                        }) + "\n")
            final = stream.get_final_message()

        # Build assistant blocks for next-turn context
        assistant_blocks: list[dict] = []
        if final and getattr(final, "content", None):
            for b in final.content:
                btype = getattr(b, "type", None)
                if btype == "thinking":
                    assistant_blocks.append({
                        "type": "thinking",
                        "thinking": getattr(b, "thinking", ""),
                        "signature": getattr(b, "signature", "")})
                elif btype == "text":
                    assistant_blocks.append({
                        "type": "text", "text": getattr(b, "text", "")})
                elif btype == "tool_use":
                    assistant_blocks.append({
                        "type": "tool_use", "id": getattr(b, "id", ""),
                        "name": getattr(b, "name", ""),
                        "input": getattr(b, "input", {})})

        # Feed thinking+text to SessionDetector (rules fire here)
        sp_blocks: list[StreamBlock] = []
        for blk in assistant_blocks:
            if blk["type"] == "thinking":
                sp_blocks.append(StreamBlock(
                    type="thinking", t_first=0, t_stop=0,
                    text=blk["thinking"], chunks=1, turn=turn))
            elif blk["type"] == "text":
                sp_blocks.append(StreamBlock(
                    type="text", t_first=0, t_stop=0,
                    text=blk["text"], chunks=1, turn=turn))
        new_acts = session_detector.feed_turn(sp_blocks)
        fired_rules_this_turn = [a.rule_name for a in new_acts]

        # ── STAGED MODE: dispatch prefetches DURING the turn ──
        dispatched_this_turn: list[dict] = []
        if mode == "staged" and stager is not None:
            seen_phys: set[str] = set()
            for act in new_acts:
                phys_files = [
                    resolve_logical(p, prefix_map) for p in act.detected_files
                ]
                phys_files = [
                    p for p in phys_files
                    if Path(p).is_file() and p not in seen_phys
                ]
                for p in phys_files:
                    seen_phys.add(p)
                if not phys_files:
                    continue
                hint = DataHint(
                    detected_files=tuple(phys_files),
                    tier=1 if len(phys_files) <= 10 else 3,
                    fired_at_ms=act.fired_at_ms or 0.0,
                    rule_id=f"turn{turn}:{act.rule_name}",
                )
                stager.prefetch(hint)
                dispatched_this_turn.append({
                    "rule": act.rule_name,
                    "n_files": len(phys_files),
                })
                n_prefetched_total += len(phys_files)

        messages.append({"role": "assistant", "content": assistant_blocks})

        # Execute tools
        tool_results = []
        tool_result_blocks_for_detector: list[StreamBlock] = []
        any_tool = False
        for blk in assistant_blocks:
            if blk["type"] != "tool_use":
                continue
            any_tool = True
            tool_use_count += 1
            out = execute(blk["name"], blk["input"])
            tool_use_log.write(json.dumps({
                "name": blk["name"], "id": blk["id"],
                "parsed_input": blk["input"]}) + "\n")
            tool_result_log.write(json.dumps({
                "tool_use_id": blk["id"], "content": out}) + "\n")
            if blk["name"] in ("open_file", "read_file"):
                files_opened.append(blk["input"].get("path", ""))
            tool_results.append({
                "type": "tool_result", "tool_use_id": blk["id"],
                "content": out})
            tool_result_blocks_for_detector.append(StreamBlock(
                type="tool_result", t_first=0, t_stop=0,
                text=out, chunks=1, turn=turn))

        # Feed tool_results to the session detector — file listings
        # returned by list_dir reveal concrete filenames that the agent's
        # thinking didn't necessarily verbalize. This is the same
        # extension load-bearing for sparse-mode in AIOB (E-021).
        if tool_result_blocks_for_detector:
            tr_acts = session_detector.feed_tool_results(
                tool_result_blocks_for_detector)
            new_acts = list(new_acts) + list(tr_acts)
            for a in tr_acts:
                fired_rules_this_turn.append(a.rule_name)
            # Re-dispatch staged-mode prefetches for the newly-fired rules
            if mode == "staged" and stager is not None and tr_acts:
                seen_phys = set()
                for act in tr_acts:
                    phys_files = [
                        resolve_logical(p, prefix_map) for p in act.detected_files
                    ]
                    phys_files = [
                        p for p in phys_files
                        if Path(p).is_file() and p not in seen_phys
                    ]
                    for p in phys_files:
                        seen_phys.add(p)
                    if not phys_files:
                        continue
                    hint = DataHint(
                        detected_files=tuple(phys_files),
                        tier=1 if len(phys_files) <= 10 else 3,
                        fired_at_ms=act.fired_at_ms or 0.0,
                        rule_id=f"turn{turn}:{act.rule_name}:from_tool_result",
                    )
                    stager.prefetch(hint)
                    dispatched_this_turn.append({
                        "rule": act.rule_name,
                        "n_files": len(phys_files),
                        "source": "tool_result",
                    })
                    n_prefetched_total += len(phys_files)

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        turn_elapsed = time.monotonic() - turn_start
        per_turn.append({
            "turn": turn,
            "duration_s": round(turn_elapsed, 3),
            "n_tool_uses": sum(1 for b in assistant_blocks
                                if b["type"] == "tool_use"),
            "tool_names": [b["name"] for b in assistant_blocks
                            if b["type"] == "tool_use"],
            "fired_rules": fired_rules_this_turn,
            "dispatched_prefetches": dispatched_this_turn,
        })

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        if not any_tool:
            break

    session_elapsed = time.monotonic() - session_start

    # Look for a submission file to verify the agent actually finished
    submission_path = workspace_dir / "submission.csv"
    submitted = submission_path.is_file()
    submission_size = submission_path.stat().st_size if submitted else 0

    if stager is not None:
        stager.shutdown(wait=True)

    return {
        "session_elapsed_s": round(session_elapsed, 3),
        "n_turns": len(per_turn),
        "n_tool_uses": tool_use_count,
        "files_opened_logical": files_opened,
        "submitted": submitted,
        "submission_bytes": submission_size,
        "n_prefetched_files": n_prefetched_total,
        "per_turn": per_turn,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, help="DSBench task id")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--mode", choices=["baseline", "staged"], required=True)
    parser.add_argument("--prompt-mode", choices=["hinted", "sparse"],
                        default="hinted")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_dsbmt")
    parser.add_argument("--max-turns", type=int, default=15)
    args = parser.parse_args()

    if not SHIM.is_file():
        print(f"FATAL: shim missing at {SHIM}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    workload = load_dsbench_task(args.task)
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    targets = [
        Path(workload.task.train_csv),
        Path(workload.task.test_csv),
    ]
    if workload.task.sample_csv:
        targets.append(Path(workload.task.sample_csv))

    # Verified-cold eviction before session start
    ev = evict(targets)
    print(f"E-040 DSBench agentic | task={args.task} model={args.model} "
          f"mode={args.mode} prompt={args.prompt_mode}")
    print(f"  data files: {len(targets)} "
          f"({sum(p.stat().st_size for p in targets)/1024/1024:.1f} MB)")
    print(f"  resident_frac_after_evict: {ev.get('resident_frac_sample','?')}")
    print()

    result = run_session(
        workload=workload, model=args.model, mode=args.mode,
        prompt_mode=args.prompt_mode, out_dir=args.out,
        hot_root=Path(args.hot_root), max_turns=args.max_turns,
    )
    result["task"] = args.task
    result["mode"] = args.mode
    result["model"] = args.model
    result["prompt_mode"] = args.prompt_mode
    result["evict"] = ev
    result["experiment"] = "E-040"

    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"  session: {result['session_elapsed_s']}s | turns={result['n_turns']} | "
          f"tool_uses={result['n_tool_uses']} | submitted={result['submitted']}")
    if args.mode == "staged":
        print(f"  prefetched: {result['n_prefetched_files']} files")
    print(f"  wrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
