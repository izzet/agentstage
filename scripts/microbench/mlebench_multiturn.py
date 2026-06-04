"""E-041 — MLE-bench full agentic loop with AgentStage integrated.

Mirrors dsbench_multiturn.py exactly but for MLE-bench competitions.
The LLM agent receives the competition description.md (verbatim from
MLE-bench's `prepare`) and a /data/<competition_id>/ sandbox.

Tools: list_dir, open_file, read_file, write_file, run_shell_command.
The agent reads the task, explores the data, writes a Python solution,
and runs it via run_shell_command. The session is timed end-to-end
(first LLM request → final assistant message) for two modes:

  --mode baseline : Stager + shim DISABLED. Subprocess runs naked.
  --mode staged   : SessionDetector + AutoRuleGenerator fire during
                    streaming → Stager.prefetch fires when rules
                    activate, copying cold files to /dev/shm. The
                    subprocess launched by run_shell_command runs with
                    LD_PRELOAD=libagentstage_shim.so so its reads
                    redirect to the hot tier.

NO benchmark-specific path tweaks — agent uses natural relative paths
(`data/<competition_id>/<file>`); shim follows symlinks via realpath().
The MLE-bench description.md is passed through unedited.

Usage:
    python scripts/microbench/mlebench_multiturn.py \\
        --task histopathologic-cancer-detection \\
        --model claude-haiku-4-5 --mode staged \\
        --out outputs/mlebench_mt/histo_staged_r1
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
from agentstage.workloads.mlebench import (  # noqa: E402
    MLEWorkload, load_mle_competition, load_mle_competition_dispatch,
)

REPO = Path(__file__).resolve().parents[2]
SHIM = (REPO / "src" / "agentstage" / "stager" / "shim"
        / "libagentstage_shim.so").resolve()


def evict(paths: list[Path], *, verify: bool = True) -> dict:
    """Cold-cache methodology, same as E-030/E-039/E-040."""
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


def resolve_logical(path: str, prefix_map) -> str:
    for lp, rp in prefix_map:
        if path.startswith(lp):
            return rp + path[len(lp):]
    return path


TOOLS_SCHEMA = [
    {"name": "list_dir",
     "description": "List the immediate children of a directory under "
                    "/data/<competition>/.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "open_file",
     "description": "Read the first ~4 KB of a file under /data/<competition>/. "
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
                    "workspace contains /data/<competition>/ (input data) and "
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


def _aggregate_io(shell_io_log: "list[dict]") -> dict:
    """Sum per-call /proc/[pid]/io counters into per-session totals."""
    keys = ("rchar", "read_bytes", "wchar", "write_bytes", "syscr", "syscw")
    agg = {k: 0 for k in keys}
    n = 0
    elapsed_total_s = 0.0
    for entry in shell_io_log:
        n += 1
        elapsed_total_s += entry.get("elapsed_s", 0.0)
        for k in keys:
            agg[k] += entry.get(k, 0) or 0
    agg["n_shell_calls"] = n
    agg["shell_elapsed_total_s"] = round(elapsed_total_s, 3)
    return agg


def make_tool_executor(workload: MLEWorkload, workspace_dir: Path,
                       *, mode: str, hot_root: Path, cold_root: str,
                       shell_timeout: int = 180,
                       io_log: "list[dict] | None" = None):
    import threading
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    log_root = prefix_map[0][0].rstrip("/")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    def _capture_proc_io(pid: int) -> dict:
        out: dict = {}
        try:
            with open(f"/proc/{pid}/io") as f:
                for line in f:
                    k, _, v = line.partition(":")
                    try: out[k.strip()] = int(v.strip())
                    except ValueError: pass
        except (FileNotFoundError, ProcessLookupError, PermissionError):
            pass
        return out

    # Sorted prefix list, longest-first, so multi-entry prefix_maps
    # resolve correctly even if one prefix is a substring of another.
    _sorted_prefixes = sorted(prefix_map, key=lambda x: -len(x[0]))

    def _resolve(path: str) -> tuple[str, bool]:
        # Auto-fix: treat relative paths (no leading /) as workspace-relative
        # so agents don't burn an LLM turn learning the path convention.
        if path and not path.startswith("/"):
            path = "/workspace/" + path
        if path.startswith("/data/") or path == "/data":
            for log_pre, real_pre in _sorted_prefixes:
                log_root_p = log_pre.rstrip("/")
                real_root_p = real_pre.rstrip("/")
                if path == log_root_p:
                    return real_root_p, True
                if path.startswith(log_pre):
                    rel = path[len(log_pre):]
                    phys = f"{real_root_p}/{rel}" if rel else real_root_p
                    return phys, True
            # No prefix match — fall back to first prefix's real root.
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
                        f"Use /data/{workload.task.competition_id}/ or /workspace/.")
            p = Path(phys)
            if not p.is_dir():
                return f"ERROR: list_dir: not a directory: {path}"
            # Cap output at 200 entries to avoid blowing up the LLM
            # context (dogs-vs-cats train/ has 22.5k files; an uncapped
            # listing returns ~2 MB of text which exceeds the model's
            # context window on the next request and crashes the runner).
            MAX_ENTRIES = 200
            entries = sorted(p.iterdir())
            n_total = len(entries)
            shown = entries[:MAX_ENTRIES]
            display = path.rstrip("/")
            header = (f"# Listing of {display} ({n_total} entries"
                      + (f", showing first {MAX_ENTRIES}):" if n_total > MAX_ENTRIES else "):"))
            lines = [header]
            for e in shown:
                kind = "FILE" if e.is_file() else "DIR "
                sz = e.stat().st_size if e.is_file() else 0
                full = f"{display}/{e.name}" + ("/" if e.is_dir() else "")
                lines.append(f"  {kind}  {full}  ({sz} bytes)" if e.is_file()
                             else f"  {kind}  {full}")
            if n_total > MAX_ENTRIES:
                lines.append(f"  ... ({n_total - MAX_ENTRIES} more entries elided)")
            return "\n".join(lines)
        elif name in ("open_file", "read_file"):
            path = args.get("path", "")
            phys, ok = _resolve(path)
            if not ok:
                return f"ERROR: {name}({path!r}): outside sandbox."
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
            env = os.environ.copy()
            for var in ("VIRTUAL_ENV", "PYTHONPATH", "PYTHONHOME",
                        "UV_PROJECT_ENVIRONMENT"):
                env.pop(var, None)
            path_parts = env.get("PATH", "/usr/bin:/bin").split(":")
            env["PATH"] = ":".join(p for p in path_parts
                                    if "/.venv/" not in p
                                    and "/agentstage/.venv" not in p)
            env["MPLBACKEND"] = "Agg"
            if mode == "staged":
                env["LD_PRELOAD"] = str(SHIM)
                env["AGENTSTAGE_HOT_ROOT"] = str(hot_root)
                env["AGENTSTAGE_COLD_ROOTS"] = cold_root
                env["AGENTSTAGE_RETRY_SPIN_MS"] = "0"
                # Tiered hot: propagate overflow root if set.
                _ovr = os.environ.get("AGENTSTAGE_HOT_OVERFLOW")
                if _ovr:
                    env["AGENTSTAGE_HOT_OVERFLOW"] = _ovr
            else:
                env.pop("LD_PRELOAD", None)
                env["AGENTSTAGE_SHIM_DISABLE"] = "1"
            agent_cwd = workspace_dir
            (agent_cwd / "data").mkdir(exist_ok=True)
            data_link = agent_cwd / "data" / workload.task.competition_id
            if not data_link.exists():
                try:
                    data_link.symlink_to(data_phys_root)
                except FileExistsError:
                    pass
            t0 = time.monotonic()
            proc = subprocess.Popen(
                ["/bin/bash", "-c", cmd],
                cwd=str(agent_cwd), env=env,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True,
            )
            last_io: dict = {}
            stop = threading.Event()

            def _poll_io():
                while not stop.is_set():
                    snapshot = _capture_proc_io(proc.pid)
                    if snapshot:
                        last_io.clear()
                        last_io.update(snapshot)
                    time.sleep(0.05)
            poller = threading.Thread(target=_poll_io, daemon=True)
            poller.start()
            try:
                stdout, stderr = proc.communicate(timeout=shell_timeout)
                rc = proc.returncode
            except subprocess.TimeoutExpired:
                proc.kill()
                stdout, stderr = proc.communicate()
                rc = -9
                stderr = (stderr or "") + (
                    f"\n[TIMEOUT after {shell_timeout}s — solution too slow; "
                    "use a faster baseline]"
                )
            finally:
                stop.set()
                poller.join(timeout=1)
            elapsed = time.monotonic() - t0
            if io_log is not None:
                io_log.append({
                    "cmd": cmd[:200],
                    "elapsed_s": round(elapsed, 3),
                    "rc": rc,
                    "rchar": last_io.get("rchar", 0),
                    "read_bytes": last_io.get("read_bytes", 0),
                    "wchar": last_io.get("wchar", 0),
                    "write_bytes": last_io.get("write_bytes", 0),
                    "syscr": last_io.get("syscr", 0),
                    "syscw": last_io.get("syscw", 0),
                })
            out = (stdout or "")[-3000:]
            err = (stderr or "")[-1000:]
            return (f"# run_shell_command (rc={rc}, {elapsed:.2f}s):\n"
                    f"## stdout:\n{out}\n"
                    f"## stderr:\n{err}\n")
        return f"ERROR: unknown tool: {name}"

    return execute


def run_session(*, workload: MLEWorkload, model: str, mode: str,
                prompt_mode: str, out_dir: Path,
                hot_root: Path, max_turns: int = 12,
                thinking_budget: int = 4096,
                shell_timeout: int = 180) -> dict:
    """Dispatch to provider-specific session runner based on model name."""
    if model.lower().startswith("gemini"):
        return _run_session_gemini(
            workload=workload, model=model, mode=mode,
            prompt_mode=prompt_mode, out_dir=out_dir,
            hot_root=hot_root, max_turns=max_turns,
            shell_timeout=shell_timeout,
        )
    if model.lower().startswith("qwen") or model.lower().startswith("oss-") \
            or "/" in model:
        return _run_session_oss(
            workload=workload, model=model, mode=mode,
            prompt_mode=prompt_mode, out_dir=out_dir,
            hot_root=hot_root, max_turns=max_turns,
            shell_timeout=shell_timeout,
        )
    return _run_session_anthropic(
        workload=workload, model=model, mode=mode,
        prompt_mode=prompt_mode, out_dir=out_dir,
        hot_root=hot_root, max_turns=max_turns,
        thinking_budget=thinking_budget,
        shell_timeout=shell_timeout,
    )


def _run_session_anthropic(*, workload: MLEWorkload, model: str, mode: str,
                            prompt_mode: str, out_dir: Path,
                            hot_root: Path, max_turns: int = 12,
                            thinking_budget: int = 4096,
                            shell_timeout: int = 180) -> dict:
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

    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    cold_root_anc = str(Path(data_phys_root).parent.parent.parent)
    # data_phys_root = .../<comp>/prepared/public/  → cold_root_anc =
    # .../mlebench-data, the directory containing all competitions

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
        _ovr = os.environ.get("AGENTSTAGE_HOT_OVERFLOW") or None
        _cap_gb = os.environ.get("AGENTSTAGE_HOT_PRIMARY_CAP_GB")
        _cap = int(float(_cap_gb) * 1024**3) if _cap_gb else None
        stager = Stager(
            hot_root=hot_root, cold_roots=[Path(cold_root_anc)],
            max_workers=4, capacity_bytes=64 * 1024**3,
            hot_overflow_root=_ovr,
            hot_primary_capacity_bytes=_cap,
        )

    workspace_dir = out_dir / "agent_workspace"
    shell_io_log: list[dict] = []
    execute = make_tool_executor(workload, workspace_dir,
                                  mode=mode, hot_root=hot_root,
                                  cold_root=cold_root_anc,
                                  shell_timeout=shell_timeout,
                                  io_log=shell_io_log)

    turns_dir = out_dir / "turns"; turns_dir.mkdir(parents=True, exist_ok=True)
    cid = workload.task.competition_id

    if prompt_mode == "hinted":
        # Enumerate the public-dir contents in the user msg so the agent
        # sees what's available without needing a list_dir round trip
        files_listing = "\n".join(
            f"  /data/{cid}/{Path(p[1]).parent.relative_to(workload.task.public_dir)}/{Path(p[1]).name}"
            if False else  # placeholder; we just list workspace_prior values
            f"  /data/{cid}/{Path(p).name}"
            for k, paths in workload.workspace_prior.items()
            if k not in ("all_files", "output_submission")
            for p in paths[:3]
        )
        user_msg = (
            f"Task: {cid}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"Public competition files visible to you:\n{files_listing}\n\n"
            f"Working directory is /workspace/. Save the final "
            f"submission to /workspace/submission.csv. "
            f"Think step-by-step about which files you need to read."
        )
    else:  # sparse
        user_msg = (
            f"Task: {cid}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"Task data is under /data/. Use list_dir to discover what's "
            f"available. Working directory is /workspace/. Save the final "
            f"submission to /workspace/submission.csv."
        )

    system_msg = (
        "You are a Kaggle competition agent. You are solving an "
        "MLE-bench task, where the goal is to produce a valid "
        "submission.csv in the format described by the competition.\n"
        "\n"
        "You are an agent: keep going until the task is fully complete, before\n"
        "ending your turn. The task is complete ONLY when 'submission.csv'\n"
        "exists in your CWD and is valid (non-empty, columns match the\n"
        "competition's sample_submission, plausible values). Do NOT stop after\n"
        "inspecting the data; produce the submission file.\n"
        "\n"
        "If unsure about file structure or contents, use your tools to read;\n"
        "do NOT guess. Be skeptical of nan / empty / 0-row intermediate\n"
        "results.\n"
        "\n"
        "Workspace layout:\n"
        f"  data/{cid}/                  — competition files (read-only)\n"
        "  submission.csv                — write your final output here (CWD-relative)\n"
        "  (the list_dir / open_file tools also accept the logical form\n"
        f"   /data/{cid}/<file>)\n"
        "\n"
        "Example Python script (relative paths resolved from CWD):\n"
        "  import pandas as pd\n"
        f"  train = pd.read_csv('data/{cid}/train.csv')\n"
        "  # ... model ...\n"
        "  sub.to_csv('submission.csv', index=False)\n"
        "\n"
        "Python libraries available (DO NOT pip install — these are already there):\n"
        "  pandas, numpy, scipy, sklearn, lightgbm, openpyxl, matplotlib, PIL\n"
        "  For zip files: zipfile, io. For images: PIL.Image, numpy.\n"
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
        "    - constant predictor (mean / mode) — always valid first attempt\n"
        "    - sklearn LinearRegression / LogisticRegression\n"
        "    - LightGBM with n_estimators<=30, num_leaves<=31\n"
        "    - For images: small CNN with 1 epoch on a subset, or simple\n"
        "      pixel-feature + logistic regression\n"
        "  Avoid: full image training, n_estimators>50, hyperparameter search.\n"
        "  If your script times out, REWRITE it simpler — do not retry."
    )

    messages: list[dict] = [{"role": "user", "content": user_msg}]
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

        # Feed tool_results to session detector (file listings reveal
        # concrete filenames; same load-bearing extension as E-021 / E-040)
        if tool_result_blocks_for_detector:
            tr_acts = session_detector.feed_tool_results(
                tool_result_blocks_for_detector)
            for a in tr_acts:
                fired_rules_this_turn.append(a.rule_name)
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
        "shell_io": _aggregate_io(shell_io_log),
        "shell_io_per_call": shell_io_log,
    }


def _run_session_gemini(*, workload: MLEWorkload, model: str, mode: str,
                         prompt_mode: str, out_dir: Path,
                         hot_root: Path, max_turns: int = 12,
                         shell_timeout: int = 180) -> dict:
    """Gemini-streaming variant. Mirrors _run_session_anthropic detector +
    stager + tool-exec logic but uses the google.genai SDK."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    cold_root_anc = str(Path(data_phys_root).parent.parent.parent)

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
        _ovr = os.environ.get("AGENTSTAGE_HOT_OVERFLOW") or None
        _cap_gb = os.environ.get("AGENTSTAGE_HOT_PRIMARY_CAP_GB")
        _cap = int(float(_cap_gb) * 1024**3) if _cap_gb else None
        stager = Stager(
            hot_root=hot_root, cold_roots=[Path(cold_root_anc)],
            max_workers=4, capacity_bytes=64 * 1024**3,
            hot_overflow_root=_ovr,
            hot_primary_capacity_bytes=_cap,
        )

    workspace_dir = out_dir / "agent_workspace"
    shell_io_log: list[dict] = []
    execute = make_tool_executor(workload, workspace_dir,
                                  mode=mode, hot_root=hot_root,
                                  cold_root=cold_root_anc,
                                  shell_timeout=shell_timeout,
                                  io_log=shell_io_log)

    turns_dir = out_dir / "turns"; turns_dir.mkdir(parents=True, exist_ok=True)
    cid = workload.task.competition_id

    if prompt_mode == "hinted":
        files_listing = "\n".join(
            f"  /data/{cid}/{Path(p).name}"
            for k, paths in workload.workspace_prior.items()
            if k not in ("all_files", "output_submission")
            for p in paths[:3]
        )
        user_msg = (
            f"Task: {cid}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"Public competition files visible to you:\n{files_listing}\n\n"
            f"Working directory is /workspace/. Save the final "
            f"submission to /workspace/submission.csv."
        )
    else:
        user_msg = (
            f"Task: {cid}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"Task data is under /data/. Use list_dir to discover what's available. "
            f"Save submission to /workspace/submission.csv."
        )

    system_msg = (
        "You are a Kaggle competition agent solving an MLE-bench task.\n"
        "You are an agent: keep going until the task is fully complete. The task is "
        "complete ONLY when 'submission.csv' exists and is valid (non-empty, columns "
        "match the sample_submission, plausible values). Do NOT stop after exploring "
        "the data; produce the submission. If unsure about file contents, use tools "
        "to read; do NOT guess. Be skeptical of nan / empty / 0-row intermediate "
        "results.\n"
        f"Workspace layout: data/{cid}/ for inputs (read-only); CWD is /workspace/. "
        f"Use list_dir/open_file to inspect, write_file to save solution.py, "
        f"run_shell_command 'python solution.py' to execute. Shell commands "
        f"time out at 180s — prefer fast baselines (constant predictor, "
        f"Ridge, small LightGBM). Submission goes to /workspace/submission.csv."
    )

    # Gemini function declarations from TOOLS_SCHEMA
    fn_decls = []
    for ts in TOOLS_SCHEMA:
        props = {}
        for pname, pspec in ts["input_schema"]["properties"].items():
            props[pname] = {"type": "STRING"}
        fn_decls.append({
            "name": ts["name"],
            "description": ts["description"],
            "parameters": {
                "type": "OBJECT",
                "properties": props,
                "required": ts["input_schema"].get("required", []),
            },
        })
    tools = [types.Tool(function_declarations=fn_decls)]

    contents: list = [{"role": "user", "parts": [{"text": user_msg}]}]
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

        cfg = types.GenerateContentConfig(
            temperature=1.0,
            tools=tools,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            system_instruction=system_msg or None,
        )
        # Stream; collect parts into a unified `assistant_blocks` shape
        # (same format as Anthropic path) so the post-stream logic can be
        # shared via copy.
        assistant_blocks: list[dict] = []
        fn_calls: list[dict] = []
        for chunk in client.models.generate_content_stream(
            model=model, contents=contents, config=cfg
        ):
            t_ms = time.monotonic() * 1000 - stream_started_ms
            if not chunk.candidates:
                continue
            cand = chunk.candidates[0]
            if not cand.content or not cand.content.parts:
                continue
            for part in cand.content.parts:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    thinking_log.write(json.dumps({
                        "t_ms": round(t_ms, 1), "block": 0,
                        "delta": part.text,
                    }) + "\n")
                    assistant_blocks.append({
                        "type": "thinking", "thinking": part.text,
                    })
                elif getattr(part, "text", None):
                    text_log.write(json.dumps({
                        "t_ms": round(t_ms, 1), "block": 0,
                        "delta": part.text,
                    }) + "\n")
                    assistant_blocks.append({
                        "type": "text", "text": part.text,
                    })
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    name = getattr(fc, "name", "") or ""
                    args = dict(getattr(fc, "args", {}) or {})
                    sid = f"gem_{turn}_{len(fn_calls)}"
                    fn_calls.append({"id": sid, "name": name, "args": args})
                    assistant_blocks.append({
                        "type": "tool_use", "id": sid, "name": name,
                        "input": args,
                    })

        # Append model turn into contents for next request
        contents.append({"role": "model", "parts": [
            {"text": b.get("text") or b.get("thinking", "")}
            if b["type"] in ("text", "thinking")
            else {"function_call": {"name": b["name"], "args": b["input"]}}
            for b in assistant_blocks
        ]})

        # Feed thinking + text to detector
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

        dispatched_this_turn: list[dict] = []
        if mode == "staged" and stager is not None:
            seen_phys: set[str] = set()
            for act in new_acts:
                phys_files = [resolve_logical(p, prefix_map)
                              for p in act.detected_files]
                phys_files = [p for p in phys_files
                              if Path(p).is_file() and p not in seen_phys]
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
                    "rule": act.rule_name, "n_files": len(phys_files),
                })
                n_prefetched_total += len(phys_files)

        # Execute tools, write logs, append responses to contents
        responses = []
        tool_result_blocks_for_detector: list[StreamBlock] = []
        any_tool = False
        for fc in fn_calls:
            any_tool = True
            tool_use_count += 1
            out = execute(fc["name"], fc["args"])
            tool_use_log.write(json.dumps({
                "name": fc["name"], "id": fc["id"],
                "parsed_input": fc["args"]}) + "\n")
            tool_result_log.write(json.dumps({
                "tool_use_id": fc["id"], "content": out}) + "\n")
            if fc["name"] in ("open_file", "read_file"):
                files_opened.append(fc["args"].get("path", ""))
            responses.append({"function_response": {
                "name": fc["name"], "response": {"output": out}}})
            tool_result_blocks_for_detector.append(StreamBlock(
                type="tool_result", t_first=0, t_stop=0,
                text=out, chunks=1, turn=turn))

        # Feed tool_results to detector (file listings reveal concrete
        # filenames — E-021/E-040 extension)
        if tool_result_blocks_for_detector:
            tr_acts = session_detector.feed_tool_results(
                tool_result_blocks_for_detector)
            for a in tr_acts:
                fired_rules_this_turn.append(a.rule_name)
            if mode == "staged" and stager is not None and tr_acts:
                seen_phys = set()
                for act in tr_acts:
                    phys_files = [resolve_logical(p, prefix_map)
                                  for p in act.detected_files]
                    phys_files = [p for p in phys_files
                                  if Path(p).is_file() and p not in seen_phys]
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
                        "rule": act.rule_name, "n_files": len(phys_files),
                        "source": "tool_result",
                    })
                    n_prefetched_total += len(phys_files)

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        turn_elapsed = time.monotonic() - turn_start
        per_turn.append({
            "turn": turn, "duration_s": round(turn_elapsed, 3),
            "n_tool_uses": len(fn_calls),
            "tool_names": [fc["name"] for fc in fn_calls],
            "fired_rules": fired_rules_this_turn,
            "dispatched_prefetches": dispatched_this_turn,
        })

        if responses:
            contents.append({"role": "function", "parts": responses})
        if not any_tool:
            break

    session_elapsed = time.monotonic() - session_start
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
        "shell_io": _aggregate_io(shell_io_log),
        "shell_io_per_call": shell_io_log,
    }


# ============================================================================
# OSS (vLLM / OpenAI-compatible) dispatcher
# ============================================================================

def _run_session_oss(*, workload: MLEWorkload, model: str, mode: str,
                     prompt_mode: str, out_dir: Path,
                     hot_root: Path, max_turns: int = 12,
                     shell_timeout: int = 180) -> dict:
    from openai import OpenAI
    import httpx

    base_url = os.environ.get("OSS_MODEL_BASE_URL", "http://localhost:8002/v1")
    api_key = os.environ.get("OSS_MODEL_API_KEY", "EMPTY")
    # Per-phase httpx timeouts kill stuck Qwen streams (vLLM stall) fast.
    client = OpenAI(
        base_url=base_url, api_key=api_key,
        timeout=httpx.Timeout(300.0, connect=10.0, read=60.0, write=10.0),
    )

    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    cold_root_anc = str(Path(data_phys_root).parent.parent)

    auto_rs = AutoRuleGenerator(
        workload_id=workload.task.competition_id,
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
        _ovr = os.environ.get("AGENTSTAGE_HOT_OVERFLOW") or None
        _cap_gb = os.environ.get("AGENTSTAGE_HOT_PRIMARY_CAP_GB")
        _cap = int(float(_cap_gb) * 1024**3) if _cap_gb else None
        stager = Stager(
            hot_root=hot_root, cold_roots=[Path(cold_root_anc)],
            max_workers=4, capacity_bytes=64 * 1024**3,
            hot_overflow_root=_ovr,
            hot_primary_capacity_bytes=_cap,
        )

    workspace_dir = out_dir / "agent_workspace"
    shell_io_log: list[dict] = []
    execute = make_tool_executor(workload, workspace_dir,
                                  mode=mode, hot_root=hot_root,
                                  cold_root=cold_root_anc,
                                  shell_timeout=shell_timeout,
                                  io_log=shell_io_log)
    turns_dir = out_dir / "turns"; turns_dir.mkdir(parents=True, exist_ok=True)

    cid = workload.task.competition_id
    paths = list(workload.workspace_prior.get("all_files", ()))[:3] \
        if "all_files" in workload.workspace_prior else []
    files_listing = "\n".join(f"  - {p}" for p in paths) if paths else \
        f"  (use list_dir on /data/{cid}/)"
    if prompt_mode == "hinted":
        user_msg = (
            f"Task: {cid}\n\n{workload.task.task_inst}\n\n"
            f"Public competition files visible to you:\n{files_listing}\n\n"
            f"Working directory is /workspace/. Save the final "
            f"submission to /workspace/submission.csv. "
            f"Think step-by-step about which files you need to read."
        )
    else:
        user_msg = (
            f"Task: {cid}\n\n{workload.task.task_inst}\n\n"
            f"Task data is under /data/. Use list_dir to discover what's "
            f"available. Save the final submission to /workspace/submission.csv."
        )
    system_msg = (
        "You are a Kaggle competition agent solving an MLE-bench task.\n"
        "\n"
        "You are an agent: keep going until the task is fully complete, before\n"
        "ending your turn. The task is complete ONLY when 'submission.csv'\n"
        "exists in your CWD and is valid (non-empty, columns match the\n"
        "competition's sample_submission, plausible values). Do NOT stop after\n"
        "inspecting the data; produce the submission. If unsure about file\n"
        "contents, use tools to read; do NOT guess. Be skeptical of nan /\n"
        "empty / 0-row intermediate results.\n"
        "\n"
        "Path conventions (READ CAREFULLY):\n"
        f"  • Tools (list_dir, open_file, write_file) take ABSOLUTE LOGICAL paths:\n"
        f"      /data/{cid}/<file>     (input data; with leading slash)\n"
        "      /workspace/<file>     (your scratch + output area)\n"
        f"  • Inside Python scripts, refer to data via the CWD-RELATIVE form\n"
        f"      data/{cid}/<file>     (NO leading slash). Submission goes to\n"
        "      'submission.csv' (relative). There is a 'data/' symlink in your CWD.\n"
        f"  • The shell runs `python solution.py` directly — DO NOT prefix with\n"
        "      'cd /workspace'. /workspace is logical, not a real shell path.\n"
        "\n"
        "Example Python script (saved via write_file to /workspace/solution.py):\n"
        "  import pandas as pd\n"
        f"  train = pd.read_csv('data/{cid}/train.csv')\n"
        "  # ... model ...\n"
        "  sub.to_csv('submission.csv', index=False)\n"
        "\n"
        "Python libraries available (no pip install — no network):\n"
        "  pandas, numpy, scipy, sklearn, lightgbm, openpyxl, matplotlib,\n"
        "  PIL, zipfile, io.\n"
        "\n"
        "Workflow:\n"
        f"  1. list_dir('/data/{cid}/')\n"
        f"  2. open_file('/data/{cid}/<sample>')\n"
        "  3. write_file('/workspace/solution.py', '<your full script>')\n"
        "  4. run_shell_command('python solution.py')\n"
        "  5. Iterate, then stop calling tools when done.\n"
        f"\nSolution-speed budget (shell commands time out at {shell_timeout}s):\n"
        "  Prefer FAST baselines (constant predictor, LightGBM<=30 trees,\n"
        "  small CNN with 1 epoch). Avoid full image training or hyperparam search.\n"
        "\nThinking style: mention specific filenames you plan to read\n"
        "  (train.zip, test.zip, sample_submission.csv) — the harness uses\n"
        "  these mentions to pre-cache files for faster reads."
    )

    tools = [
        {"type": "function",
         "function": {"name": ts["name"],
                      "description": ts["description"],
                      "parameters": ts["input_schema"]}}
        for ts in TOOLS_SCHEMA
    ]

    messages: list[dict] = []
    if system_msg:
        messages.append({"role": "system", "content": system_msg})
    messages.append({"role": "user", "content": user_msg})

    session_start = time.monotonic()
    tool_use_count = 0
    files_opened: list[str] = []
    n_prefetched_total = 0
    seen_phys: set[str] = set()
    per_turn: list[dict] = []
    submission_size = 0
    submitted = False

    for turn in range(max_turns):
        turn_dir = turns_dir / f"turn_{turn:02d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        thinking_log = (turn_dir / "thinking.jsonl").open("w")
        text_log = (turn_dir / "text.jsonl").open("w")
        tool_use_log = (turn_dir / "tool_use.jsonl").open("w")
        tool_result_log = (turn_dir / "tool_result.jsonl").open("w")
        turn_start = time.monotonic()
        stream_started_ms = turn_start * 1000

        pending_calls: dict[int, dict] = {}
        thinking_buf: list[str] = []
        text_buf: list[str] = []

        # Qwen3 thinking-mode officially recommended sampling params.
        stream = client.chat.completions.create(
            model=model, messages=messages, tools=tools,
            tool_choice="auto", stream=True,
            max_tokens=8192,
            temperature=0.6,
            top_p=0.95,
            presence_penalty=1.5,
            extra_body={"top_k": 20},
        )

        for chunk in stream:
            t_ms = time.monotonic() * 1000 - stream_started_ms
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            reasoning = (getattr(delta, "reasoning", None)
                         or getattr(delta, "reasoning_content", None))
            if reasoning:
                thinking_log.write(json.dumps({
                    "t_ms": round(t_ms, 1), "block": 0,
                    "delta": reasoning}) + "\n")
                thinking_buf.append(reasoning)
            content = getattr(delta, "content", None)
            if content:
                text_log.write(json.dumps({
                    "t_ms": round(t_ms, 1), "block": 0,
                    "delta": content}) + "\n")
                text_buf.append(content)
            tcs = getattr(delta, "tool_calls", None) or []
            for tc in tcs:
                idx = getattr(tc, "index", 0)
                slot = pending_calls.setdefault(idx, {
                    "id": "", "name": "", "arguments": ""})
                tc_id = getattr(tc, "id", None)
                if tc_id:
                    slot["id"] = tc_id
                fn = getattr(tc, "function", None)
                if fn is not None:
                    if getattr(fn, "name", None):
                        slot["name"] = fn.name
                    if getattr(fn, "arguments", None):
                        slot["arguments"] += fn.arguments

        fn_calls: list[dict] = []
        for idx in sorted(pending_calls):
            slot = pending_calls[idx]
            try:
                args = json.loads(slot["arguments"] or "{}")
            except json.JSONDecodeError:
                args = {}
            fn_calls.append({
                "id": slot["id"] or f"oss_{turn}_{idx}",
                "name": slot["name"], "args": args})

        asst_msg: dict = {"role": "assistant"}
        asst_msg["content"] = "".join(text_buf) if text_buf else None
        if fn_calls:
            asst_msg["tool_calls"] = [{
                "id": fc["id"], "type": "function",
                "function": {"name": fc["name"],
                              "arguments": json.dumps(fc["args"])}}
                for fc in fn_calls]
        messages.append(asst_msg)

        sp_blocks: list[StreamBlock] = []
        if thinking_buf:
            sp_blocks.append(StreamBlock(
                type="thinking", t_first=0, t_stop=0,
                text="".join(thinking_buf), chunks=1, turn=turn))
        if text_buf:
            sp_blocks.append(StreamBlock(
                type="text", t_first=0, t_stop=0,
                text="".join(text_buf), chunks=1, turn=turn))
        new_acts = session_detector.feed_turn(sp_blocks)
        fired_rules_this_turn = [a.rule_name for a in new_acts]
        dispatched_this_turn: list[dict] = []
        if mode == "staged":
            for act in new_acts:
                phys_files = [
                    resolve_logical(p, prefix_map) for p in act.detected_files
                ]
                phys_files = [p for p in phys_files
                              if Path(p).is_file() and p not in seen_phys]
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
                    "rule": act.rule_name, "n_files": len(phys_files)})
                n_prefetched_total += len(phys_files)

        any_tool = False
        tool_result_blocks_for_detector: list[StreamBlock] = []
        for fc in fn_calls:
            any_tool = True
            tool_use_count += 1
            out = execute(fc["name"], fc["args"])
            tool_use_log.write(json.dumps({
                "name": fc["name"], "id": fc["id"],
                "parsed_input": fc["args"]}) + "\n")
            tool_result_log.write(json.dumps({
                "tool_use_id": fc["id"], "content": out}) + "\n")
            if fc["name"] in ("open_file", "read_file"):
                files_opened.append(fc["args"].get("path", ""))
            messages.append({
                "role": "tool", "tool_call_id": fc["id"],
                "name": fc["name"], "content": out})
            tool_result_blocks_for_detector.append(StreamBlock(
                type="tool_result", t_first=0, t_stop=0,
                text=out, chunks=1, turn=turn))

        if tool_result_blocks_for_detector:
            tr_acts = session_detector.feed_tool_results(
                tool_result_blocks_for_detector)
            for a in tr_acts:
                fired_rules_this_turn.append(a.rule_name)
            if mode == "staged" and tr_acts:
                for act in tr_acts:
                    phys_files = [
                        resolve_logical(p, prefix_map) for p in act.detected_files
                    ]
                    phys_files = [p for p in phys_files
                                  if Path(p).is_file() and p not in seen_phys]
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
                        "rule": act.rule_name, "n_files": len(phys_files),
                        "source": "tool_result"})
                    n_prefetched_total += len(phys_files)

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        turn_elapsed = time.monotonic() - turn_start
        per_turn.append({
            "turn": turn, "duration_s": round(turn_elapsed, 3),
            "n_tool_uses": len(fn_calls),
            "tool_names": [fc["name"] for fc in fn_calls],
            "fired_rules": fired_rules_this_turn,
            "dispatched_prefetches": dispatched_this_turn,
        })

        sub_path = workspace_dir / "submission.csv"
        if sub_path.is_file():
            submitted = True
            submission_size = sub_path.stat().st_size

        if not any_tool:
            break

    session_elapsed = time.monotonic() - session_start
    n_outputs = sum(1 for _ in workspace_dir.rglob("*") if _.is_file()) \
                if workspace_dir.is_dir() else 0

    if stager is not None:
        stager.shutdown(wait=True)

    return {
        "session_elapsed_s": round(session_elapsed, 3),
        "n_turns": len(per_turn), "n_tool_uses": tool_use_count,
        "files_opened_logical": files_opened,
        "submitted": submitted,
        "submission_bytes": submission_size,
        "n_workspace_outputs": n_outputs,
        "n_prefetched_files": n_prefetched_total,
        "per_turn": per_turn,
        "shell_io": _aggregate_io(shell_io_log),
        "shell_io_per_call": shell_io_log,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        help="MLE-bench competition id (e.g. aerial-cactus-identification)")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--mode", choices=["baseline", "staged"], required=True)
    parser.add_argument("--prompt-mode", choices=["hinted", "sparse"],
                        default="hinted")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_mlemt")
    parser.add_argument("--max-turns", type=int, default=12)
    parser.add_argument("--shell-timeout", type=int, default=180,
                        help="Per-shell-command timeout in seconds")
    args = parser.parse_args()

    if not SHIM.is_file():
        print(f"FATAL: shim missing at {SHIM}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    workload = load_mle_competition_dispatch(args.task)
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")

    # Collect physical file paths for eviction
    targets = []
    for k, paths in workload.workspace_prior.items():
        if k in ("all_files", "output_submission"):
            continue
        for p in paths:
            phys = resolve_logical(p, prefix_map)
            if Path(phys).is_file():
                targets.append(Path(phys))

    ev = evict(targets)
    total_mb = sum(p.stat().st_size for p in targets) / 1024 / 1024
    print(f"E-041 MLE-bench agentic | task={args.task} model={args.model} "
          f"mode={args.mode} prompt={args.prompt_mode}")
    print(f"  data files: {len(targets)} ({total_mb:.1f} MB)")
    print(f"  resident_frac_after_evict: {ev.get('resident_frac_sample','?')}")
    print()

    # Wrap in try/except so a runner crash (e.g., API context overflow,
    # SDK error, subprocess oddity) still writes a summary.json with the
    # crash info — otherwise the sweep loses the slot entirely.
    crash_info = None
    try:
        result = run_session(
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
            "session_elapsed_s": None,
            "n_turns": None,
            "n_tool_uses": None,
            "files_opened_logical": [],
            "submitted": False,
            "submission_bytes": 0,
            "n_prefetched_files": 0,
            "per_turn": [],
            "crash": crash_info,
        }
    result["task"] = args.task
    result["mode"] = args.mode
    result["model"] = args.model
    result["prompt_mode"] = args.prompt_mode
    result["evict"] = ev
    result["experiment"] = "E-041"
    result["total_input_mb"] = round(total_mb, 2)

    (args.out / "summary.json").write_text(json.dumps(result, indent=2))
    print(f"  session: {result['session_elapsed_s']}s | turns={result['n_turns']} | "
          f"tool_uses={result['n_tool_uses']} | submitted={result['submitted']}")
    if args.mode == "staged":
        print(f"  prefetched: {result['n_prefetched_files']} files")
    print(f"  wrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
