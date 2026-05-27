"""E-042 — AIOB full-agentic-loop with AgentStage integrated.

Mirror of scripts/microbench/dsbench_multiturn.py adapted for AIOB
workloads (aiob_104 / aiob_107 / aiob_110). The LLM agent gets the
AIOB task description and a /data/<dataset>/ sandbox; tools are
list_dir / open_file / read_file / write_file / run_shell_command.

Differences from DSBench/MLE-bench:
  - AIOB workspace_prior buckets can have thousands of files (e.g.,
    aiob_107 band_08 = ~5000 NetCDFs). We cap per-rule prefetch to
    `STAGER_BUCKET_CAP = 200` files; larger buckets get tier-3 marked
    and skipped (same policy that path_b_multiturn.py uses).
  - No standard 'submission.csv' output — AIOB tasks write task-specific
    artifacts. We track session wall time + per-turn timings + the
    final shell rc; "submitted" becomes "produced any /workspace output".
  - Cold root is the AIOB datasets ancestor (one level above per-task
    subdirs, so the shim covers all 3 datasets simultaneously).

Two modes:
  --mode baseline : Stager + shim DISABLED. Subprocess runs naked.
  --mode staged   : SessionDetector + AutoRuleGenerator fire during
                    streaming → Stager.prefetch fires when rules
                    activate, copying cold files to /dev/shm. The
                    subprocess launched by run_shell_command runs with
                    LD_PRELOAD=libagentstage_shim.so so reads under the
                    cold root redirect to hot copies.

Supports both Anthropic and Gemini models (dispatch by model name).

Usage:
    python scripts/microbench/aiob_multiturn.py --task aiob_110 \\
        --model claude-haiku-4-5 --mode staged \\
        --out outputs/aiob_mt/aiob_110_haiku_staged_r1
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
from agentstage.workloads.aiob import (  # noqa: E402
    Workload, load_aiob_104, load_aiob_107, load_aiob_110,
)

REPO = Path(__file__).resolve().parents[2]
SHIM = (REPO / "src" / "agentstage" / "stager" / "shim"
        / "libagentstage_shim.so").resolve()

# Cap per-rule prefetch size — AIOB buckets can have 1000+ files
# (band_08 NetCDFs etc). Prefetching all of them would blow past any
# reasoning slack. Same policy as path_b_multiturn.py.
STAGER_BUCKET_CAP = 200

TASK_LOADERS = {
    "aiob_104": load_aiob_104,
    "aiob_107": load_aiob_107,
    "aiob_110": load_aiob_110,
}


def evict(paths: list[Path], *, verify: bool = True) -> dict:
    """Cold-cache methodology, same as E-030/E-039/E-040/E-041."""
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
     "description": "List the immediate children of a directory under /data/<dataset>/.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "open_file",
     "description": "Read the first ~4 KB of a file under /data/<dataset>/.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "read_file",
     "description": "Alias for open_file.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "run_shell_command",
     "description": "Run a shell command in the task workspace. Use to "
                    "execute Python scripts you've written. /data/<dataset>/ "
                    "(input data, read-only) and /workspace/ (writable, CWD) "
                    "are accessible. Returns stdout+stderr (truncated to 4 KB).",
     "input_schema": {"type": "object",
                       "properties": {"cmd": {"type": "string"}},
                       "required": ["cmd"]}},
    {"name": "write_file",
     "description": "Write content to /workspace/<filename>.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"},
                                       "content": {"type": "string"}},
                       "required": ["path", "content"]}},
]


def make_tool_executor(workload: Workload, workspace_dir: Path,
                       *, mode: str, hot_root: Path, cold_root: str):
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")
    log_root = prefix_map[0][0].rstrip("/")
    workspace_dir.mkdir(parents=True, exist_ok=True)

    def _resolve(path: str) -> tuple[str, bool]:
        if path.startswith("/data/") or path == "/data":
            phys = resolve_logical(path, prefix_map)
            if phys.startswith(data_phys_root):
                return phys, True
            # Fallback: cold_root + relative
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
                return (f"ERROR: list_dir({path!r}): outside sandbox. "
                        f"Use /data/ or /workspace/.")
            p = Path(phys)
            if not p.is_dir():
                return f"ERROR: list_dir: not a directory: {path}"
            MAX = 200
            entries = sorted(p.iterdir())
            n_total = len(entries)
            shown = entries[:MAX]
            display = path.rstrip("/")
            header = (f"# Listing of {display} ({n_total} entries"
                      + (f", showing first {MAX}):" if n_total > MAX else "):"))
            lines = [header]
            for e in shown:
                kind = "FILE" if e.is_file() else "DIR "
                sz = e.stat().st_size if e.is_file() else 0
                full = f"{display}/{e.name}" + ("/" if e.is_dir() else "")
                lines.append(f"  {kind}  {full}  ({sz} bytes)" if e.is_file()
                             else f"  {kind}  {full}")
            if n_total > MAX:
                lines.append(f"  ... ({n_total - MAX} more entries elided)")
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
                return "ERROR: write_file: can only write to /workspace/"
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
                env["AGENTSTAGE_RETRY_SPIN_MS"] = "20"
            else:
                env.pop("LD_PRELOAD", None)
                env["AGENTSTAGE_SHIM_DISABLE"] = "1"
            # Set up data/<dataset>/<subpath> symlink in the agent's
            # workspace cwd. log_root is like "/data/steinmetz_neuropixels/raw"
            # — we mirror that under <workspace>/data/.
            ds_path = log_root[len("/data/"):]  # e.g. "steinmetz_neuropixels/raw"
            data_link = workspace_dir / "data" / ds_path
            if not data_link.exists():
                data_link.parent.mkdir(parents=True, exist_ok=True)
                try:
                    data_link.symlink_to(data_phys_root)
                except FileExistsError:
                    pass
            t0 = time.monotonic()
            try:
                r = subprocess.run(["/bin/bash", "-c", cmd],
                                   cwd=str(workspace_dir), env=env,
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
                          + "\n[TIMEOUT after 180s — solution too slow]")
            elapsed = time.monotonic() - t0
            out = stdout[-3000:]
            err = stderr[-1000:]
            return (f"# run_shell_command (rc={rc}, {elapsed:.2f}s):\n"
                    f"## stdout:\n{out}\n"
                    f"## stderr:\n{err}\n")
        return f"ERROR: unknown tool: {name}"

    return execute


def _build_prompts(workload: Workload, prompt_mode: str) -> tuple[str, str]:
    """Returns (user_msg, system_msg). AIOB task descriptions are long and
    detailed; we pass them through verbatim (no rephrasing — preserves
    benchmark integrity)."""
    log_root = workload.prefix_map[0][0].rstrip("/")
    ds_name = log_root[len("/data/"):]
    tid = workload.task_id

    if prompt_mode == "hinted":
        user_msg = (
            f"Task: {tid}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"The dataset lives at /data/{ds_name}/. Use list_dir / open_file "
            f"to inspect; write your Python solution to /workspace/solution.py "
            f"and run it with run_shell_command. Any output artifacts go to "
            f"/workspace/."
        )
    else:
        user_msg = (
            f"Task: {tid}\n\n"
            f"{workload.task.task_inst}\n\n"
            f"Use list_dir to discover data layout under /data/. "
            f"Working directory is /workspace/."
        )

    system_msg = (
        "You are a scientific-data agent solving an AgentIOBench task. "
        f"Workspace: data/{ds_name}/ for inputs (read-only), CWD is "
        f"/workspace/. Tools: list_dir, open_file, read_file, write_file, "
        f"run_shell_command. Pre-installed: pandas, numpy, scipy, sklearn, "
        f"lightgbm, openpyxl, h5py, netCDF4, pynwb, pyarrow, matplotlib. "
        f"Shell commands time out at 180s — prefer fast operations "
        f"(read a sample, simple stats, avoid full-dataset training). "
        f"For a quick-and-correct first pass, do summary statistics on a "
        f"subset, then iterate."
    )
    return user_msg, system_msg


def _dispatch_prefetches(*, new_acts, stager, prefix_map, turn: int,
                          source: str, dispatched: list, n_total_holder: list) -> None:
    """Shared logic for dispatching Stager.prefetch from rule activations.
    Caps each rule's dispatched file count at STAGER_BUCKET_CAP."""
    if stager is None:
        return
    seen_phys: set[str] = set()
    for act in new_acts:
        phys_files = [resolve_logical(p, prefix_map) for p in act.detected_files]
        phys_files = [p for p in phys_files
                      if Path(p).is_file() and p not in seen_phys]
        # CAP per-bucket size to prevent runaway many-files staging
        if len(phys_files) > STAGER_BUCKET_CAP:
            phys_files = phys_files[:STAGER_BUCKET_CAP]
        for p in phys_files:
            seen_phys.add(p)
        if not phys_files:
            continue
        rule_id = f"turn{turn}:{act.rule_name}"
        if source != "thinking":
            rule_id += f":{source}"
        hint = DataHint(
            detected_files=tuple(phys_files),
            tier=1 if len(phys_files) <= 10 else 3,
            fired_at_ms=act.fired_at_ms or 0.0,
            rule_id=rule_id,
        )
        stager.prefetch(hint)
        dispatched.append({
            "rule": act.rule_name, "n_files": len(phys_files),
            "source": source,
        })
        n_total_holder[0] += len(phys_files)


def run_session(*, workload: Workload, model: str, mode: str,
                prompt_mode: str, out_dir: Path,
                hot_root: Path, max_turns: int = 12,
                thinking_budget: int = 4096) -> dict:
    if model.lower().startswith("gemini"):
        return _run_session_gemini(
            workload=workload, model=model, mode=mode,
            prompt_mode=prompt_mode, out_dir=out_dir,
            hot_root=hot_root, max_turns=max_turns,
        )
    return _run_session_anthropic(
        workload=workload, model=model, mode=mode,
        prompt_mode=prompt_mode, out_dir=out_dir,
        hot_root=hot_root, max_turns=max_turns,
        thinking_budget=thinking_budget,
    )


def _run_session_anthropic(*, workload: Workload, model: str, mode: str,
                            prompt_mode: str, out_dir: Path,
                            hot_root: Path, max_turns: int = 12,
                            thinking_budget: int = 4096) -> dict:
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
    cold_root_anc = str(Path(data_phys_root).parent.parent)
    # data_phys_root = .../<dataset>/raw  → cold_root_anc = .../agentiobench/datasets

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
    user_msg, system_msg = _build_prompts(workload, prompt_mode)

    messages: list[dict] = [{"role": "user", "content": user_msg}]
    session_start = time.monotonic()
    tool_use_count = 0
    files_opened: list[str] = []
    n_prefetched_holder = [0]
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
                        thinking_log.write(json.dumps({
                            "t_ms": round(t_ms, 1), "block": idx,
                            "delta": piece}) + "\n")
                elif dt == "text_delta":
                    piece = getattr(d, "text", "")
                    if piece:
                        text_log.write(json.dumps({
                            "t_ms": round(t_ms, 1), "block": idx,
                            "delta": piece}) + "\n")
            final = stream.get_final_message()

        assistant_blocks: list[dict] = []
        if final and getattr(final, "content", None):
            for b in final.content:
                btype = getattr(b, "type", None)
                if btype == "thinking":
                    assistant_blocks.append({
                        "type": "thinking", "thinking": getattr(b, "thinking", ""),
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
        if mode == "staged":
            _dispatch_prefetches(
                new_acts=new_acts, stager=stager, prefix_map=prefix_map,
                turn=turn, source="thinking",
                dispatched=dispatched_this_turn,
                n_total_holder=n_prefetched_holder,
            )

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

        if tool_result_blocks_for_detector:
            tr_acts = session_detector.feed_tool_results(
                tool_result_blocks_for_detector)
            for a in tr_acts:
                fired_rules_this_turn.append(a.rule_name)
            if mode == "staged" and tr_acts:
                _dispatch_prefetches(
                    new_acts=tr_acts, stager=stager, prefix_map=prefix_map,
                    turn=turn, source="tool_result",
                    dispatched=dispatched_this_turn,
                    n_total_holder=n_prefetched_holder,
                )

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        turn_elapsed = time.monotonic() - turn_start
        per_turn.append({
            "turn": turn, "duration_s": round(turn_elapsed, 3),
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

    # AIOB tasks don't have a standardized "submission" — count any
    # workspace artifacts as "produced output"
    n_outputs = sum(1 for _ in workspace_dir.rglob("*") if _.is_file()) \
                if workspace_dir.is_dir() else 0
    workspace_bytes = sum(p.stat().st_size for p in workspace_dir.rglob("*")
                         if p.is_file()) if workspace_dir.is_dir() else 0

    if stager is not None:
        stager.shutdown(wait=True)

    return {
        "session_elapsed_s": round(session_elapsed, 3),
        "n_turns": len(per_turn),
        "n_tool_uses": tool_use_count,
        "files_opened_logical": files_opened,
        "submitted": n_outputs > 0,
        "submission_bytes": workspace_bytes,
        "n_workspace_outputs": n_outputs,
        "n_prefetched_files": n_prefetched_holder[0],
        "per_turn": per_turn,
    }


def _run_session_gemini(*, workload: Workload, model: str, mode: str,
                         prompt_mode: str, out_dir: Path,
                         hot_root: Path, max_turns: int = 12) -> dict:
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

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
        stager = Stager(
            hot_root=hot_root, cold_roots=[Path(cold_root_anc)],
            max_workers=4, capacity_bytes=64 * 1024**3,
        )

    workspace_dir = out_dir / "agent_workspace"
    execute = make_tool_executor(workload, workspace_dir,
                                  mode=mode, hot_root=hot_root,
                                  cold_root=cold_root_anc)
    turns_dir = out_dir / "turns"; turns_dir.mkdir(parents=True, exist_ok=True)
    user_msg, system_msg = _build_prompts(workload, prompt_mode)

    fn_decls = []
    for ts in TOOLS_SCHEMA:
        props = {}
        for pname in ts["input_schema"]["properties"]:
            props[pname] = {"type": "STRING"}
        fn_decls.append({
            "name": ts["name"], "description": ts["description"],
            "parameters": {"type": "OBJECT", "properties": props,
                            "required": ts["input_schema"].get("required", [])},
        })
    tools = [types.Tool(function_declarations=fn_decls)]

    contents: list = [{"role": "user", "parts": [{"text": user_msg}]}]
    session_start = time.monotonic()
    tool_use_count = 0
    files_opened: list[str] = []
    n_prefetched_holder = [0]
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
            temperature=1.0, tools=tools,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            system_instruction=system_msg or None,
        )
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
                        "delta": part.text}) + "\n")
                    assistant_blocks.append({
                        "type": "thinking", "thinking": part.text})
                elif getattr(part, "text", None):
                    text_log.write(json.dumps({
                        "t_ms": round(t_ms, 1), "block": 0,
                        "delta": part.text}) + "\n")
                    assistant_blocks.append({
                        "type": "text", "text": part.text})
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    name = getattr(fc, "name", "") or ""
                    args = dict(getattr(fc, "args", {}) or {})
                    sid = f"gem_{turn}_{len(fn_calls)}"
                    fn_calls.append({"id": sid, "name": name, "args": args})
                    assistant_blocks.append({
                        "type": "tool_use", "id": sid, "name": name,
                        "input": args})

        contents.append({"role": "model", "parts": [
            {"text": b.get("text") or b.get("thinking", "")}
            if b["type"] in ("text", "thinking")
            else {"function_call": {"name": b["name"], "args": b["input"]}}
            for b in assistant_blocks
        ]})

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
        if mode == "staged":
            _dispatch_prefetches(
                new_acts=new_acts, stager=stager, prefix_map=prefix_map,
                turn=turn, source="thinking",
                dispatched=dispatched_this_turn,
                n_total_holder=n_prefetched_holder,
            )

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

        if tool_result_blocks_for_detector:
            tr_acts = session_detector.feed_tool_results(
                tool_result_blocks_for_detector)
            for a in tr_acts:
                fired_rules_this_turn.append(a.rule_name)
            if mode == "staged" and tr_acts:
                _dispatch_prefetches(
                    new_acts=tr_acts, stager=stager, prefix_map=prefix_map,
                    turn=turn, source="tool_result",
                    dispatched=dispatched_this_turn,
                    n_total_holder=n_prefetched_holder,
                )

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
    n_outputs = sum(1 for _ in workspace_dir.rglob("*") if _.is_file()) \
                if workspace_dir.is_dir() else 0
    workspace_bytes = sum(p.stat().st_size for p in workspace_dir.rglob("*")
                         if p.is_file()) if workspace_dir.is_dir() else 0

    if stager is not None:
        stager.shutdown(wait=True)

    return {
        "session_elapsed_s": round(session_elapsed, 3),
        "n_turns": len(per_turn), "n_tool_uses": tool_use_count,
        "files_opened_logical": files_opened,
        "submitted": n_outputs > 0,
        "submission_bytes": workspace_bytes,
        "n_workspace_outputs": n_outputs,
        "n_prefetched_files": n_prefetched_holder[0],
        "per_turn": per_turn,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True, choices=list(TASK_LOADERS.keys()))
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--mode", choices=["baseline", "staged"], required=True)
    parser.add_argument("--prompt-mode", choices=["hinted", "sparse"],
                        default="hinted")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--hot-root", default="/dev/shm/agentstage_aiobmt")
    parser.add_argument("--max-turns", type=int, default=12)
    args = parser.parse_args()

    if not SHIM.is_file():
        print(f"FATAL: shim missing at {SHIM}", file=sys.stderr)
        return 2

    args.out.mkdir(parents=True, exist_ok=True)
    workload = TASK_LOADERS[args.task]()
    prefix_map = workload.prefix_map
    data_phys_root = prefix_map[0][1].rstrip("/")

    # Eviction targets: a representative sample from each bucket. Full
    # workspace_prior can be 18k files for aiob_107 — too many to evict
    # one-by-one. Cap at the first 200 from all_workspace_paths.
    EVICT_CAP = 200
    targets: list[Path] = []
    for p in workload.all_workspace_paths[:EVICT_CAP]:
        phys = resolve_logical(p, prefix_map)
        if Path(phys).is_file():
            targets.append(Path(phys))

    ev = evict(targets)
    print(f"E-042 AIOB agentic | task={args.task} model={args.model} "
          f"mode={args.mode} prompt={args.prompt_mode}")
    print(f"  data root: {data_phys_root}")
    print(f"  evicted: {ev.get('files',0)} files / {ev.get('bytes',0)/1024/1024:.1f} MB "
          f"(sample, capped at {EVICT_CAP})")
    print(f"  resident_frac_after_evict: {ev.get('resident_frac_sample','?')}")
    print()

    crash_info = None
    try:
        result = run_session(
            workload=workload, model=args.model, mode=args.mode,
            prompt_mode=args.prompt_mode, out_dir=args.out,
            hot_root=Path(args.hot_root), max_turns=args.max_turns,
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
