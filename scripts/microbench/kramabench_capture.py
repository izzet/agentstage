"""E-035 — KramaBench cross-benchmark multi-turn capture.

Captures Anthropic / Gemini streaming sessions on KramaBench tasks
into the same per-turn JSONL layout that path_b_multiturn produces, so
the existing replay machinery (auto_vs_hand, subset_replay, h7_loo,
falsepos) consumes them without modification.

Out of scope for this script: stager, shim, hot-tier dispatch. We only
need the reasoning + tool trail so we can replay AIOB-trained auto-rules
against KramaBench captures for H6 (frozen-rules cross-benchmark
generalization).

Usage:
    python scripts/microbench/kramabench_capture.py \\
        --task kb_wildfire_easy_1 \\
        --model claude-haiku-4-5 \\
        --prompt-mode hinted \\
        --out outputs/multi_turn/kb_wf1_haiku_hinted
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Load .env files (ours + sciiobench fallback) so AZURE_FOUNDRY_KEY etc.
# resolve without needing to source manually.
def _load_dotenv(p: Path) -> None:
    if not p.is_file():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k = k.strip()
        v = v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv(Path(__file__).resolve().parents[2] / ".env")
if os.environ.get("SCIIOBENCH_ROOT"):
    _load_dotenv(Path(os.environ["SCIIOBENCH_ROOT"]) / ".env")

from agentstage.workloads.kramabench import KB_MINIMAL_SLICE, load_kramabench_task  # noqa: E402


# ---------------------------------------------------------------------------
# Tool sandbox — KramaBench-specific (data lives outside AIOB allowlist)
# ---------------------------------------------------------------------------

def _resolve_logical(path: str, prefix_map: tuple[tuple[str, str], ...]) -> str:
    for lp, rp in prefix_map:
        if path.startswith(lp):
            return rp + path[len(lp):]
    return path


def execute_tool(name: str, args: dict, *, prefix_map, allow_root: str,
                 max_bytes: int = 4096, max_entries: int = 200) -> str:
    path = args.get("path", "")
    if not path:
        return f"ERROR: {name}: missing 'path' argument"
    phys = _resolve_logical(path, prefix_map)
    if not phys.startswith(allow_root):
        return (f"ERROR: {name}({path!r}): path outside permitted dataset roots. "
                f"Use /data/<domain>/... to access the benchmark data.")
    p = Path(phys)
    display = path.rstrip("/")
    try:
        if name == "list_dir":
            if not p.exists() or not p.is_dir():
                return f"ERROR: list_dir: not a directory: {phys}"
            entries = sorted(p.iterdir())[:max_entries]
            n_total = sum(1 for _ in p.iterdir())
            lines = [f"# Listing of {display} ({n_total} total entries; showing {len(entries)}):"]
            for e in entries:
                if e.is_file():
                    lines.append(f"  FILE  {display}/{e.name}  ({e.stat().st_size} bytes)")
                else:
                    lines.append(f"  DIR   {display}/{e.name}/")
            return "\n".join(lines)
        elif name in ("open_file", "read_file"):
            if not p.is_file():
                return f"ERROR: {name}: file does not exist: {phys}"
            size = p.stat().st_size
            with open(phys, "rb") as f:
                head = f.read(min(max_bytes, size))
            try:
                txt = head.decode("utf-8")
                return f"# Contents of {path} (first {len(head)}/{size} bytes):\n{txt}"
            except UnicodeDecodeError:
                return f"# Binary file {path} (size {size} bytes). First 64 bytes hex:\n{head[:64].hex()}"
        else:
            return f"ERROR: unknown tool: {name}"
    except Exception as e:
        return f"ERROR: {name}: {e!r}"


TOOLS_SCHEMA = [
    {
        "name": "list_dir",
        "description": "List the immediate children of a directory under /data/.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "open_file",
        "description": "Read the first few KB of a file under /data/.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Same as open_file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Stream capture — Anthropic SDK
# ---------------------------------------------------------------------------

def capture_anthropic(*, model: str, system: str, user_msg: str,
                      task_id: str, out_dir: Path, prefix_map, allow_root: str,
                      max_turns: int = 8, thinking_budget: int = 4096) -> dict:
    """Drive Anthropic Messages API turn-by-turn, recording every block."""
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
        raise RuntimeError("no Anthropic key (AZURE_FOUNDRY_KEY / ANTHROPIC_API_KEY)")

    messages: list[dict] = [{"role": "user", "content": user_msg}]
    turns_dir = out_dir / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    tool_use_count = 0
    files_opened: list[str] = []

    for turn in range(max_turns):
        turn_dir = turns_dir / f"turn_{turn:02d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        thinking_log = (turn_dir / "thinking.jsonl").open("w")
        text_log = (turn_dir / "text.jsonl").open("w")
        tool_use_log = (turn_dir / "tool_use.jsonl").open("w")
        tool_result_log = (turn_dir / "tool_result.jsonl").open("w")

        stream_started_ms = time.monotonic() * 1000

        body = {
            "model": model,
            "max_tokens": 8192,
            "temperature": 1.0,
            "messages": messages,
            "tools": TOOLS_SCHEMA,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        }
        if system:
            body["system"] = system

        with client.messages.stream(**body) as stream:
            block_kind: dict[int, str] = {}
            block_text: dict[int, list[str]] = {}
            block_first_ms: dict[int, float] = {}
            tool_inputs: dict[int, list[str]] = {}
            tool_meta: dict[int, dict] = {}

            for event in stream:
                t_ms = time.monotonic() * 1000 - stream_started_ms
                etype = getattr(event, "type", None)
                if etype == "content_block_start":
                    idx = event.index
                    b = event.content_block
                    kind = getattr(b, "type", None)
                    block_kind[idx] = kind
                    block_first_ms[idx] = t_ms
                    block_text[idx] = []
                    if kind == "tool_use":
                        tool_meta[idx] = {
                            "id": getattr(b, "id", ""),
                            "name": getattr(b, "name", ""),
                        }
                        tool_inputs[idx] = []
                elif etype == "content_block_delta":
                    idx = event.index
                    d = event.delta
                    dt = getattr(d, "type", None)
                    if dt == "thinking_delta":
                        s = getattr(d, "thinking", "")
                        if s:
                            block_text[idx].append(s)
                            thinking_log.write(json.dumps({
                                "t_ms": round(t_ms, 1), "block": idx,
                                "delta": s,
                            }) + "\n")
                    elif dt == "text_delta":
                        s = getattr(d, "text", "")
                        if s:
                            block_text[idx].append(s)
                            text_log.write(json.dumps({
                                "t_ms": round(t_ms, 1), "block": idx,
                                "delta": s,
                            }) + "\n")
                    elif dt == "input_json_delta":
                        s = getattr(d, "partial_json", "")
                        if s and idx in tool_inputs:
                            tool_inputs[idx].append(s)
            final = stream.get_final_message()

        # Snapshot assistant content for the next turn's history
        assistant_blocks: list[dict] = []
        if final and getattr(final, "content", None):
            for b in final.content:
                t = getattr(b, "type", None)
                if t == "thinking":
                    assistant_blocks.append({
                        "type": "thinking",
                        "thinking": getattr(b, "thinking", ""),
                        "signature": getattr(b, "signature", ""),
                    })
                elif t == "text":
                    assistant_blocks.append({"type": "text", "text": getattr(b, "text", "")})
                elif t == "tool_use":
                    assistant_blocks.append({
                        "type": "tool_use",
                        "id": getattr(b, "id", ""),
                        "name": getattr(b, "name", ""),
                        "input": getattr(b, "input", {}),
                    })
        messages.append({"role": "assistant", "content": assistant_blocks})

        # Execute every tool_use in this turn, log + add tool_result
        tool_results: list[dict] = []
        any_tool = False
        for b in assistant_blocks:
            if b["type"] != "tool_use":
                continue
            any_tool = True
            tool_use_count += 1
            t_in = b["input"]
            t_out = execute_tool(
                b["name"], t_in,
                prefix_map=prefix_map, allow_root=allow_root,
            )
            tool_use_log.write(json.dumps({
                "name": b["name"], "id": b["id"],
                "parsed_input": t_in,
            }) + "\n")
            tool_result_log.write(json.dumps({
                "tool_use_id": b["id"], "content": t_out,
            }) + "\n")
            if b["name"] in ("open_file", "read_file"):
                files_opened.append(t_in.get("path", ""))
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": b["id"],
                "content": t_out,
            })

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        if not any_tool:
            break

    return {
        "n_turns": turn + 1,
        "n_tool_uses": tool_use_count,
        "files_opened_logical": files_opened,
    }


# ---------------------------------------------------------------------------
# Stream capture — Gemini SDK
# ---------------------------------------------------------------------------

def capture_gemini(*, model: str, system: str, user_msg: str, task_id: str,
                   out_dir: Path, prefix_map, allow_root: str,
                   max_turns: int = 8) -> dict:
    """Drive Gemini SDK turn-by-turn, write same per-turn layout."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    fn_decls = []
    for ts in TOOLS_SCHEMA:
        fn_decls.append({
            "name": ts["name"],
            "description": ts["description"],
            "parameters": {
                "type": "OBJECT",
                "properties": {"path": {"type": "STRING"}},
                "required": ["path"],
            },
        })
    tools = [types.Tool(function_declarations=fn_decls)]

    turns_dir = out_dir / "turns"
    turns_dir.mkdir(parents=True, exist_ok=True)

    contents: list = [{"role": "user", "parts": [{"text": user_msg}]}]
    files_opened: list[str] = []
    tool_use_count = 0

    for turn in range(max_turns):
        turn_dir = turns_dir / f"turn_{turn:02d}"
        turn_dir.mkdir(parents=True, exist_ok=True)
        thinking_log = (turn_dir / "thinking.jsonl").open("w")
        text_log = (turn_dir / "text.jsonl").open("w")
        tool_use_log = (turn_dir / "tool_use.jsonl").open("w")
        tool_result_log = (turn_dir / "tool_result.jsonl").open("w")

        cfg = types.GenerateContentConfig(
            temperature=1.0,
            tools=tools,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            system_instruction=system or None,
        )
        stream_started_ms = time.monotonic() * 1000
        idx_counter = 0
        fn_calls: list[dict] = []
        assistant_parts: list = []

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
                        "t_ms": round(t_ms, 1), "block": idx_counter,
                        "delta": part.text,
                    }) + "\n")
                    assistant_parts.append({"text": part.text, "thought": True})
                elif getattr(part, "text", None):
                    text_log.write(json.dumps({
                        "t_ms": round(t_ms, 1), "block": idx_counter,
                        "delta": part.text,
                    }) + "\n")
                    assistant_parts.append({"text": part.text})
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    name = getattr(fc, "name", "")
                    args = dict(getattr(fc, "args", {}) or {})
                    synthetic_id = f"gem_call_{turn}_{len(fn_calls)}"
                    fn_calls.append({"id": synthetic_id, "name": name, "args": args})
                    tool_use_count += 1
                    assistant_parts.append({"function_call": {"name": name, "args": args}})

        # Persist assistant turn into contents for context
        contents.append({"role": "model", "parts": [
            {"text": p["text"]} if "text" in p else
            {"function_call": p["function_call"]}
            for p in assistant_parts
        ]})

        # Run any fn calls + record tool_use / tool_result
        responses: list[dict] = []
        for fc in fn_calls:
            out = execute_tool(
                fc["name"], fc["args"],
                prefix_map=prefix_map, allow_root=allow_root,
            )
            tool_use_log.write(json.dumps({
                "name": fc["name"], "id": fc["id"],
                "parsed_input": fc["args"],
            }) + "\n")
            tool_result_log.write(json.dumps({
                "tool_use_id": fc["id"], "content": out,
            }) + "\n")
            if fc["name"] in ("open_file", "read_file"):
                files_opened.append(fc["args"].get("path", ""))
            responses.append({
                "function_response": {
                    "name": fc["name"],
                    "response": {"output": out},
                },
            })

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        if responses:
            contents.append({"role": "function", "parts": responses})
        if not fn_calls:
            break

    return {
        "n_turns": turn + 1,
        "n_tool_uses": tool_use_count,
        "files_opened_logical": files_opened,
    }


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

def build_user_msg(workload, prompt_mode: str) -> str:
    """Construct the user message for this KramaBench task."""
    task = workload.task
    if prompt_mode == "hinted":
        # Hinted: include the listed data_sources from the task spec, so the
        # agent knows what filenames matter.
        sources_hint = ""
        if task.data_sources:
            sources_hint = (
                "\n\nThe data files relevant to this task are: "
                + ", ".join(task.data_sources)
                + "."
            )
        return (
            f"Task: {workload.task_id}\n\n"
            f"{task.query}{sources_hint}\n\n"
            f"All input data lives under /data/{task.domain}/. "
            f"Use list_dir and open_file to explore. "
            f"Think step-by-step about which files you need."
        )
    else:  # sparse
        return (
            f"Task: {workload.task_id}\n\n"
            f"{task.query}\n\n"
            f"All input data lives under /data/{task.domain}/. "
            f"Use list_dir to discover what's available."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True,
                        help="kb_<domain>_<task-suffix>")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--prompt-mode", choices=["hinted", "sparse"],
                        default="hinted")
    parser.add_argument("--max-turns", type=int, default=8)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    # Resolve task
    if args.task in KB_MINIMAL_SLICE:
        workload = KB_MINIMAL_SLICE[args.task]()
    else:
        # Allow kb_<domain>_<rest>: parse domain
        rest = args.task.removeprefix("kb_")
        domain, _, task_suffix = rest.partition("_")
        task_id = f"{domain}-{task_suffix.replace('_', '-')}"
        workload = load_kramabench_task(domain, task_id)

    domain = workload.task.domain
    allow_root = str((Path(__file__).resolve().parents[2]
                      / "external" / "benchmarks" / "kramabench"
                      / "data" / domain / "input").resolve())

    system_msg = ("You are a data-analysis agent. When you reason about "
                  "which files to open, prefer concrete absolute paths "
                  "under /data/. Then call list_dir or open_file.")
    user_msg = build_user_msg(workload, args.prompt_mode)

    provider = "gemini" if args.model.lower().startswith("gemini") else "anthropic"
    print(f"E-035 KramaBench capture | task={args.task} model={args.model} "
          f"mode={args.prompt_mode} provider={provider}")
    print(f"  prior keys: {list(workload.workspace_prior.keys())[:4]}...")
    print(f"  GT files:   {len(workload.ground_truth_full)}")
    print(f"  allow_root: {allow_root}")
    print()

    t0 = time.monotonic()
    if provider == "anthropic":
        result = capture_anthropic(
            model=args.model, system=system_msg, user_msg=user_msg,
            task_id=workload.task_id, out_dir=args.out,
            prefix_map=workload.prefix_map, allow_root=allow_root,
            max_turns=args.max_turns,
        )
    else:
        result = capture_gemini(
            model=args.model, system=system_msg, user_msg=user_msg,
            task_id=workload.task_id, out_dir=args.out,
            prefix_map=workload.prefix_map, allow_root=allow_root,
            max_turns=args.max_turns,
        )
    elapsed = time.monotonic() - t0

    summary = {
        "experiment": "E-035",
        "task_id": workload.task_id,
        "domain": workload.task.domain,
        "model": args.model,
        "provider": provider,
        "prompt_mode": args.prompt_mode,
        "elapsed_s": round(elapsed, 2),
        "n_turns": result["n_turns"],
        "n_tool_uses": result["n_tool_uses"],
        "files_opened_logical": result["files_opened_logical"],
        "gt_files": list(workload.ground_truth_full),
        "n_prior_files": len(workload.all_workspace_paths),
        "prior_buckets": list(workload.workspace_prior.keys()),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  captured {result['n_turns']} turns, "
          f"{result['n_tool_uses']} tool uses in {elapsed:.1f}s")
    print(f"  files opened: {result['files_opened_logical']}")
    print(f"  wrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
