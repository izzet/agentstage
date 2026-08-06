"""E-037 — SAB cross-benchmark multi-turn capture (mock-sandbox).

Captures Anthropic / Gemini streaming sessions on ScienceAgentBench
tasks. Since the SAB data zip isn't bundled, the sandbox is MOCKED:
list_dir returns the task's dataset_folder_tree, open_file returns
dataset_preview. This is sufficient for H6 detection-quality measurement
— the agent's reasoning about which files to read is unchanged by
whether the bytes are real.

Usage:
    python scripts/microbench/sab_capture.py \\
        --instance-id 4 --model claude-haiku-4-5 \\
        --prompt-mode hinted \\
        --out outputs/multi_turn/sab_004_haiku_hinted
"""

from __future__ import annotations

import argparse
import json
import os
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
if os.environ.get("SCIIOBENCH_ROOT"):
    _load_dotenv(Path(os.environ["SCIIOBENCH_ROOT"]) / ".env")

from agentstage.workloads.scienceagentbench import (  # noqa: E402
    load_sab_task, sab_minimal_slice, SABWorkload,
)


def mock_execute_tool(name: str, args: dict, *,
                       workload: SABWorkload) -> str:
    """Mock sandbox: list_dir → folder tree slice; open_file → preview."""
    path = args.get("path", "")
    if not path:
        return f"ERROR: {name}: missing 'path' argument"

    if name == "list_dir":
        # Filter folder tree to entries under `path` if asked specifically.
        # For SAB, return the full tree always — the tree IS the listing.
        prior_files = workload.all_workspace_paths
        sub = [p for p in prior_files if p.startswith(path.rstrip("/") + "/")
               or p == path]
        lines = [f"# Listing of {path} ({len(sub)} entries):"]
        for p in sub:
            lines.append(f"  FILE  {p}  (unknown bytes — mock sandbox)")
        if not sub:
            lines.append(f"  (no entries under {path}; full tree follows)")
            lines.append(workload.task.dataset_folder_tree)
        return "\n".join(lines)
    elif name in ("open_file", "read_file"):
        preview = workload.task.dataset_preview
        return (f"# Contents of {path} (mock preview, real file not bundled):\n"
                f"{preview[:2000]}")
    else:
        return f"ERROR: unknown tool: {name}"


TOOLS_SCHEMA = [
    {"name": "list_dir",
     "description": "List directory under /data/.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "open_file",
     "description": "Read first KB of file under /data/.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
    {"name": "read_file",
     "description": "Same as open_file.",
     "input_schema": {"type": "object",
                       "properties": {"path": {"type": "string"}},
                       "required": ["path"]}},
]


def capture_anthropic(*, model: str, system: str, user_msg: str,
                      workload: SABWorkload, out_dir: Path,
                      max_turns: int = 6, thinking_budget: int = 4096) -> dict:
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

        body = {"model": model, "max_tokens": 8192, "temperature": 1.0,
                "messages": messages, "tools": TOOLS_SCHEMA,
                "thinking": {"type": "enabled", "budget_tokens": thinking_budget}}
        if system:
            body["system"] = system

        with client.messages.stream(**body) as stream:
            for event in stream:
                t_ms = time.monotonic() * 1000 - stream_started_ms
                etype = getattr(event, "type", None)
                if etype == "content_block_delta":
                    idx = event.index
                    d = event.delta
                    dt = getattr(d, "type", None)
                    if dt == "thinking_delta":
                        s = getattr(d, "thinking", "")
                        if s:
                            thinking_log.write(json.dumps({
                                "t_ms": round(t_ms, 1), "block": idx, "delta": s,
                            }) + "\n")
                    elif dt == "text_delta":
                        s = getattr(d, "text", "")
                        if s:
                            text_log.write(json.dumps({
                                "t_ms": round(t_ms, 1), "block": idx, "delta": s,
                            }) + "\n")
            final = stream.get_final_message()

        assistant_blocks: list[dict] = []
        if final and getattr(final, "content", None):
            for b in final.content:
                t = getattr(b, "type", None)
                if t == "thinking":
                    assistant_blocks.append({"type": "thinking",
                                             "thinking": getattr(b, "thinking", ""),
                                             "signature": getattr(b, "signature", "")})
                elif t == "text":
                    assistant_blocks.append({"type": "text",
                                             "text": getattr(b, "text", "")})
                elif t == "tool_use":
                    assistant_blocks.append({"type": "tool_use",
                                             "id": getattr(b, "id", ""),
                                             "name": getattr(b, "name", ""),
                                             "input": getattr(b, "input", {})})
        messages.append({"role": "assistant", "content": assistant_blocks})

        tool_results = []
        any_tool = False
        for b in assistant_blocks:
            if b["type"] != "tool_use":
                continue
            any_tool = True
            tool_use_count += 1
            out = mock_execute_tool(b["name"], b["input"], workload=workload)
            tool_use_log.write(json.dumps({"name": b["name"], "id": b["id"],
                                            "parsed_input": b["input"]}) + "\n")
            tool_result_log.write(json.dumps({"tool_use_id": b["id"],
                                               "content": out}) + "\n")
            if b["name"] in ("open_file", "read_file"):
                files_opened.append(b["input"].get("path", ""))
            tool_results.append({"type": "tool_result",
                                 "tool_use_id": b["id"], "content": out})

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        if tool_results:
            messages.append({"role": "user", "content": tool_results})
        if not any_tool:
            break

    return {"n_turns": turn + 1, "n_tool_uses": tool_use_count,
            "files_opened_logical": files_opened}


def capture_gemini(*, model: str, system: str, user_msg: str,
                   workload: SABWorkload, out_dir: Path,
                   max_turns: int = 6) -> dict:
    from google import genai
    from google.genai import types
    api_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError("GOOGLE_GEMINI_API_KEY not set")
    client = genai.Client(api_key=api_key)

    fn_decls = [{"name": ts["name"], "description": ts["description"],
                 "parameters": {"type": "OBJECT",
                                "properties": {"path": {"type": "STRING"}},
                                "required": ["path"]}}
                for ts in TOOLS_SCHEMA]
    tools = [types.Tool(function_declarations=fn_decls)]

    turns_dir = out_dir / "turns"; turns_dir.mkdir(parents=True, exist_ok=True)
    contents = [{"role": "user", "parts": [{"text": user_msg}]}]
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
            temperature=1.0, tools=tools,
            thinking_config=types.ThinkingConfig(include_thoughts=True),
            system_instruction=system or None)
        stream_started_ms = time.monotonic() * 1000
        fn_calls = []
        assistant_parts = []
        for chunk in client.models.generate_content_stream(
            model=model, contents=contents, config=cfg
        ):
            t_ms = time.monotonic() * 1000 - stream_started_ms
            if not chunk.candidates or not chunk.candidates[0].content:
                continue
            for part in chunk.candidates[0].content.parts:
                if getattr(part, "thought", False) and getattr(part, "text", None):
                    thinking_log.write(json.dumps({
                        "t_ms": round(t_ms, 1), "block": 0,
                        "delta": part.text}) + "\n")
                    assistant_parts.append({"text": part.text, "thought": True})
                elif getattr(part, "text", None):
                    text_log.write(json.dumps({
                        "t_ms": round(t_ms, 1), "block": 0,
                        "delta": part.text}) + "\n")
                    assistant_parts.append({"text": part.text})
                elif getattr(part, "function_call", None):
                    fc = part.function_call
                    sid = f"gem_{turn}_{len(fn_calls)}"
                    fn_calls.append({"id": sid, "name": fc.name,
                                     "args": dict(fc.args or {})})
                    tool_use_count += 1
                    assistant_parts.append({"function_call": {
                        "name": fc.name, "args": dict(fc.args or {})}})

        contents.append({"role": "model", "parts": [
            {"text": p["text"]} if "text" in p else
            {"function_call": p["function_call"]}
            for p in assistant_parts
        ]})

        responses = []
        for fc in fn_calls:
            out = mock_execute_tool(fc["name"], fc["args"], workload=workload)
            tool_use_log.write(json.dumps({"name": fc["name"], "id": fc["id"],
                                            "parsed_input": fc["args"]}) + "\n")
            tool_result_log.write(json.dumps({"tool_use_id": fc["id"],
                                               "content": out}) + "\n")
            if fc["name"] in ("open_file", "read_file"):
                files_opened.append(fc["args"].get("path", ""))
            responses.append({"function_response": {
                "name": fc["name"], "response": {"output": out}}})

        thinking_log.close(); text_log.close()
        tool_use_log.close(); tool_result_log.close()

        if responses:
            contents.append({"role": "function", "parts": responses})
        if not fn_calls:
            break

    return {"n_turns": turn + 1, "n_tool_uses": tool_use_count,
            "files_opened_logical": files_opened}


def build_user_msg(workload: SABWorkload, prompt_mode: str) -> str:
    t = workload.task
    if prompt_mode == "hinted":
        return (
            f"Task (SAB instance {t.instance_id}, domain={t.domain}):\n\n"
            f"{t.task_inst}\n\n"
            f"Dataset folder tree:\n{t.dataset_folder_tree}\n\n"
            f"Output filename: {t.output_fname}\n\n"
            f"Use list_dir / open_file to inspect any file under /data/. "
            f"Think step-by-step about which files you need."
        )
    else:  # sparse — drop tree
        return (
            f"Task (SAB instance {t.instance_id}, domain={t.domain}):\n\n"
            f"{t.task_inst}\n\n"
            f"All input data lives under /data/. Use list_dir to discover what's available."
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--instance-id", type=int)
    parser.add_argument("--task", help="sab_NNN slot (overrides --instance-id)")
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--prompt-mode", choices=["hinted", "sparse"], default="hinted")
    parser.add_argument("--max-turns", type=int, default=6)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.task:
        picks = sab_minimal_slice()
        if args.task in picks:
            iid = picks[args.task]
        else:
            iid = int(args.task.removeprefix("sab_"))
    else:
        iid = args.instance_id
    if iid is None:
        print("FATAL: need --instance-id or --task sab_NNN", file=sys.stderr)
        return 2

    workload = load_sab_task(iid)
    system_msg = ("You are a data-analysis agent. Reason about which files "
                  "to read using concrete absolute paths under /data/.")
    user_msg = build_user_msg(workload, args.prompt_mode)
    provider = "gemini" if args.model.lower().startswith("gemini") else "anthropic"
    print(f"E-037 SAB capture | sab_{iid:03d} ({workload.task.domain}) "
          f"model={args.model} mode={args.prompt_mode}")
    print(f"  prior n_files: {len(workload.all_workspace_paths)}")

    t0 = time.monotonic()
    if provider == "anthropic":
        result = capture_anthropic(
            model=args.model, system=system_msg, user_msg=user_msg,
            workload=workload, out_dir=args.out, max_turns=args.max_turns,
        )
    else:
        result = capture_gemini(
            model=args.model, system=system_msg, user_msg=user_msg,
            workload=workload, out_dir=args.out, max_turns=args.max_turns,
        )
    elapsed = time.monotonic() - t0

    summary = {
        "experiment": "E-037",
        "task_id": workload.task_id,
        "instance_id": iid,
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
        "sandbox": "mock — real SAB data zip not bundled",
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"\n  {result['n_turns']} turns, {result['n_tool_uses']} tool uses, "
          f"{elapsed:.1f}s")
    print(f"  files: {result['files_opened_logical']}")
    print(f"  wrote {args.out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
