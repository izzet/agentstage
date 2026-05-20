"""Path A — minimum live smoke for AgentStage.

Single Anthropic Haiku 4.5 call on aiob_107 with planning prompt.
Streaming → predictor → stager pipeline runs live. We catch the first
tool_use the LLM emits, execute its open() through the LD_PRELOAD shim,
time the read. For comparison: same file, shim-disabled, freshly evicted.

Validates that the live pipeline works end-to-end with real LLM thinking
content (vs Path 0 which replayed a recorded stream).

Usage (via scripts/path_a_run.sh which sets LD_PRELOAD):

  LD_PRELOAD=$SHIM \
  AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path_a \
  AGENTSTAGE_COLD_ROOTS=/mnt/common/datasets-staging/agentiobench/datasets \
  AZURE_FOUNDRY_KEY=$KEY \
  AZURE_FOUNDRY_ANTHROPIC_URL=https://... \
  python -m agentstage.runners.path_a_smoke --out outputs/path_a/<ts>/
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from agentstage.client.anthropic import AnthropicClient
from agentstage.predictor.rules import get_ruleset
from agentstage.stager import Stager, StagingReport
from agentstage.workloads.aiob import load_aiob_107, load_aiob_107_s3


# Task prompt — same shape the PoC used for aiob_107 sonnet+PP probes
TASK_USER_MSG = """\
Task: aiob_107_meteorology_goes_cmi_composites

Analyze the downloaded GOES-16 ABI-L2-CMIPC bundle for 2024-05-01 through 2024-05-07 UTC.
The dataset contains band 08, 09, and 10 NetCDF files for the CONUS scene at ~5-minute cadence
(roughly 6000 files total).

Extract brightness temperature time series at five named locations for all three bands:
  Houston       (grid row 457, col  690)
  Atlanta       (grid row 241, col 1306)
  Dallas        (grid row 306, col  663)
  Nashville     (grid row 124, col 1202)
  Oklahoma City (grid row 177, col  672)

For each file, read the CMI (brightness temperature) and DQF (data quality flag) variables
for a 10x10 pixel box centred on each location, filter to DQF == 0, compute spatial mean.

Save:
  - /repo/result/goes_cmi_timeseries.csv  (one row per location-band-file)
  - /repo/result/goes_cmi_point_timeseries.png  (diurnal cycle per band)
  - /repo/result/report.md

The workspace layout under /data/goes_cmi_composites/raw/ is:
  YYYY/DDD/HH/OR_ABI-L2-CMIPC-M6C{08,09,10}_G16_s<timestamp>_e<...>_c<...>.nc
where DDD is the 3-digit day-of-year (122-128) and HH is the 2-digit hour (00-23).

Available tools (single-turn — produce a tool_use block for the FIRST file you want to inspect):
  open_file(path)  - read a single NetCDF file's metadata + first data block
  list_dir(path)   - list contents of a directory
"""

PLANNING_PROMPT_SUFFIX = """

Before issuing any tool call, think step-by-step about:
  (1) which specific file path(s) you plan to inspect first, and why,
  (2) the order in which you plan to process the rest of the data,
  (3) what memory budget and chunk layout you will need.
"""


def evict_cold_file(path: Path) -> float:
    """Drop the page cache for `path` via POSIX_FADV_DONTNEED. Returns
    resident_frac afterwards (best-effort verification)."""
    try:
        from agentiobench.utils.cache import _resident_pages
    except ImportError:
        _resident_pages = None  # type: ignore
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(fd)
    os.sync()
    if _resident_pages is None:
        return -1.0
    try:
        resident, total = _resident_pages(path)
        return resident / total if total else 0.0
    except OSError:
        return -1.0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5",
                        help="Anthropic model id (default: claude-haiku-4-5)")
    parser.add_argument("--budget", type=int, default=8192,
                        help="thinking_budget tokens (default: 8192)")
    parser.add_argument("--workload", default="aiob_107",
                        choices=["aiob_107", "aiob_107_s3"],
                        help="Which workload to load (default: aiob_107). "
                             "aiob_107_s3 uses mountpoint-s3 against the "
                             "public NOAA bucket; data lives on S3.")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output directory (one run)")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Resolve API endpoint. Two paths:
    #   1. Azure Foundry — AZURE_FOUNDRY_KEY + base_url (default known-good
    #      URL matching the PoC if AZURE_FOUNDRY_ANTHROPIC_URL is unset)
    #   2. Direct Anthropic — ANTHROPIC_API_KEY, no base_url
    azure_key = os.environ.get("AZURE_FOUNDRY_KEY", "")
    direct_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if azure_key:
        api_key = azure_key
        # The PoC's default. SDK's base_url strips /v1/messages itself.
        azure_url = os.environ.get(
            "AZURE_FOUNDRY_ANTHROPIC_URL",
            "https://izzet-2249-resource.openai.azure.com/anthropic/v1/messages",
        )
        base_url = azure_url.split("/v1/messages")[0]
        if not base_url.endswith("/anthropic"):
            base_url = base_url.rstrip("/") + "/anthropic"
    elif direct_key:
        api_key = direct_key
        base_url = None
    else:
        print("FATAL: neither AZURE_FOUNDRY_KEY nor ANTHROPIC_API_KEY set",
              file=sys.stderr)
        return 2

    # Load workload + ruleset. S3 variant shares aiob_107's rules
    # (predictor rules match against thinking text + logical paths;
    # the data's physical location doesn't affect what the agent thinks).
    if args.workload == "aiob_107_s3":
        workload = load_aiob_107_s3()
    else:
        workload = load_aiob_107()
    ruleset = get_ruleset("aiob_107")
    print(f"workload: {workload.task_id}, prior buckets: {len(workload.workspace_prior)}, "
          f"total files in prior: {sum(len(v) for v in workload.workspace_prior.values())}",
          file=sys.stderr)

    # Translate logical→physical for the prior so the stager dispatches
    # real cold-tier paths. The predictor's rules match against logical
    # text in the thinking content, but the predicted_files list it
    # produces will use logical paths from the prior; we translate
    # before dispatching.
    prefix_map = workload.prefix_map
    def to_physical(logical: str) -> str:
        for lp, rp in prefix_map:
            if logical.startswith(lp):
                return rp + logical[len(lp):]
        return logical

    physical_prior = {
        k: tuple(to_physical(p) for p in v)
        for k, v in workload.workspace_prior.items()
    }

    # Set up stager
    hot_root = Path(os.environ.get("AGENTSTAGE_HOT_ROOT", "/dev/shm/agentstage_path_a"))
    cold_root = Path(os.environ.get(
        "AGENTSTAGE_COLD_ROOTS",
        "/mnt/common/datasets-staging/agentiobench/datasets",
    ).split(":")[0])
    hot_root.mkdir(parents=True, exist_ok=True)
    report = StagingReport()
    stager = Stager(
        hot_root=hot_root,
        cold_roots=[cold_root],
        max_workers=4,
        capacity_bytes=32 * 1024**3,
        report=report,
    )

    # Set up live client
    client = AnthropicClient(
        api_key=api_key,
        base_url=base_url,
        stager=stager,
        workspace_prior=physical_prior,
        ruleset=ruleset,
    )

    # Single-turn task prompt + planning prompt
    user_msg = TASK_USER_MSG + PLANNING_PROMPT_SUFFIX
    messages = [{"role": "user", "content": user_msg}]
    system = "You are a careful scientific computing agent."

    # Stream the call
    print(f"calling {args.model} (budget {args.budget}, base_url={base_url or 'default'})...",
          file=sys.stderr)
    stream_log: list[dict] = []
    t0 = time.monotonic_ns()

    # Declare tools so Haiku emits a tool_use block. Without this, the model
    # responds text-only (which is fine for the streaming smoke but doesn't
    # give us the tool_use we need to identify the target file).
    tools = [{
        "name": "open_file",
        "description": "Open and read the contents of a single file under /data/.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }, {
        "name": "list_dir",
        "description": "List the contents of a directory under /data/.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    }]

    # max_tokens must exceed thinking_budget (Anthropic API constraint).
    max_tokens = args.budget + 4096
    response = client.stream(
        model=args.model,
        messages=messages,
        max_tokens=max_tokens,
        thinking_budget=args.budget,
        temperature=1.0,
        system=system,
        # tool_choice can't be forced when thinking is enabled per
        # Anthropic API constraint. Trust the model to emit a tool_use
        # given the prompt explicitly says "produce a tool_use block".
        extra_body={"tools": tools},
    )

    first_tool_use_event = None
    for event in response.events():
        # Log every event for debugging (lightweight)
        etype = getattr(event, "type", None)
        stream_log.append({
            "t_ms": (time.monotonic_ns() - t0) / 1e6,
            "type": etype,
        })
        # Stop after we get the first tool_use (single-turn smoke)
        if etype == "content_block_start":
            cb_type = getattr(event.content_block, "type", None)
            if cb_type == "tool_use":
                first_tool_use_event = event
                print(f"  [t={(time.monotonic_ns() - t0)/1e6:.0f}ms] "
                      f"tool_use {event.content_block.name}",
                      file=sys.stderr)
                # Let the SDK finish the block so the input_json is complete,
                # then we'll close the stream
        elif etype == "message_stop":
            break

    session = response.session
    print(f"  fired {len(session.fired_rule_names)} rules during thinking",
          file=sys.stderr)
    print(f"  slack_ms: {session.slack_ms:.0f}" if session.slack_ms else "  no slack",
          file=sys.stderr)
    print(f"  tool_calls: {len(session.tool_calls)}", file=sys.stderr)

    # Identify the file to measure. Prefer agent's first open_file/file-target
    # tool call; otherwise fall back to the FIRST file the stager staged
    # (which is what the predictor's tier-1 rule pointed at).
    target_logical: str | None = None
    target_physical: str | None = None
    target_source = "none"
    if session.tool_calls:
        for tc in session.tool_calls:
            try:
                tool_input = json.loads(tc.input_json) if tc.input_json else {}
            except json.JSONDecodeError:
                tool_input = {}
            candidate = tool_input.get("path")
            if not candidate:
                continue
            cand_phys = to_physical(candidate)
            if Path(cand_phys).is_file():
                target_logical = candidate
                target_physical = cand_phys
                target_source = f"tool_use:{tc.name}"
                print(f"  target from tool_use: {tc.name}({candidate!r})",
                      file=sys.stderr)
                break
    if target_physical is None and report.events:
        # Fall back to first successfully-staged file
        for ev in report.events:
            if ev.outcome in ("staged", "hit") and Path(ev.cold_path).is_file():
                target_physical = ev.cold_path
                target_logical = ev.cold_path  # already physical
                target_source = "stager:first_staged"
                print(f"  target from stager: {ev.cold_path}", file=sys.stderr)
                print(f"  (LLM tool_uses were: "
                      f"{[(t.name, t.input_json) for t in session.tool_calls]})",
                      file=sys.stderr)
                break

    # Measure first-read latency on the target file
    measurements: dict = {}
    if target_physical and Path(target_physical).is_file():
        # Hot read (via shim, file should be staged from the live run)
        # First check: is it actually staged?
        was_staged = stager.is_staged(target_physical)
        print(f"  target_physical: {target_physical}", file=sys.stderr)
        print(f"  was_staged at time of tool_use: {was_staged}", file=sys.stderr)

        # If not staged, force-prefetch and wait (so we can still measure
        # the with-stager case)
        if not was_staged:
            from agentstage.stager import DataHint
            print("  force-prefetching target for measurement...", file=sys.stderr)
            futures = stager.prefetch(DataHint(
                predicted_files=(target_physical,),
                tier=1,
                fired_at_ms=0.0,
                rule_id="path_a_force",
            ))
            for f in futures:
                f.result(timeout=60)

        # Evict cold + measure hot via shim
        evict_cold_file(Path(target_physical))
        th0 = time.monotonic_ns()
        with open(target_physical, "rb") as f:
            f.read(4096)
        hot_ms = (time.monotonic_ns() - th0) / 1e6

        # Measure cold via shim-disable
        os.environ["AGENTSTAGE_SHIM_DISABLE"] = "1"
        evict_cold_file(Path(target_physical))
        # IMPORTANT: AGENTSTAGE_SHIM_DISABLE is read once at first call by
        # the shim's pthread_once-guarded cfg_init. Setting it now AFTER
        # the shim is already loaded won't disable it. So we measure cold
        # via subprocess instead.
        import subprocess
        env_no_shim = os.environ.copy()
        env_no_shim.pop("LD_PRELOAD", None)
        env_no_shim["AGENTSTAGE_SHIM_DISABLE"] = "1"
        r = subprocess.run(
            ["python3", "-c",
             f"import os, time; "
             f"fd=os.open({target_physical!r}, os.O_RDONLY); "
             f"os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); "
             f"import os; "
             f"t0=time.monotonic_ns(); "
             f"open({target_physical!r},'rb').read(4096); "
             f"print((time.monotonic_ns()-t0)/1e6)"],
            env=env_no_shim, capture_output=True, text=True, timeout=30,
        )
        cold_ms = float(r.stdout.strip()) if r.returncode == 0 else None

        measurements = {
            "target_logical": target_logical,
            "target_physical": target_physical,
            "target_source": target_source,
            "size_bytes": Path(target_physical).stat().st_size,
            "was_staged_at_tool_use": was_staged,
            "hot_read_ms": round(hot_ms, 3),
            "cold_read_ms": round(cold_ms, 3) if cold_ms is not None else None,
            "speedup": round(cold_ms / hot_ms, 1) if cold_ms and hot_ms > 0 else None,
        }
    else:
        print(f"  could not measure: no target_physical or file missing "
              f"(target_logical={target_logical}, target_physical={target_physical})",
              file=sys.stderr)
        measurements = {"error": "no_valid_target",
                        "target_logical": target_logical,
                        "target_physical": target_physical}

    # Persist artifacts
    summary = {
        "model": args.model,
        "budget": args.budget,
        "base_url": base_url,
        "task": workload.task_id,
        "fired_rules": sorted(session.fired_rule_names),
        "n_fired_rules": len(session.fired_rule_names),
        "slack_ms": session.slack_ms,
        "first_thinking_chunk_at_ms": session.first_thinking_chunk_at_ms,
        "first_tool_use_at_ms": session.first_tool_use_at_ms,
        "tool_calls": [
            {"name": tc.name, "input_json": tc.input_json, "id": tc.id}
            for tc in session.tool_calls
        ],
        "measurements": measurements,
        "staging_report": report.summary(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "staging_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    (args.out / "stream_log.json").write_text(json.dumps(stream_log, indent=2))
    print(f"\nResults written to {args.out}", file=sys.stderr)

    # Headline
    if measurements.get("hot_read_ms") is not None and measurements.get("cold_read_ms") is not None:
        print(f"\n  hot_read:  {measurements['hot_read_ms']:.3f} ms", file=sys.stderr)
        print(f"  cold_read: {measurements['cold_read_ms']:.3f} ms", file=sys.stderr)
        print(f"  speedup:   {measurements['speedup']}x", file=sys.stderr)

    stager.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
