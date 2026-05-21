"""Path B — multi-turn live Haiku capture runner.

Extends Path A (single-turn smoke) into a real agent loop:
  - up to --max-turns assistant turns
  - executes list_dir / open_file / read_file tool calls between turns
  - feeds tool_results back to the assistant on the next turn
  - records per-turn stream.jsonl, tool_use.jsonl, tool_result.jsonl,
    wall_clock.jsonl in outputs/multi_turn/<run>/turns/turn_NN/
  - drives the detector through SessionDetector (multi-turn stateful)
  - the stager + shim engage live as in Path A

Used for:
  E-011: hinted multi-turn baseline capture (--prompt-mode hinted)
  E-014: sparse-prompt capture for Regime B replay (--prompt-mode sparse)
  E-015: sparse live end-to-end (same as E-014 but with measurement step)

Usage:

  LD_PRELOAD=$SHIM \\
  AGENTSTAGE_HOT_ROOT=/dev/shm/agentstage_path_b \\
  AGENTSTAGE_COLD_ROOTS=/mnt/common/datasets-staging/agentiobench/datasets \\
  AZURE_FOUNDRY_KEY=$KEY \\
  python -m agentstage.runners.path_b_multiturn \\
      --workload aiob_107_s3 \\
      --prompt-mode sparse \\
      --max-turns 8 \\
      --out outputs/multi_turn/<ts>/
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

from agentstage.client.anthropic import AnthropicClient
from agentstage.client.gemini import GeminiClient
from agentstage.detector.engine import StreamBlock
from agentstage.detector.rules import get_ruleset
from agentstage.detector.session import SessionDetector
from agentstage.stager import DataHint, Stager, StagingReport
from agentstage.workloads.aiob import (
    load_aiob_107,
    load_aiob_107_s3,
    load_aiob_110,
)

# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

HINTED_PROMPT_AIOB_107 = """\
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
"""

SPARSE_PROMPT_AIOB_107 = """\
Task: aiob_107_meteorology_goes_cmi_composites

Analyze the staged GOES-16 ABI-L2 weather satellite data under
/data/goes_cmi_composites/. The dataset spans a 7-day window in May 2024.

Extract brightness-temperature time series at several US city locations
for each available channel band that the dataset provides, computing the
diurnal cycle.

Save:
  - /repo/result/goes_cmi_timeseries.csv
  - /repo/result/goes_cmi_point_timeseries.png
  - /repo/result/report.md

Use the available tools to explore the workspace and the data files
before deciding how to process them. The dataset's layout, file count,
and per-file structure are not given to you upfront — discover them.
"""

PATHFUL_PROMPTS: dict[str, str] = {
    # V1 (original, E-020): soft instruction, permits "use list_dir first
    # then name paths" — LLM responded with templates ('M6C{08,09,10}').
    "v1": (
        "\n\nIMPORTANT: When reasoning about which files you intend to "
        "access, write the FULL absolute path of each file in your "
        "thinking. Example: 'I will open /data/foo/bar.csv next' rather "
        "than 'I will open the CSV file'. List every file you plan to "
        "read so the data-staging system can pre-fetch them. If you do "
        "not yet know the exact paths, first use list_dir to discover "
        "them; then in subsequent reasoning name them by full path."
    ),

    # V2: explicit anti-template, with bad/good examples. Targets the
    # exact failure mode V1 produced ('M6C{08,09,10}', '<timestamp>').
    "v2": (
        "\n\n"
        "## CRITICAL — How to write file paths\n"
        "\n"
        "A data-staging system reads your reasoning and pre-fetches the "
        "files you name BEFORE your tools run. Pre-fetch works ONLY by "
        "EXACT PATH MATCH against the filesystem. Templated paths cannot "
        "be matched and are useless to it.\n"
        "\n"
        "Rules:\n"
        "1. When you intend to read a file, write its FULL ABSOLUTE PATH "
        "exactly as it exists on disk. No abbreviations.\n"
        "2. NEVER use placeholders, wildcards, braces, or template "
        "variables. Specifically forbidden:\n"
        "   - brace expansion:  /data/foo_{a,b,c}.nc\n"
        "   - wildcards:        /data/foo_*.nc, /data/foo_?.nc\n"
        "   - template vars:    /data/<timestamp>.nc, /data/[N].nc, "
        "/data/YYYY/MM/file.nc\n"
        "   - ellipses:         /data/foo_001.nc, /data/foo_002.nc, ...\n"
        "3. After any list_dir result, your next reasoning MUST "
        "enumerate the concrete files you discovered that you plan to "
        "act on — write each one on its own line.\n"
        "4. If you would otherwise write a template covering many files, "
        "instead pick the SPECIFIC FILE you plan to inspect next and "
        "write that one concrete path.\n"
        "\n"
        "Examples:\n"
        "BAD : I'll open the C{08,09,10} files in /data/raw/2024/122/00/\n"
        "BAD : Read OR_ABI-L2-CMIPC-M6C*_G16_*.nc\n"
        "GOOD: I'll open /data/raw/2024/122/00/OR_ABI-L2-CMIPC-M6C08_G16_s20241220001170_e20241220003543_c20241220004042.nc first\n"
    ),

    # V4: most directive — require copying specific filenames from the
    # most recent tool_result into NEXT_FILES, with a worked example.
    # Targets the V3 failure mode where the LLM produced an empty block.
    "v4": (
        "\n\n"
        "## File-staging contract (mandatory)\n"
        "\n"
        "Your reasoning is read by an automated data-staging system that "
        "pre-fetches files into a fast tier BEFORE your tool calls run. "
        "Pre-fetch needs EXACT FILE PATHS — strings that exist on the "
        "filesystem verbatim.\n"
        "\n"
        "RULE: After EVERY list_dir result that returned a non-empty "
        "directory listing, your next response MUST contain this block "
        "BEFORE any tool_use:\n"
        "\n"
        "    NEXT_FILES:\n"
        "    <concrete path 1>\n"
        "    <concrete path 2>\n"
        "    ...\n"
        "\n"
        "The block MUST contain at least one path COPIED VERBATIM from "
        "the most recent tool_result, and it MUST be a path you intend "
        "to read next. Empty NEXT_FILES blocks are not acceptable when "
        "you have just received a directory listing.\n"
        "\n"
        "FORBIDDEN: templates (`*`, `{a,b,c}`, `<timestamp>`), "
        "abbreviations (`the C08 file`), or ellipses (`...`).\n"
        "\n"
        "Worked example:\n"
        "  tool_result for list_dir(/data/raw/2024/122/00): \n"
        "    FILE  OR_ABI-L2-CMIPC-M6C08_G16_s20241220001170_e20241220003543_c20241220004042.nc  (2960235 bytes)\n"
        "    FILE  OR_ABI-L2-CMIPC-M6C09_G16_s20241220001170_e20241220003543_c20241220004029.nc  (1234567 bytes)\n"
        "    ... and so on ...\n"
        "  your next response:\n"
        "    NEXT_FILES:\n"
        "    /data/raw/2024/122/00/OR_ABI-L2-CMIPC-M6C08_G16_s20241220001170_e20241220003543_c20241220004042.nc\n"
        "    /data/raw/2024/122/00/OR_ABI-L2-CMIPC-M6C09_G16_s20241220001170_e20241220003543_c20241220004029.nc\n"
        "    (then call open_file on the first one)\n"
    ),

    # V3: structured response format. Asks the LLM to mark a specific
    # section after each tool result with the literal paths it intends
    # to read.
    "v3": (
        "\n\n"
        "## File-staging contract\n"
        "\n"
        "After EVERY tool result, structure your next reasoning to "
        "include this exact block somewhere before your next tool_use:\n"
        "\n"
        "    NEXT_FILES:\n"
        "    /full/path/to/file1\n"
        "    /full/path/to/file2\n"
        "    (etc.)\n"
        "\n"
        "Rules for this block:\n"
        "- Each line is ONE concrete absolute path that exists on disk.\n"
        "- NO templates, NO wildcards, NO braces, NO placeholders.\n"
        "- If you don't yet have any concrete paths (e.g. before your "
        "first list_dir), write an empty block: 'NEXT_FILES:' with no "
        "lines after.\n"
        "- The block is consumed by an automated staging system that "
        "pre-fetches the listed files. Templates cannot be matched.\n"
        "\n"
        "Example:\n"
        "    NEXT_FILES:\n"
        "    /data/goes_cmi_composites/raw/2024/122/00/OR_ABI-L2-CMIPC-M6C08_G16_s20241220001170_e20241220003543_c20241220004042.nc\n"
        "    /data/goes_cmi_composites/raw/2024/122/00/OR_ABI-L2-CMIPC-M6C09_G16_s20241220001170_e20241220003543_c20241220004029.nc\n"
    ),
}


PLANNING_PROMPT_SUFFIX = """

Before issuing any tool call, think step-by-step about:
  (1) what you know vs. need to learn,
  (2) the smallest discovery step you should take first,
  (3) what files you ultimately need to read.

Use the tools (list_dir / read_file / open_file) iteratively. The task
is complete when you have produced enough information that you could
write the analysis script.
"""

# ---------------------------------------------------------------------------
# Tool execution
# ---------------------------------------------------------------------------


def _safe_dataset_dir(workload_id: str) -> str:
    if workload_id == "aiob_107_s3":
        return os.environ.get("AGENTSTAGE_COLD_ROOTS",
                              "/tmp/s3-noaa-goes16").split(":")[0]
    return os.environ.get("AGENTSTAGE_COLD_ROOTS",
                          "/mnt/common/datasets-staging/agentiobench/datasets").split(":")[0]


def _resolve_logical_to_physical(
    path: str,
    prefix_map: tuple[tuple[str, str], ...],
    *,
    cold_root: str | None = None,
) -> str:
    """Translate a logical path to its physical location.

    Rules (in order):
      1. Exact-match: any logical prefix in `prefix_map` → its real prefix.
      2. If `cold_root` is provided, treat `/data` as an alias for it:
           /data            → cold_root
           /data/<subpath>  → cold_root/<subpath> (after stripping the
             workload-subdir-known prefix if it's redundant)
      3. Otherwise return path unchanged (caller checks existence).
    """
    for lp, rp in prefix_map:
        if path.startswith(lp):
            return rp + path[len(lp):]
    if cold_root and (path == "/data" or path.startswith("/data/")):
        rel = path[len("/data"):].lstrip("/")
        # Strip any known workload-subdir prefix that the agent might
        # include (e.g. /data/goes_cmi_composites/raw/2024 should map
        # to <cold_root>/2024 if the workload's prefix_map already
        # tells us /data/goes_cmi_composites/raw/ → <cold_root>/).
        for lp, _rp in prefix_map:
            lp_rel = lp[len("/data/"):].rstrip("/")
            if rel.startswith(lp_rel + "/"):
                rel = rel[len(lp_rel) + 1:]
                break
            if rel == lp_rel:
                rel = ""
                break
        return cold_root.rstrip("/") + ("/" + rel if rel else "")
    return path


_ALLOWED_PHYSICAL_PREFIXES: tuple[str, ...] = (
    "/tmp/s3-",                                  # mountpoint-s3 mount
    "/mnt/common/datasets-staging/agentiobench",  # local AIOB dataset root
    "/dev/shm/agentstage",                       # hot tier (sanity)
)


def _physical_is_allowed(phys: str) -> bool:
    return any(phys.startswith(p) for p in _ALLOWED_PHYSICAL_PREFIXES)


def enrich_prior_from_tool_result(
    prior: dict[str, tuple[str, ...] | list[str]],
    tool_result_text: str,
    *,
    bucket_name: str = "discovered",
    file_extensions: tuple[str, ...] = (".nc", ".csv", ".parquet", ".json",
                                         ".h5", ".hdf5", ".tsv", ".txt",
                                         ".md", ".bam", ".vcf", ".nwb",
                                         ".npy", ".npz", ".pkl"),
) -> int:
    """Parse a tool_result listing produced by execute_tool's list_dir
    branch and add discovered concrete file paths to the workspace
    prior under ``bucket_name``.

    Mutates ``prior`` in place. Returns the number of NEW paths added.

    The expected text format (produced by execute_tool above) is:

        # Listing of <PARENT> (... total entries; showing ...):
          FILE  <PARENT>/<filename>  (<size> bytes)
          DIR   <PARENT>/<subdir>/
          ...

    where ``<PARENT>`` is a logical path the LLM understands. Adding
    discovered files to the prior lets hot_path_scan match against the
    paths the LLM writes after discovery — closing the
    workload-spec-vs-actual-filesystem gap that defeated V4 sparse
    pathful prompts.
    """
    import re as _re
    new_paths: list[str] = []
    # Match lines of the form "  FILE  <path>  (<size> bytes)"
    # The path is everything between two-space-separated tokens; allow it
    # to contain slashes, dashes, dots, underscores, digits, letters, plus.
    line_pat = _re.compile(r"^\s*FILE\s+(\S+)\s+\(\d+\s+bytes\)\s*$",
                            _re.MULTILINE)
    for match in line_pat.finditer(tool_result_text):
        candidate = match.group(1)
        # Only count entries that look like real paths (have a recognized
        # extension and at least one slash) — avoid metadata noise
        if "/" not in candidate:
            continue
        if not any(candidate.endswith(ext) for ext in file_extensions):
            continue
        new_paths.append(candidate)
    if not new_paths:
        return 0
    existing = list(prior.get(bucket_name, ()))
    existing_set = set(existing)
    added = 0
    for p in new_paths:
        if p not in existing_set:
            existing.append(p)
            existing_set.add(p)
            added += 1
    if added > 0:
        prior[bucket_name] = tuple(existing)
    return added


def _synthesize_ancestor_listing(
    logical_path: str,
    prefix_map: tuple[tuple[str, str], ...],
) -> str | None:
    """If `logical_path` is a strict ANCESTOR of any prefix_map entry's
    logical prefix, synthesize a directory listing showing the next path
    component. This lets the agent navigate from /data → /data/<workload>
    → /data/<workload>/raw → real data, matching AIOB's container layout.

    Returns None if `logical_path` is not an ancestor of any LP.
    """
    path_norm = logical_path.rstrip("/")
    next_components: set[str] = set()
    for lp, _rp in prefix_map:
        lp_norm = lp.rstrip("/")
        if lp_norm == path_norm:
            return None  # exact match; let physical listing handle it
        if lp_norm.startswith(path_norm + "/"):
            rest = lp_norm[len(path_norm) + 1:]
            next_components.add(rest.split("/")[0])
    if not next_components:
        return None
    lines = [f"# Listing of {logical_path} (synthesized from workload prefix_map):"]
    for c in sorted(next_components):
        lines.append(f"  DIR   {c}/")
    return "\n".join(lines)


def execute_tool(
    name: str,
    args: dict,
    *,
    prefix_map: tuple[tuple[str, str], ...],
    cold_root: str | None = None,
    max_bytes: int = 4096,
    max_entries: int = 100,
) -> str:
    """Run one tool call. Returns a string for the tool_result content.

    Supported tools:
      list_dir(path)     - list immediate children of a directory
      open_file(path)    - read first max_bytes bytes (text or binary preview)
      read_file(path)    - same as open_file but explicit name

    The tool is sandboxed: only paths that resolve to an allowed
    physical prefix (S3 mount, AIOB dataset root, or hot tier) are
    accepted; everything else returns a controlled ERROR so the agent
    can re-plan. This prevents the agent from listing host directories
    when it can't find /data.
    """
    path = args.get("path", "")
    if not path:
        return f"ERROR: {name}: missing 'path' argument"
    # Ancestor synthesis: if the agent lists a logical parent of the
    # workload's prefix_map (e.g. /data when LP=/data/goes_cmi.../raw/),
    # synthesize the next-level dir listing so the agent can drill down.
    if name == "list_dir":
        synth = _synthesize_ancestor_listing(path, prefix_map)
        if synth is not None:
            return synth
    phys = _resolve_logical_to_physical(path, prefix_map, cold_root=cold_root)
    if not _physical_is_allowed(phys):
        return (f"ERROR: {name}({path!r}): path outside permitted dataset "
                f"roots. Use /data/<dataset>/... to access the staged data.")
    p = Path(phys)
    # Use the LOGICAL path (what the LLM asked for) in the listing header
    # so the LLM continues to write in logical address space. This keeps
    # tool_result content matchable against the workspace_prior (which is
    # also in logical space) via hot_path_scan.
    display_dir = path.rstrip("/")
    try:
        if name == "list_dir":
            if not p.exists():
                return f"ERROR: list_dir: path does not exist: {phys}"
            if not p.is_dir():
                return f"ERROR: list_dir: not a directory: {phys}"
            entries = sorted(p.iterdir())[:max_entries]
            lines = [f"# Listing of {display_dir} ({len(list(p.iterdir()))} total entries; showing {len(entries)}):"]
            for e in entries:
                kind = "DIR " if e.is_dir() else "FILE"
                size = e.stat().st_size if e.is_file() else 0
                full_path = f"{display_dir}/{e.name}"
                if e.is_file():
                    lines.append(f"  {kind}  {full_path}  ({size} bytes)")
                else:
                    lines.append(f"  {kind}  {full_path}/")
            return "\n".join(lines)
        elif name in ("open_file", "read_file"):
            if not p.exists():
                return f"ERROR: {name}: file does not exist: {phys}"
            if not p.is_file():
                return f"ERROR: {name}: not a file: {phys}"
            size = p.stat().st_size
            # For binary files (NetCDF, etc.) we return a hex preview of the
            # first 256 bytes plus the size. For text-like files, return text.
            with open(phys, "rb") as f:
                head = f.read(min(max_bytes, size))
            try:
                text = head.decode("utf-8")
                return (f"# Contents of {path} (first {len(head)}/{size} bytes):\n"
                        f"{text}")
            except UnicodeDecodeError:
                return (f"# Binary file {path} (size {size} bytes). First 64 bytes hex:\n"
                        f"{head[:64].hex()}")
        else:
            return f"ERROR: unknown tool: {name}"
    except Exception as e:
        return f"ERROR: {name}: {e!r}"


TOOLS_SCHEMA = [
    {
        "name": "list_dir",
        "description": "List the immediate children of a directory under /data/. "
                       "Returns names, types (FILE/DIR), and sizes.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "open_file",
        "description": "Read the first few KB of a file under /data/. "
                       "Returns text content (or hex preview for binary).",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "read_file",
        "description": "Alias for open_file. Reads the start of a file.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
]


# ---------------------------------------------------------------------------
# Per-turn recorder
# ---------------------------------------------------------------------------


class TurnRecorder:
    """Records per-turn events. One subdirectory per turn."""

    def __init__(self, out_root: Path, turn: int) -> None:
        self.turn = turn
        self.turn_dir = out_root / "turns" / f"turn_{turn:02d}"
        self.turn_dir.mkdir(parents=True, exist_ok=True)
        self.stream_jsonl = open(self.turn_dir / "stream.jsonl", "w")
        self.events: list[dict] = []
        self.t_started_ms = time.monotonic() * 1000

    def record_event(self, event: dict) -> None:
        event["t_ms_in_turn"] = time.monotonic() * 1000 - self.t_started_ms
        self.events.append(event)
        self.stream_jsonl.write(json.dumps(event) + "\n")
        self.stream_jsonl.flush()

    def finalize(self, *, tool_uses: list[dict], tool_results: list[dict],
                 thinking_text: str, fired_rules: list[str]) -> None:
        (self.turn_dir / "tool_use.jsonl").write_text(
            "\n".join(json.dumps(tu) for tu in tool_uses) + "\n" if tool_uses else "")
        (self.turn_dir / "tool_result.jsonl").write_text(
            "\n".join(json.dumps(tr) for tr in tool_results) + "\n" if tool_results else "")
        (self.turn_dir / "thinking.txt").write_text(thinking_text)
        (self.turn_dir / "summary.json").write_text(json.dumps({
            "turn": self.turn,
            "n_events": len(self.events),
            "n_tool_uses": len(tool_uses),
            "thinking_chars": len(thinking_text),
            "fired_rules_on_turn": fired_rules,
            "duration_ms": time.monotonic() * 1000 - self.t_started_ms,
        }, indent=2))
        self.stream_jsonl.close()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="claude-haiku-4-5")
    parser.add_argument("--budget", type=int, default=4096,
                        help="thinking_budget per turn (default 4096)")
    parser.add_argument("--workload",
                        choices=["aiob_107", "aiob_107_s3", "aiob_110"],
                        default="aiob_107_s3")
    parser.add_argument("--prompt-mode", choices=["hinted", "sparse"],
                        default="hinted",
                        help="hinted: original full task prompt; "
                             "sparse: I/O hints stripped (Regime B)")
    parser.add_argument("--pathful-prompt", action="store_true",
                        help="Inject a system-prompt clause asking the model "
                             "to write full file paths in its thinking. Enables "
                             "the literal-path detection mode (hot_path_scan), "
                             "potentially replacing all hand-coded regex rules. "
                             "Set up by E-020 ablation.")
    parser.add_argument("--pathful-version",
                        choices=list(PATHFUL_PROMPTS.keys()),
                        default="v2",
                        help="Which version of the pathful prompt to inject. "
                             "v1 = original (soft); v2 = explicit anti-template; "
                             "v3 = structured NEXT_FILES block. Default v2.")
    parser.add_argument("--max-turns", type=int, default=8,
                        help="Maximum assistant turns before giving up (default 8)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--measure-target-after", action="store_true",
                        help="After loop ends, evict + measure cold/hot read "
                             "on the first file the agent opened. Mirrors "
                             "Path A measurement step.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)

    # Determine provider from model name (claude-* → anthropic; gemini-* → gemini)
    provider = "gemini" if args.model.lower().startswith("gemini") else "anthropic"

    # Resolve API endpoint by provider
    if provider == "anthropic":
        azure_key = os.environ.get("AZURE_FOUNDRY_KEY", "")
        direct_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if azure_key:
            api_key = azure_key
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
    else:  # gemini
        gemini_key = os.environ.get("GOOGLE_GEMINI_API_KEY", "")
        if not gemini_key:
            print("FATAL: GOOGLE_GEMINI_API_KEY not set for Gemini provider",
                  file=sys.stderr)
            return 2
        api_key = gemini_key
        base_url = None

    # Load workload + ruleset
    loaders = {
        "aiob_107": load_aiob_107,
        "aiob_107_s3": load_aiob_107_s3,
        "aiob_110": load_aiob_110,
    }
    workload = loaders[args.workload]()
    rules_key = args.workload.replace("_s3", "")
    ruleset = get_ruleset(rules_key)
    prefix_map = workload.prefix_map

    physical_prior = {
        k: tuple(_resolve_logical_to_physical(p, prefix_map)
                 for p in v)
        for k, v in workload.workspace_prior.items()
    }

    # Set up stager
    hot_root = Path(os.environ.get("AGENTSTAGE_HOT_ROOT", "/dev/shm/agentstage_path_b"))
    cold_root = Path(_safe_dataset_dir(args.workload))
    hot_root.mkdir(parents=True, exist_ok=True)
    report = StagingReport()
    stager = Stager(
        hot_root=hot_root,
        cold_roots=[cold_root],
        max_workers=4,
        capacity_bytes=32 * 1024**3,
        report=report,
    )

    # Set up session detector (multi-turn, tool_result-aware).
    # Use the LOGICAL prior here: LLM-written paths in thinking/text are
    # in logical form (e.g. /data/goes_cmi_composites/raw/.../C08.nc).
    # If we used the physical prior (/tmp/s3-noaa-goes16/...) hot_path_scan
    # would never match what the LLM writes. Dispatch sites below translate
    # logical → physical right before calling stager.prefetch.
    session_pred = SessionDetector(prior=workload.workspace_prior,
                                   ruleset=ruleset)

    # Set up live client based on provider. Both expose the same .stream()
    # interface and yield SDK-shaped events (Gemini events are wrapped to
    # match Anthropic's event shape, see client/gemini.py:_Event).
    if provider == "anthropic":
        client = AnthropicClient(
            api_key=api_key,
            base_url=base_url,
            stager=None,
            workspace_prior=physical_prior,
            ruleset=ruleset,
        )
    else:  # gemini
        client = GeminiClient(
            api_key=api_key,
            stager=None,
            workspace_prior=physical_prior,
            ruleset=ruleset,
        )
    print(f"  provider: {provider}, model: {args.model}", file=sys.stderr)

    # Build the initial user message
    if args.workload in ("aiob_107", "aiob_107_s3"):
        if args.prompt_mode == "sparse":
            base_prompt = SPARSE_PROMPT_AIOB_107
        else:
            base_prompt = HINTED_PROMPT_AIOB_107
    else:
        # For aiob_110 we just paraphrase from task_inst; can refine later
        base_prompt = f"Task: {workload.task_id}\n\n{workload.task.task_inst}"

    initial_user_msg = base_prompt + PLANNING_PROMPT_SUFFIX
    messages: list[dict] = [{"role": "user", "content": initial_user_msg}]
    system = "You are a careful scientific computing agent."
    if args.pathful_prompt:
        system += PATHFUL_PROMPTS[args.pathful_version]
        print(f"  pathful prompt version: {args.pathful_version} "
              f"({len(PATHFUL_PROMPTS[args.pathful_version])} chars)",
              file=sys.stderr)

    print(f"workload: {workload.task_id} ({args.prompt_mode} prompt mode)",
          file=sys.stderr)
    print(f"max_turns: {args.max_turns}", file=sys.stderr)
    print(f"physical_prior: {sum(len(v) for v in physical_prior.values())} files "
          f"across {len(physical_prior)} buckets", file=sys.stderr)

    all_tool_uses_executed: list[dict] = []
    first_target_physical: str | None = None
    t0 = time.monotonic()

    for turn in range(args.max_turns):
        print(f"\n=== Turn {turn} ===", file=sys.stderr)
        recorder = TurnRecorder(args.out, turn)

        # Stream the call
        t_call_start_ms = (time.monotonic() - t0) * 1000
        response = client.stream(
            model=args.model,
            messages=messages,
            max_tokens=args.budget + 4096,
            thinking_budget=args.budget,
            temperature=1.0,
            system=system,
            extra_body={"tools": TOOLS_SCHEMA},
        )

        # Collect events to rebuild the assistant's content for next-turn input
        thinking_text_parts: list[str] = []
        text_parts: list[str] = []
        tool_uses_this_turn: list[dict] = []
        signature_by_idx: dict[int, str] = {}  # for thinking blocks
        assistant_blocks: list[dict] = []  # for messages.append

        # Block-idx → kind, and accumulators
        block_kind: dict[int, str] = {}
        thinking_by_idx: dict[int, list[str]] = {}
        text_by_idx: dict[int, list[str]] = {}
        tool_use_by_idx: dict[int, dict] = {}

        for event in response.events():
            etype = getattr(event, "type", None)
            evt_rec = {"t_ms": (time.monotonic() - t0) * 1000 - t_call_start_ms,
                       "type": etype}
            if etype == "content_block_start":
                idx = event.index
                block = event.content_block
                kind = getattr(block, "type", None)
                block_kind[idx] = kind
                evt_rec.update({"block_idx": idx, "block_type": kind})
                if kind == "tool_use":
                    tu = {
                        "block_idx": idx,
                        "name": getattr(block, "name", ""),
                        "id": getattr(block, "id", ""),
                        "input_json": "",
                    }
                    tool_use_by_idx[idx] = tu
                    evt_rec["tool_name"] = tu["name"]
                elif kind == "thinking":
                    thinking_by_idx[idx] = []
                elif kind == "text":
                    text_by_idx[idx] = []
            elif etype == "content_block_delta":
                idx = event.index
                delta = event.delta
                dtype = getattr(delta, "type", None)
                evt_rec.update({"block_idx": idx, "delta_type": dtype})
                if dtype == "thinking_delta":
                    piece = getattr(delta, "thinking", "")
                    thinking_by_idx.setdefault(idx, []).append(piece)
                    evt_rec["chunk"] = piece[:200]  # truncate to keep jsonl small
                elif dtype == "text_delta":
                    piece = getattr(delta, "text", "")
                    text_by_idx.setdefault(idx, []).append(piece)
                    evt_rec["chunk"] = piece[:200]
                elif dtype == "input_json_delta":
                    pj = getattr(delta, "partial_json", "")
                    if idx in tool_use_by_idx:
                        tool_use_by_idx[idx]["input_json"] += pj
                    evt_rec["chunk"] = pj[:200]
                elif dtype == "signature_delta":
                    sig = getattr(delta, "signature", "")
                    signature_by_idx[idx] = signature_by_idx.get(idx, "") + sig
            elif etype == "content_block_stop":
                evt_rec["block_idx"] = event.index
            elif etype == "message_stop":
                pass
            recorder.record_event(evt_rec)

        # Reconstruct assistant message blocks in order
        # (we trust the SDK's index order to match arrival order)
        for idx in sorted(block_kind.keys()):
            kind = block_kind[idx]
            if kind == "thinking":
                t_text = "".join(thinking_by_idx.get(idx, []))
                blk = {"type": "thinking", "thinking": t_text}
                if idx in signature_by_idx:
                    blk["signature"] = signature_by_idx[idx]
                assistant_blocks.append(blk)
                thinking_text_parts.append(t_text)
            elif kind == "text":
                t_text = "".join(text_by_idx.get(idx, []))
                if t_text:
                    assistant_blocks.append({"type": "text", "text": t_text})
                    text_parts.append(t_text)
            elif kind == "tool_use":
                tu = tool_use_by_idx[idx]
                try:
                    parsed_input = json.loads(tu["input_json"]) if tu["input_json"] else {}
                except json.JSONDecodeError:
                    parsed_input = {}
                assistant_blocks.append({
                    "type": "tool_use",
                    "id": tu["id"],
                    "name": tu["name"],
                    "input": parsed_input,
                })
                tool_uses_this_turn.append({**tu, "parsed_input": parsed_input})

        thinking_text = "".join(thinking_text_parts)

        # Feed this turn's blocks to the session detector.
        # We feed BOTH thinking AND text so multi-turn continuation
        # responses (which often skip thinking and emit visible text)
        # still produce activations.
        sp_blocks: list[StreamBlock] = []
        for blk in assistant_blocks:
            if blk["type"] == "thinking":
                sp_blocks.append(StreamBlock(
                    type="thinking",
                    t_first=t_call_start_ms,
                    t_stop=(time.monotonic() - t0) * 1000,
                    text=blk.get("thinking", ""),
                    chunks=1,
                ))
            elif blk["type"] == "text":
                sp_blocks.append(StreamBlock(
                    type="text",
                    t_first=t_call_start_ms,
                    t_stop=(time.monotonic() - t0) * 1000,
                    text=blk.get("text", ""),
                    chunks=1,
                ))
        new_acts = session_pred.feed_turn(sp_blocks)

        # Auto-dispatch tier-1 hints from THIS turn's new activations.
        # Detector returns LOGICAL paths (from workload.workspace_prior);
        # translate to PHYSICAL before stager.prefetch.
        dispatched_this_turn: list[str] = []
        for act in new_acts:
            n_files = len(set(act.detected_files))
            tier = 1 if n_files <= 10 else (2 if n_files <= 200 else 3)
            if tier > 1:
                continue
            phys_files = tuple(
                _resolve_logical_to_physical(p, prefix_map, cold_root=str(cold_root))
                for p in act.detected_files
            )
            hint = DataHint(
                detected_files=phys_files,
                tier=tier,
                fired_at_ms=act.fired_at_ms or 0.0,
                rule_id=f"turn{turn}:{act.rule_name}",
                byte_estimate=0,
            )
            stager.prefetch(hint)
            dispatched_this_turn.append(act.rule_name)

        # Pathful-prompt path: dispatch any literal-path hits the
        # session detector has discovered. This runs alongside the rule
        # dispatch above — a file mentioned literally AND matched by a
        # rule is fine (the stager dedupes on the cold_path key).
        dispatched_hot_paths: list[str] = []
        new_hot = session_pred.new_hot_paths()
        for logical_path in new_hot:
            phys = _resolve_logical_to_physical(
                logical_path, prefix_map, cold_root=str(cold_root))
            if not Path(phys).is_file():
                continue
            hint = DataHint(
                detected_files=(phys,),
                tier=1,
                fired_at_ms=new_hot[logical_path] or 0.0,
                rule_id=f"turn{turn}:hot_path",
                byte_estimate=0,
            )
            stager.prefetch(hint)
            dispatched_hot_paths.append(logical_path)

        print(f"  thinking: {len(thinking_text)} chars", file=sys.stderr)
        print(f"  tool_uses this turn: {len(tool_uses_this_turn)}", file=sys.stderr)
        print(f"  new rules fired: {[a.rule_name for a in new_acts]} "
              f"({sum(1 for a in new_acts if a.source=='thinking')} thinking, "
              f"{sum(1 for a in new_acts if a.source=='text')} text, "
              f"{sum(1 for a in new_acts if a.source=='tool_result')} tool_result)",
              file=sys.stderr)
        print(f"  dispatched tier-1 prefetches: {dispatched_this_turn}",
              file=sys.stderr)
        if dispatched_hot_paths:
            print(f"  dispatched literal-path prefetches: "
                  f"{len(dispatched_hot_paths)} files (e.g. "
                  f"{dispatched_hot_paths[0].split('/')[-1]})",
                  file=sys.stderr)

        # Append the assistant message
        messages.append({"role": "assistant", "content": assistant_blocks})

        # If no tool calls, the model is done (or stuck) — terminate
        if not tool_uses_this_turn:
            print("  no tool_use this turn — agent considers task complete or stuck.",
                  file=sys.stderr)
            recorder.finalize(
                tool_uses=tool_uses_this_turn,
                tool_results=[],
                thinking_text=thinking_text,
                fired_rules=[a.rule_name for a in new_acts],
            )
            break

        # Execute tools, collect tool_results
        tool_results_blocks: list[dict] = []
        tr_blocks_for_detector: list[StreamBlock] = []
        for tu in tool_uses_this_turn:
            result_text = execute_tool(
                tu["name"], tu["parsed_input"], prefix_map=prefix_map,
                cold_root=str(cold_root),
            )
            tool_results_blocks.append({
                "type": "tool_result",
                "tool_use_id": tu["id"],
                "content": result_text,
            })
            tr_blocks_for_detector.append(StreamBlock(
                type="tool_result",
                t_first=(time.monotonic() - t0) * 1000,
                t_stop=(time.monotonic() - t0) * 1000,
                text=result_text,
                chunks=1,
            ))
            # Track first concrete file target the agent opens (for measurement)
            if first_target_physical is None and tu["name"] in ("open_file", "read_file"):
                path = tu["parsed_input"].get("path", "")
                phys = _resolve_logical_to_physical(path, prefix_map, cold_root=str(cold_root))
                if Path(phys).is_file():
                    first_target_physical = phys

        # Append the tool_result message
        messages.append({"role": "user", "content": tool_results_blocks})

        # Dynamic prior enrichment: parse each tool_result for concrete
        # file paths and add them to the workspace prior under a
        # 'discovered' bucket. Closes the workspace-spec-vs-actual-
        # filesystem gap that defeated V4 sparse pathful prompts:
        # the agent in sparse mode picks bands outside the AIOB task
        # spec's 6042-file working set (C08-C10), but the discovered
        # bucket now contains those Band 01/02 paths from list_dir, so
        # hot_path_scan can match the LLM's NEXT_FILES emissions.
        n_enriched_this_turn = 0
        for tr_block in tr_blocks_for_detector:
            n_enriched_this_turn += enrich_prior_from_tool_result(
                workload.workspace_prior, tr_block.text,
            )
        if n_enriched_this_turn > 0:
            print(f"  enriched prior with {n_enriched_this_turn} new "
                  f"file paths from this turn's list_dir",
                  file=sys.stderr)

        # Feed tool_results to session detector (these stamp turn=current_turn)
        tr_acts = session_pred.feed_tool_results(tr_blocks_for_detector)
        # Dispatch new tier-1 hints from tool_result activations as well.
        # Same logical → physical translation as above.
        for act in tr_acts:
            n_files = len(set(act.detected_files))
            tier = 1 if n_files <= 10 else (2 if n_files <= 200 else 3)
            if tier > 1:
                continue
            phys_files = tuple(
                _resolve_logical_to_physical(p, prefix_map, cold_root=str(cold_root))
                for p in act.detected_files
            )
            hint = DataHint(
                detected_files=phys_files,
                tier=tier,
                fired_at_ms=act.fired_at_ms or 0.0,
                rule_id=f"turn{turn}_tr:{act.rule_name}",
                byte_estimate=0,
            )
            stager.prefetch(hint)
        if tr_acts:
            print(f"  tool_result fired NEW rules: "
                  f"{[a.rule_name for a in tr_acts]}",
                  file=sys.stderr)

        # Also dispatch any literal-path hits the tool_result revealed
        # (e.g. agent did `list_dir /raw/2024/122/00` and the result
        # text contains paths the LLM now knows literally).
        new_hot_tr = session_pred.new_hot_paths()
        for logical_path in new_hot_tr:
            phys = _resolve_logical_to_physical(
                logical_path, prefix_map, cold_root=str(cold_root))
            if not Path(phys).is_file():
                continue
            stager.prefetch(DataHint(
                detected_files=(phys,),
                tier=1,
                fired_at_ms=new_hot_tr[logical_path] or 0.0,
                rule_id=f"turn{turn}_tr:hot_path",
                byte_estimate=0,
            ))
        if new_hot_tr:
            print(f"  tool_result revealed literal paths: "
                  f"{len(new_hot_tr)} new files staged",
                  file=sys.stderr)

        all_tool_uses_executed.extend(tool_uses_this_turn)

        recorder.finalize(
            tool_uses=tool_uses_this_turn,
            tool_results=tool_results_blocks,
            thinking_text=thinking_text,
            fired_rules=[a.rule_name for a in new_acts + tr_acts],
        )

    print(f"\n=== Run complete ===", file=sys.stderr)
    print(f"total turns: {turn+1}", file=sys.stderr)
    print(f"total tool_uses: {len(all_tool_uses_executed)}", file=sys.stderr)
    print(f"total fired rules: {len(session_pred.fired_rule_names)}",
          file=sys.stderr)
    print(f"first agent-opened file: {first_target_physical}", file=sys.stderr)

    # Optional: measurement step on the first file the agent opened
    measurements: dict = {}
    if args.measure_target_after and first_target_physical:
        was_staged = stager.is_staged(first_target_physical)
        print(f"  target was_staged at end of run: {was_staged}", file=sys.stderr)
        if not was_staged:
            futures = stager.prefetch(DataHint(
                detected_files=(first_target_physical,),
                tier=1,
                fired_at_ms=0.0,
                rule_id="force",
            ))
            for f in futures:
                try:
                    f.result(timeout=120)
                except Exception as e:
                    print(f"  prefetch error: {e!r}", file=sys.stderr)

        # Evict and measure hot (via shim)
        fd = os.open(first_target_physical, os.O_RDONLY)
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        os.sync()
        th0 = time.monotonic_ns()
        with open(first_target_physical, "rb") as f:
            f.read(4096)
        hot_ms = (time.monotonic_ns() - th0) / 1e6

        # Cold via subprocess (shim disabled)
        env_no_shim = os.environ.copy()
        env_no_shim.pop("LD_PRELOAD", None)
        env_no_shim["AGENTSTAGE_SHIM_DISABLE"] = "1"
        r = subprocess.run(
            ["python3", "-c",
             f"import os, time; "
             f"fd=os.open({first_target_physical!r}, os.O_RDONLY); "
             f"os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd); "
             f"t0=time.monotonic_ns(); "
             f"open({first_target_physical!r},'rb').read(4096); "
             f"print((time.monotonic_ns()-t0)/1e6)"],
            env=env_no_shim, capture_output=True, text=True, timeout=120,
        )
        cold_ms = float(r.stdout.strip()) if r.returncode == 0 else None
        measurements = {
            "target_physical": first_target_physical,
            "size_bytes": Path(first_target_physical).stat().st_size,
            "was_staged_at_end": was_staged,
            "hot_read_ms": round(hot_ms, 3),
            "cold_read_ms": round(cold_ms, 3) if cold_ms is not None else None,
            "speedup": round(cold_ms / hot_ms, 1) if cold_ms and hot_ms > 0 else None,
        }
        print(f"  hot_read:  {hot_ms:.3f} ms", file=sys.stderr)
        if cold_ms is not None:
            print(f"  cold_read: {cold_ms:.3f} ms", file=sys.stderr)
            print(f"  speedup:   {measurements['speedup']}x", file=sys.stderr)

    # Persist summary
    summary = {
        "workload": workload.task_id,
        "prompt_mode": args.prompt_mode,
        "model": args.model,
        "budget": args.budget,
        "max_turns_allowed": args.max_turns,
        "turns_used": turn + 1,
        "total_tool_uses": len(all_tool_uses_executed),
        "fired_rule_names": sorted(session_pred.fired_rule_names),
        "activations": [
            {
                "rule_name": a.rule_name,
                "source": a.source,
                "turn": a.turn,
                "fired_at_ms": a.fired_at_ms,
                "n_detected_files": len(a.detected_files),
            }
            for a in session_pred.activations
        ],
        "first_target_physical": first_target_physical,
        "measurements": measurements,
        "staging_report": report.summary(),
    }
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    (args.out / "messages.json").write_text(json.dumps(messages, indent=2, default=str))
    (args.out / "staging_report.json").write_text(json.dumps(report.to_dict(), indent=2))
    print(f"\nResults written to {args.out}", file=sys.stderr)

    stager.shutdown(wait=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
