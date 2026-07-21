"""Detector engine — parses a stream.jsonl into blocks, replays thinking
text through the frozen rule library, and emits a tiered `Detection`.

This is the load-bearing port from `poc/probe_reasoning_slack.py`:
- `parse_anthropic_stream` ⇐ `block_timing_anthropic`
- `parse_gemini_stream` ⇐ `block_timing_gemini`
- `hot_path_scan` ⇐ `hot_path_scan`
- `run_detector` ⇐ `run_detector`

The engine is provider-aware (Anthropic vs Gemini SSE shapes differ) but
otherwise rule-library-agnostic — pass any `RuleSet` and it will fire
the regex matches against the thinking text in time order, then assemble
the tiered detected set.

Tiering rule (matches PoC §6.2):
  - target_keys resolve to ≤ 10 files → tier 1 (specific, immediate need)
  - ≤ 200 files → tier 2 (medium granularity)
  - > 200 files → tier 3 (broad, eventual working set)

Cumulative: tier_2 = tier_1 ∪ medium; tier_3 = tier_2 ∪ broad.

Ported on 2026-05-19.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from agentstage.detector.rules import (
    RULE_LIBRARY_HASH,
    RULE_LIBRARY_VERSION,
    Rule,
    RuleSet,
)

# ---------------------------------------------------------------------------
# Stream parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StreamBlock:
    """One thinking / text / tool-use / tool-result block.

    `t_first` and `t_stop` are the timestamps (in ms from urlopen start) of
    the first and last delta event for this block. `text` is the
    concatenated content; for tool_use blocks it's the JSON string of the
    arguments; for tool_result blocks it's the result text returned to
    the model on the next turn.

    Block types:
      - "thinking"     : extended thinking content (Anthropic thinking_delta)
      - "text"         : visible assistant text (text_delta)
      - "tool_use"     : assistant-emitted tool call (JSON args)
      - "tool_result"  : user-message tool_result content (next-turn input)
    """
    type: str
    t_first: float | None
    t_stop: float | None
    text: str
    chunks: int
    tool_name: str | None = None
    turn: int = 0  # Turn index (0 = first assistant turn). For multi-turn replay.


def parse_anthropic_stream(stream_path: Path) -> list[StreamBlock]:
    """Reconstruct blocks from an Anthropic SSE jsonl (content_block_*
    event family). Returns blocks in stream order."""
    blk: dict[int, dict] = {}
    order: list[int] = []
    with stream_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            e = r.get("event")
            idx = r.get("block_idx", -1)
            t = r.get("t_ms")
            if e == "content_block_start":
                blk[idx] = {
                    "type": r.get("block_type"),
                    "tool": r.get("tool_name"),
                    "t_first": None,
                    "t_stop": None,
                    "chunks": 0,
                    "text": "",
                }
                order.append(idx)
            elif e == "content_block_delta":
                b = blk.setdefault(
                    idx,
                    {"type": None, "tool": None, "t_first": None,
                     "t_stop": None, "chunks": 0, "text": ""},
                )
                if b["t_first"] is None:
                    b["t_first"] = t
                b["t_stop"] = t
                b["chunks"] += 1
                dt = r.get("delta_type")
                chunk = r.get("chunk") or ""
                if dt in ("thinking_delta", "text_delta", "input_json_delta"):
                    b["text"] += chunk
            elif e == "content_block_stop":
                b = blk.setdefault(idx, {})
                b["t_stop"] = t

    return [
        StreamBlock(
            type=blk[i].get("type") or "",
            t_first=blk[i].get("t_first"),
            t_stop=blk[i].get("t_stop"),
            text=blk[i].get("text") or "",
            chunks=blk[i].get("chunks") or 0,
            tool_name=blk[i].get("tool"),
        )
        for i in order
    ]


def parse_gemini_stream(stream_path: Path) -> list[StreamBlock]:
    """Reconstruct blocks from a Gemini SSE jsonl (thinking_delta /
    text_delta / function_call events at the top level)."""
    thinking = {"type": "thinking", "t_first": None, "t_stop": None,
                "chunks": 0, "text": "", "tool_name": None}
    text = {"type": "text", "t_first": None, "t_stop": None,
            "chunks": 0, "text": "", "tool_name": None}
    tool_calls: list[dict] = []
    with stream_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            e = r.get("event")
            t = r.get("t_ms")
            if e == "thinking_delta":
                if thinking["t_first"] is None:
                    thinking["t_first"] = t
                thinking["t_stop"] = t
                thinking["chunks"] += 1
                thinking["text"] += r.get("chunk") or ""
            elif e == "text_delta":
                if text["t_first"] is None:
                    text["t_first"] = t
                text["t_stop"] = t
                text["chunks"] += 1
                text["text"] += r.get("chunk") or ""
            elif e == "function_call":
                tool_calls.append({
                    "type": "tool_use",
                    "t_first": t, "t_stop": t,
                    "chunks": 1,
                    "text": json.dumps(r.get("args") or {}),
                    "tool_name": r.get("tool_name"),
                })

    blocks: list[StreamBlock] = []
    if thinking["text"]:
        blocks.append(StreamBlock(**thinking))
    if text["text"]:
        blocks.append(StreamBlock(**text))
    for tc in tool_calls:
        blocks.append(StreamBlock(**tc))
    return blocks


def blocks_from_messages(
    messages: list[dict],
    *,
    base_t_ms: float = 0.0,
    per_turn_ms: float = 1000.0,
) -> list[StreamBlock]:
    """Parse an Anthropic message history into StreamBlocks in chronological order.

    Used by multi-turn replay. Walks the message list:
      - assistant messages → thinking / text / tool_use blocks (turn=N)
      - user messages with tool_result content → tool_result blocks
        (turn=N, where N is the next assistant turn)

    Since the message-history view lacks per-chunk timestamps (only the
    live SDK stream has those), we synthesize block timings: each
    assistant turn N gets ``t_first = base_t_ms + N*per_turn_ms``,
    ``t_stop = base_t_ms + (N+1)*per_turn_ms``, so the relative ordering
    of activations across turns is preserved.

    For *single-turn* replay with real timings, prefer
    ``parse_anthropic_stream`` (which reads the live stream.jsonl with
    real ms-level event times).
    """
    blocks: list[StreamBlock] = []
    turn = 0
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content")
        if not content:
            continue
        # content may be a string (legacy) or a list of content blocks
        if isinstance(content, str):
            if role == "assistant":
                blocks.append(StreamBlock(
                    type="text",
                    t_first=base_t_ms + turn * per_turn_ms,
                    t_stop=base_t_ms + (turn + 1) * per_turn_ms,
                    text=content,
                    chunks=1,
                    turn=turn,
                ))
                turn += 1
            continue
        if role == "assistant":
            tf = base_t_ms + turn * per_turn_ms
            ts = base_t_ms + (turn + 1) * per_turn_ms
            for cb in content:
                cb_type = cb.get("type") if isinstance(cb, dict) else None
                if cb_type == "thinking":
                    blocks.append(StreamBlock(
                        type="thinking", t_first=tf, t_stop=ts,
                        text=cb.get("thinking", ""), chunks=1, turn=turn,
                    ))
                elif cb_type == "text":
                    blocks.append(StreamBlock(
                        type="text", t_first=tf, t_stop=ts,
                        text=cb.get("text", ""), chunks=1, turn=turn,
                    ))
                elif cb_type == "tool_use":
                    args = cb.get("input") or {}
                    import json as _json
                    blocks.append(StreamBlock(
                        type="tool_use", t_first=tf, t_stop=ts,
                        text=_json.dumps(args), chunks=1,
                        tool_name=cb.get("name"), turn=turn,
                    ))
            turn += 1
        elif role == "user":
            # tool_result content blocks feed the NEXT assistant turn
            tf = base_t_ms + turn * per_turn_ms - 0.5 * per_turn_ms
            ts = base_t_ms + turn * per_turn_ms
            for cb in content:
                cb_type = cb.get("type") if isinstance(cb, dict) else None
                if cb_type == "tool_result":
                    # content can be string or list of {type:"text",text:"..."}
                    result_content = cb.get("content", "")
                    if isinstance(result_content, list):
                        text_parts = [
                            sub.get("text", "")
                            for sub in result_content
                            if isinstance(sub, dict) and sub.get("type") == "text"
                        ]
                        text = "\n".join(text_parts)
                    else:
                        text = str(result_content)
                    blocks.append(StreamBlock(
                        type="tool_result", t_first=tf, t_stop=ts,
                        text=text, chunks=1, turn=turn,
                    ))
    return blocks


def parse_stream(stream_path: Path, provider: str | None = None) -> list[StreamBlock]:
    """Auto-dispatch by provider. If `provider` isn't given, peek at the
    first event to decide: `content_block_*` → Anthropic, otherwise Gemini.
    OpenRouter (DeepSeek-R1 in the PoC) uses an Anthropic-style block
    layout in the PoC's recorder so anthropic parsing works for it too.
    """
    if provider:
        p = provider.lower()
        if "anthropic" in p or "openrouter" in p or "azure" in p:
            return parse_anthropic_stream(stream_path)
        if "gemini" in p or "google" in p:
            return parse_gemini_stream(stream_path)
    # Auto-detect
    with stream_path.open() as f:
        for line in f:
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            e = r.get("event") or ""
            if e.startswith("content_block_"):
                return parse_anthropic_stream(stream_path)
            if e in ("thinking_delta", "text_delta", "function_call"):
                return parse_gemini_stream(stream_path)
    return []


# ---------------------------------------------------------------------------
# HOT path scan (literal substring matching)
# ---------------------------------------------------------------------------

_OUTPUT_PREFIXES = ("/output/", "/repo/result/")
_OUTPUT_KEY_PREFIXES = ("output_",)


def _is_output_path(path: str) -> bool:
    return any(path.startswith(p) for p in _OUTPUT_PREFIXES)


def hot_path_scan(
    blocks: list[StreamBlock],
    prior: dict[str, tuple[str, ...] | list[str]],
) -> dict[str, float]:
    """Substring-match workspace-prior paths in thinking text.

    Three needle forms per path: full path, basename (if unique across
    the prior), stem (if unique), and a short variant (stem without the
    "T120000" date suffix used by NWB filenames). Output paths are
    excluded — they're write targets, not staging candidates.

    Returns {path: t_ms_of_first_mention} for hits. Timestamp is
    interpolated linearly over the (t_first, t_stop) span of the block
    that contains the match, based on character offset.

    Ported faithfully from `poc/probe_reasoning_slack.py::hot_path_scan`.
    """
    input_paths: list[str] = []
    for key, paths in prior.items():
        if key.startswith(_OUTPUT_KEY_PREFIXES):
            continue
        for p in paths:
            if not _is_output_path(p):
                input_paths.append(p)

    base_counts: Counter[str] = Counter()
    stem_counts: Counter[str] = Counter()
    short_counts: Counter[str] = Counter()
    for p in input_paths:
        base = p.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        short = stem.replace("T120000", "")
        base_counts[base] += 1
        stem_counts[stem] += 1
        short_counts[short] += 1

    needles_by_path: dict[str, list[str]] = {}
    for p in input_paths:
        base = p.rsplit("/", 1)[-1]
        stem = base.rsplit(".", 1)[0]
        short = stem.replace("T120000", "")
        needles = [p]
        if base_counts[base] == 1:
            needles.append(base)
        if stem != base and stem_counts[stem] == 1:
            needles.append(stem)
        if short != stem and short_counts[short] == 1:
            needles.append(short)
        needles_by_path[p] = needles

    # Concatenate ALL detection-scannable block types and track per-block
    # char-offset segments. We scan the same set that run_detector scans
    # (thinking + text + tool_result) so a pathful-prompt experiment —
    # where the LLM writes literal paths in visible text or after seeing
    # them in tool_result — gets credit for the literal-path mention.
    segments: list[tuple[int, int, float | None, float | None]] = []
    full_text = ""
    for b in blocks:
        if b.type not in _SCANNABLE_BLOCK_TYPES:
            continue
        if not b.text:
            continue
        start = len(full_text)
        full_text += b.text
        end = len(full_text)
        segments.append((start, end, b.t_first, b.t_stop))
    if not full_text:
        return {}

    def offset_to_t_ms(offset: int) -> float | None:
        for start, end, tf, ts in segments:
            if start <= offset < end:
                if tf is None:
                    return ts
                if ts is None:
                    return tf
                if end == start:
                    return tf
                return tf + (ts - tf) * ((offset - start) / max(1, end - start))
        return segments[-1][2] if segments else None

    hot: dict[str, float] = {}
    for path, needles in needles_by_path.items():
        best_offset: int | None = None
        for n in needles:
            i = full_text.find(n)
            if i >= 0 and (best_offset is None or i < best_offset):
                best_offset = i
        if best_offset is not None:
            t = offset_to_t_ms(best_offset)
            hot[path] = t if t is not None else 0.0
    return hot


# ---------------------------------------------------------------------------
# Run detector — replay thinking text through the rule library
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleActivation:
    """One rule firing during detector replay."""
    rule_name: str
    fired_at_ms: float | None
    char_offset: int
    prior_keys: tuple[str, ...]
    detected_files: tuple[str, ...]
    source: str = "thinking"  # "thinking" | "tool_result" | "text"
    turn: int = 0  # Turn at which the rule fired (0-indexed)


@dataclass(frozen=True)
class TierResult:
    """The cumulative detected set at one tier."""
    tier: int
    detected_files: tuple[str, ...]
    activation_t_ms: float | None  # earliest activation contributing to this tier

    @property
    def size(self) -> int:
        return len(self.detected_files)


@dataclass(frozen=True)
class Detection:
    """Full detector output for one (stream, workload) pair."""
    activations: tuple[RuleActivation, ...]
    hot: dict[str, float]
    tier_1: TierResult
    tier_2: TierResult  # cumulative
    tier_3: TierResult  # cumulative
    rule_library_version: str = RULE_LIBRARY_VERSION
    rule_library_hash: str = RULE_LIBRARY_HASH

    def to_dict(self) -> dict:
        return {
            "rule_library_version": self.rule_library_version,
            "rule_library_hash": self.rule_library_hash,
            "activations": {
                a.rule_name: {
                    "t_ms": a.fired_at_ms,
                    "char_offset": a.char_offset,
                    "prior_keys": list(a.prior_keys),
                }
                for a in self.activations
            },
            "hot": {p: round(t, 1) for p, t in sorted(self.hot.items(), key=lambda kv: kv[1])},
            "tier_1_size": self.tier_1.size,
            "tier_2_size": self.tier_2.size,
            "tier_3_size": self.tier_3.size,
            "tier_1_t_ms": self.tier_1.activation_t_ms,
            "tier_2_t_ms": self.tier_2.activation_t_ms,
            "tier_3_t_ms": self.tier_3.activation_t_ms,
            "tier_1_paths": list(self.tier_1.detected_files),
            "tier_2_paths": list(self.tier_2.detected_files),
            "tier_3_paths": list(self.tier_3.detected_files),
        }


def _tier_for_size(n: int) -> int:
    """Tier assignment by target-set size.
    Tier-1 (eager stage): rules with small, immediately-needed target sets.
    Tier-2 (opportunistic): rules with medium target sets.
    Tier-3 (on-demand): rules with large target sets, staged reactively."""
    if n <= 20:
        return 1
    if n <= 200:
        return 2
    return 3


_SCANNABLE_BLOCK_TYPES = ("thinking", "text", "tool_result")
# We scan all three because in practice:
#   - Turn 0 of an assistant response: thinking carries reasoning.
#   - Turns 1+ in a multi-turn agentic loop: the model often emits
#     visible *text* instead of thinking after a tool_result, and the
#     text contains the I/O-relevant tokens (e.g. "C08 (Band 8) file").
#   - tool_result content from prior turns provides discovery signal.
# Deduplication by rule_name ensures a rule only fires once per session,
# so scanning thinking + text together does NOT double-count.


def run_detector(
    blocks: list[StreamBlock],
    prior: dict[str, tuple[str, ...] | list[str]],
    ruleset: RuleSet,
    per_char: bool = True,
) -> Detection:
    """Replay thinking text + tool_result content through the ruleset;
    emit tiered detection.

    `per_char=True` (default, unchanged PoC behavior) scans thinking blocks
    character-by-character to interpolate the exact activation timestamp.
    That loop is O(n²) per block and degenerates on long multi-turn
    transcripts where `accumulated` carries large tool_result listings.
    `per_char=False` scans thinking blocks atomically (like text/tool_result),
    yielding the identical fired-rule SET — only activation *timestamps* lose
    sub-block precision. Use it when only the detected set matters (e.g.
    byte-recall scoring of multiturn runs).

    For each rule, the FIRST match wins (across all scanned blocks in
    stream order). Activation time:
      - thinking blocks: char-offset interpolated over the block's
        (t_first, t_stop) span (PoC behavior)
      - tool_result blocks: t_first of the block (atomic arrival —
        the full tool output is delivered in one synchronous response)

    Each activation carries `source` ("thinking" | "tool_result") and
    `turn` (0-indexed) so the consumer can audit "which signal fired
    each rule, and at which point in the conversation."

    Rules are tiered by the total file count their target_keys resolve
    to (in the workspace prior), per PoC §6.2.
    """
    # Pre-compile patterns for speed (case-insensitive search)
    compiled = [(r, re.compile(r.pattern, flags=re.IGNORECASE)) for r in ruleset.rules]

    activations: list[RuleActivation] = []
    fired_names: set[str] = set()
    accumulated = ""
    for b in blocks:
        if b.type not in _SCANNABLE_BLOCK_TYPES or not b.text:
            continue
        tf = b.t_first
        ts = b.t_stop
        text = b.text
        if per_char and b.type == "thinking":
            # Per-char extension: re-evaluate every character (PoC behavior).
            # For ≤ 2 KB thinking text per block, this is fine.
            for end in range(1, len(text) + 1):
                cur = accumulated + text[:end]
                for rule, regex in compiled:
                    if rule.name in fired_names:
                        continue
                    if regex.search(cur):
                        if tf is not None and ts is not None and len(text) > 0:
                            t_est = tf + (ts - tf) * (end / len(text))
                        else:
                            t_est = tf if tf is not None else ts
                        detected = tuple(
                            p for k in rule.target_keys for p in prior.get(k, ())
                        )
                        activations.append(
                            RuleActivation(
                                rule_name=rule.name,
                                fired_at_ms=t_est,
                                char_offset=len(accumulated) + end,
                                prior_keys=rule.target_keys,
                                detected_files=detected,
                                source="thinking",
                                turn=b.turn,
                            )
                        )
                        fired_names.add(rule.name)
            accumulated += text
        else:
            # tool_result OR text: scan the block atomically.
            # Source tag is the block type so downstream consumers can
            # distinguish "rule fired because thinking" vs "rule fired
            # because visible text" vs "rule fired because tool_result".
            block_source = b.type
            # The whole result text appears at once (no per-char streaming).
            # Use the block's t_first as the activation time; if absent, ts.
            t_est_block = tf if tf is not None else ts
            full = accumulated + text
            for rule, regex in compiled:
                if rule.name in fired_names:
                    continue
                m = regex.search(full)
                if m is not None and m.start() >= len(accumulated):
                    # Rule matches inside the new tool_result text
                    detected = tuple(
                        p for k in rule.target_keys for p in prior.get(k, ())
                    )
                    activations.append(
                        RuleActivation(
                            rule_name=rule.name,
                            fired_at_ms=t_est_block,
                            char_offset=m.start(),
                            prior_keys=rule.target_keys,
                            detected_files=detected,
                            source=block_source,
                            turn=b.turn,
                        )
                    )
                    fired_names.add(rule.name)
            accumulated += text

    # Tiered aggregation
    tier_files: dict[int, set[str]] = {1: set(), 2: set(), 3: set()}
    tier_t: dict[int, float | None] = {1: None, 2: None, 3: None}
    for act in activations:
        n = len(set(act.detected_files))
        t = _tier_for_size(n)
        tier_files[t].update(act.detected_files)
        tact = act.fired_at_ms
        if tact is not None and (tier_t[t] is None or tact < tier_t[t]):
            tier_t[t] = tact

    cum_1 = tier_files[1]
    cum_2 = cum_1 | tier_files[2]
    cum_3 = cum_2 | tier_files[3]

    return Detection(
        activations=tuple(activations),
        hot=hot_path_scan(blocks, prior),
        tier_1=TierResult(1, tuple(sorted(cum_1)), tier_t[1]),
        tier_2=TierResult(2, tuple(sorted(cum_2)), tier_t[2]),
        tier_3=TierResult(3, tuple(sorted(cum_3)), tier_t[3]),
    )
