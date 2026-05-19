"""Predictor engine — parses a stream.jsonl into blocks, replays thinking
text through the frozen rule library, and emits a tiered `Prediction`.

This is the load-bearing port from `poc/probe_reasoning_slack.py`:
- `parse_anthropic_stream` ⇐ `block_timing_anthropic`
- `parse_gemini_stream` ⇐ `block_timing_gemini`
- `hot_path_scan` ⇐ `hot_path_scan`
- `run_predictor` ⇐ `run_predictor`

The engine is provider-aware (Anthropic vs Gemini SSE shapes differ) but
otherwise rule-library-agnostic — pass any `RuleSet` and it will fire
the regex matches against the thinking text in time order, then assemble
the tiered predicted set.

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

from agentstage.predictor.rules import (
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
    """One thinking / text / tool-use block reconstructed from a stream.jsonl.

    `t_first` and `t_stop` are the timestamps (in ms from urlopen start) of
    the first and last delta event for this block. `text` is the
    concatenated content; for tool_use blocks it's the JSON string of the
    arguments.
    """
    type: str           # "thinking" | "text" | "tool_use"
    t_first: float | None
    t_stop: float | None
    text: str
    chunks: int
    tool_name: str | None = None


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

    # Concatenate thinking text and track per-block char-offset segments
    segments: list[tuple[int, int, float | None, float | None]] = []
    full_text = ""
    for b in blocks:
        if b.type != "thinking":
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
# Run predictor — replay thinking text through the rule library
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RuleActivation:
    """One rule firing during predictor replay."""
    rule_name: str
    fired_at_ms: float | None
    char_offset: int
    prior_keys: tuple[str, ...]
    predicted_files: tuple[str, ...]


@dataclass(frozen=True)
class TierResult:
    """The cumulative predicted set at one tier."""
    tier: int
    predicted_files: tuple[str, ...]
    activation_t_ms: float | None  # earliest activation contributing to this tier

    @property
    def size(self) -> int:
        return len(self.predicted_files)


@dataclass(frozen=True)
class Prediction:
    """Full predictor output for one (stream, workload) pair."""
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
            "tier_1_paths": list(self.tier_1.predicted_files),
            "tier_2_paths": list(self.tier_2.predicted_files),
            "tier_3_paths": list(self.tier_3.predicted_files),
        }


def _tier_for_size(n: int) -> int:
    """Tier assignment by target-set size (PoC §6.2 thresholds)."""
    if n <= 10:
        return 1
    if n <= 200:
        return 2
    return 3


def run_predictor(
    blocks: list[StreamBlock],
    prior: dict[str, tuple[str, ...] | list[str]],
    ruleset: RuleSet,
) -> Prediction:
    """Replay thinking text through the ruleset; emit tiered prediction.

    For each rule, the FIRST char-offset at which it matches the
    accumulated thinking text wins. Activation time is interpolated
    linearly over the containing block's (t_first, t_stop) span.

    Rules are tiered by the total file count their target_keys resolve
    to (in the workspace prior), per PoC §6.2.
    """
    # Pre-compile patterns for speed (case-insensitive search)
    compiled = [(r, re.compile(r.pattern, flags=re.IGNORECASE)) for r in ruleset.rules]

    activations: list[RuleActivation] = []
    fired_names: set[str] = set()
    accumulated = ""
    for b in blocks:
        if b.type != "thinking" or not b.text:
            continue
        tf = b.t_first
        ts = b.t_stop
        text = b.text
        # Per-char extension: re-evaluate every character (PoC behavior).
        # For ≤ 2 KB thinking text, this is fine.
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
                    predicted = tuple(
                        p for k in rule.target_keys for p in prior.get(k, ())
                    )
                    activations.append(
                        RuleActivation(
                            rule_name=rule.name,
                            fired_at_ms=t_est,
                            char_offset=len(accumulated) + end,
                            prior_keys=rule.target_keys,
                            predicted_files=predicted,
                        )
                    )
                    fired_names.add(rule.name)
        accumulated += text

    # Tiered aggregation
    tier_files: dict[int, set[str]] = {1: set(), 2: set(), 3: set()}
    tier_t: dict[int, float | None] = {1: None, 2: None, 3: None}
    for act in activations:
        n = len(set(act.predicted_files))
        t = _tier_for_size(n)
        tier_files[t].update(act.predicted_files)
        tact = act.fired_at_ms
        if tact is not None and (tier_t[t] is None or tact < tier_t[t]):
            tier_t[t] = tact

    cum_1 = tier_files[1]
    cum_2 = cum_1 | tier_files[2]
    cum_3 = cum_2 | tier_files[3]

    return Prediction(
        activations=tuple(activations),
        hot=hot_path_scan(blocks, prior),
        tier_1=TierResult(1, tuple(sorted(cum_1)), tier_t[1]),
        tier_2=TierResult(2, tuple(sorted(cum_2)), tier_t[2]),
        tier_3=TierResult(3, tuple(sorted(cum_3)), tier_t[3]),
    )
