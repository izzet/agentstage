"""Drop-in Anthropic client wrapper with live streaming → detector → stager.

Sits on top of the `anthropic` SDK (`anthropic.Anthropic.messages.create`)
and tees the streaming SSE events. The caller's code path is unchanged
— `AnthropicClient.stream(...)` yields the same events the SDK would,
in the same order. Detector + stager run as side effects.

Live detector:
  Per chunk, accumulate thinking text. After each chunk, re-run the
  ruleset against the accumulated text and detect newly-fired rules.
  Newly-fired rules → DataHint → stager.prefetch(...).

  We do not implement true incremental matching (PoC §6.2's char-offset
  detection); for the live path we re-scan the accumulated text on each
  chunk and diff against the previous activation set. O(N · M) per
  chunk where N = thinking chars, M = rules. For ~2 KB thinking + 100
  rules, this is sub-millisecond per chunk — well within the streaming
  budget.

The caller iterates this object's `stream(...)` method just like
the SDK's iterator. Tool-call detection: when a `content_block_start`
event arrives with `block_type=="tool_use"`, we record the tool name
and input; the caller decides what to do (execute, log, abort).
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

import anthropic

from agentstage.detector.engine import StreamBlock
from agentstage.detector.rules import RuleSet
from agentstage.client.dispatch import RuleDispatcher
from agentstage.stager import Stager, now_ms


@dataclass
class ToolCall:
    """A single tool-use block emitted by the model."""

    block_idx: int
    name: str
    input_json: str = ""  # accumulated input deltas; valid JSON when complete
    id: str = ""


@dataclass
class StreamSession:
    """In-flight state for one streaming LLM call."""

    started_at_ms: float
    workspace_prior: dict[str, tuple[str, ...] | list[str]]
    ruleset: RuleSet
    stager: Stager | None = None

    # Per-call accumulators
    thinking_blocks: list[StreamBlock] = field(default_factory=list)
    fired_rule_names: set[str] = field(default_factory=set)
    tool_calls: list[ToolCall] = field(default_factory=list)
    first_thinking_chunk_at_ms: float | None = None
    first_tool_use_at_ms: float | None = None
    raw_events: list[dict] = field(default_factory=list)

    @property
    def slack_ms(self) -> float | None:
        if self.first_thinking_chunk_at_ms is None or self.first_tool_use_at_ms is None:
            return None
        return self.first_tool_use_at_ms - self.first_thinking_chunk_at_ms


class AnthropicClient:
    """Live AgentStage wrapper around `anthropic.Anthropic`.

    Usage:
        client = AnthropicClient(
            api_key=...,
            base_url="https://.../anthropic/v1/messages",  # optional Azure path
            stager=stager,
            workspace_prior=prior,
            ruleset=ruleset,
        )
        session = client.stream(model="claude-haiku-4-5", messages=..., ...)
        for event in session.events():
            # event is the underlying SDK event, unchanged
            ...
        # after the loop, session.tool_calls is populated, session.slack_ms is set
    """

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        stager: Stager | None = None,
        workspace_prior: dict[str, tuple[str, ...] | list[str]] | None = None,
        ruleset: RuleSet | None = None,
        detector_enabled: bool | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._stager = stager
        self._workspace_prior = workspace_prior
        self._ruleset = ruleset
        # When the detector is disabled the proxy is a pure SSE pass-through:
        # events are forwarded unchanged with zero per-event work. Explicit
        # arg wins; otherwise honour AGENTSTAGE_DETECTOR_DISABLED=1.
        if detector_enabled is None:
            detector_enabled = os.environ.get("AGENTSTAGE_DETECTOR_DISABLED", "") not in ("1", "true", "True")
        self._detector_enabled = detector_enabled

        # The anthropic SDK uses Authorization: Bearer for non-anthropic.com
        # endpoints when constructed with base_url. For Azure Foundry it
        # works out of the box.
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        self._sdk = anthropic.Anthropic(**kwargs)

    def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 8192,
        thinking_budget: int = 16384,
        temperature: float = 1.0,
        system: str | None = None,
        extra_body: dict | None = None,
    ) -> "StreamingResponse":
        """Start a streaming Messages API call. Returns a StreamingResponse
        that the caller iterates."""
        body: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": messages,
            "stream": True,
            "thinking": {"type": "enabled", "budget_tokens": thinking_budget},
        }
        if system is not None:
            body["system"] = system
        if extra_body:
            body.update(extra_body)

        session = StreamSession(
            started_at_ms=now_ms(),
            workspace_prior=self._workspace_prior or {},
            ruleset=self._ruleset,  # type: ignore[arg-type]
            stager=self._stager,
        )
        sdk_stream = self._sdk.messages.create(**body)
        return StreamingResponse(
            sdk_stream=sdk_stream,
            session=session,
            detector_enabled=self._detector_enabled,
        )


class StreamingResponse:
    """Iterable wrapper over the SDK's streaming Messages response.

    Yields the underlying SDK events unchanged. Side effect on each event:
    update the session's accumulators and dispatch DataHints to the stager
    when new detector rules fire.
    """

    def __init__(
        self,
        *,
        sdk_stream: Any,
        session: StreamSession,
        detector_enabled: bool = True,
    ) -> None:
        self._sdk_stream = sdk_stream
        self.session = session
        self._detector_enabled = detector_enabled
        # Block-index → (StreamBlock, ToolCall|None) so we can update on deltas
        self._block_kind: dict[int, str] = {}
        self._block_first_chunk_ms: dict[int, float] = {}
        self._tool_calls_by_idx: dict[int, ToolCall] = {}
        self._thinking_text_by_idx: dict[int, list[str]] = {}
        # Rule regexes compiled once per stream (lazily, on first thinking
        # chunk) so the live per-chunk scan stays cheap.
        self._dispatcher: RuleDispatcher | None = None

    def events(self) -> Iterator[Any]:
        if not self._detector_enabled:
            # Pure pass-through: forward every event unchanged, no parsing,
            # no detector, no stager. The proxy is byte-transparent here.
            yield from self._sdk_stream
            return
        for event in self._sdk_stream:
            t = (now_ms() - self.session.started_at_ms)
            self._on_event(event, t)
            yield event
        # End of stream — finalize any pending blocks
        self._finalize_thinking_blocks()

    def _on_event(self, event: Any, t_ms: float) -> None:
        etype = getattr(event, "type", None)
        if etype == "content_block_start":
            idx = event.index
            block = event.content_block
            kind = getattr(block, "type", None)
            self._block_kind[idx] = kind
            self._block_first_chunk_ms[idx] = t_ms
            if kind == "tool_use":
                tc = ToolCall(
                    block_idx=idx,
                    name=getattr(block, "name", ""),
                    id=getattr(block, "id", ""),
                )
                self._tool_calls_by_idx[idx] = tc
                self.session.tool_calls.append(tc)
                if self.session.first_tool_use_at_ms is None:
                    self.session.first_tool_use_at_ms = t_ms
            elif kind == "thinking":
                self._thinking_text_by_idx[idx] = []
                if self.session.first_thinking_chunk_at_ms is None:
                    self.session.first_thinking_chunk_at_ms = t_ms
        elif etype == "content_block_delta":
            idx = event.index
            delta = event.delta
            dtype = getattr(delta, "type", None)
            if dtype == "thinking_delta":
                text_piece = getattr(delta, "thinking", "")
                if text_piece:
                    self._thinking_text_by_idx.setdefault(idx, []).append(text_piece)
                    # Run detector on the updated accumulated text for this block
                    self._maybe_fire_rules(idx, t_ms)
            elif dtype == "input_json_delta":
                tc = self._tool_calls_by_idx.get(idx)
                if tc is not None:
                    tc.input_json += getattr(delta, "partial_json", "")
            elif dtype == "signature_delta":
                # Signed-thinking-block passthrough; we don't need to act on this
                # for Path A (single-turn). Capture for completeness.
                pass

    def _maybe_fire_rules(self, block_idx: int, t_ms: float) -> None:
        """Scan this block's accumulated thinking text and dispatch any
        newly-fired rules. See `dispatch.RuleDispatcher` for the matching
        contract and why it differs from the offline engine."""
        if self._dispatcher is None:
            self._dispatcher = RuleDispatcher(
                ruleset=self.session.ruleset,
                workspace_prior=self.session.workspace_prior,
                stager=self.session.stager,
            )
        if not self._dispatcher.enabled:
            return
        accumulated = "".join(self._thinking_text_by_idx.get(block_idx, []))
        self._dispatcher.fire(accumulated, t_ms, self.session.fired_rule_names)

    def _finalize_thinking_blocks(self) -> None:
        # Snapshot the per-block accumulated thinking text into the session
        # for downstream consumption (e.g. paper_evals can read summary.json
        # equivalents).
        for idx, parts in self._thinking_text_by_idx.items():
            text = "".join(parts)
            t_first = self._block_first_chunk_ms.get(idx, 0.0)
            self.session.thinking_blocks.append(StreamBlock(
                type="thinking", t_first=t_first, t_stop=now_ms(),
                text=text, chunks=1,
            ))

