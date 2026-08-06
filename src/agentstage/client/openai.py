"""AgentStage wrapper around the OpenAI SDK, for OpenAI-compatible endpoints.

Wraps `openai.OpenAI`'s streaming chat-completions call, runs the detector on
reasoning deltas as they arrive, and dispatches tier-1 hints to the stager.
Events are re-shaped to the Anthropic SDK's form so a harness can swap
providers without rewriting its dispatch loop.

Reasoning text is read from `delta.reasoning` (Qwen3 style) or
`delta.reasoning_content` (older vLLM). Servers that expose neither will
stream normally but stage nothing, because AgentStage has no thinking text to
detect against. That includes OpenAI's own API, which does not return
reasoning tokens: point this client at a self-hosted vLLM, DeepSeek, or
another server that surfaces reasoning.

    client = OpenAIClient(
        api_key="EMPTY",
        base_url="http://localhost:8002/v1",
        stager=stager, workspace_prior=prior, ruleset=ruleset,
    )
    response = client.stream(model="Qwen/Qwen3.6-27B", messages=[...])
    for event in response.events():
        ...
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any

from openai import OpenAI

from agentstage.client.dispatch import ContentBlock, Delta, Event, RuleDispatcher
from agentstage.detector.engine import StreamBlock
from agentstage.detector.rules import RuleSet
from agentstage.stager import Stager, now_ms


@dataclass
class OpenAIToolCall:
    """One tool call assembled from streamed argument fragments."""

    name: str
    id: str = ""
    block_idx: int = 0
    input_json: str = ""

    @property
    def args(self) -> dict:
        try:
            return json.loads(self.input_json) if self.input_json else {}
        except json.JSONDecodeError:
            return {}


@dataclass
class OpenAIStreamSession:
    """Mirror of the Anthropic and Gemini sessions, same shape so per-turn
    bookkeeping is provider-independent."""

    started_at_ms: float
    workspace_prior: dict[str, tuple[str, ...] | list[str]]
    ruleset: RuleSet | None = None
    stager: Stager | None = None
    thinking_blocks: list[StreamBlock] = field(default_factory=list)
    fired_rule_names: set[str] = field(default_factory=set)
    tool_calls: list[OpenAIToolCall] = field(default_factory=list)
    first_thinking_chunk_at_ms: float | None = None
    first_tool_use_at_ms: float | None = None
    raw_events: list[dict] = field(default_factory=list)

    @property
    def slack_ms(self) -> float | None:
        if (self.first_thinking_chunk_at_ms is None
                or self.first_tool_use_at_ms is None):
            return None
        return self.first_tool_use_at_ms - self.first_thinking_chunk_at_ms


class OpenAIClient:
    """Live AgentStage wrapper around `openai.OpenAI`."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str | None = None,
        stager: Stager | None = None,
        workspace_prior: dict[str, tuple[str, ...] | list[str]] | None = None,
        ruleset: RuleSet | None = None,
        timeout: Any | None = None,
    ) -> None:
        self._api_key = api_key
        self._base_url = base_url
        self._stager = stager
        self._workspace_prior = workspace_prior
        self._ruleset = ruleset
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        if timeout is not None:
            kwargs["timeout"] = timeout
        self._sdk = OpenAI(**kwargs)

    def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 8192,
        temperature: float = 1.0,
        system: str | None = None,
        tools: list[dict] | None = None,
        extra_body: dict | None = None,
    ) -> OpenAIStreamingResponse:
        """Start a streaming chat-completions call.

        `system`, when given, is prepended as a system message. `tools` uses
        the OpenAI function-tool schema.
        """
        msgs = list(messages)
        if system:
            msgs = [{"role": "system", "content": system}] + msgs

        kwargs: dict[str, Any] = {
            "model": model,
            "messages": msgs,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }
        if tools:
            kwargs["tools"] = tools
        if extra_body:
            kwargs["extra_body"] = extra_body

        session = OpenAIStreamSession(
            started_at_ms=now_ms(),
            workspace_prior=self._workspace_prior or {},
            ruleset=self._ruleset,
            stager=self._stager,
        )
        sdk_stream = self._sdk.chat.completions.create(**kwargs)
        return OpenAIStreamingResponse(sdk_stream=sdk_stream, session=session)


class OpenAIStreamingResponse:
    """Iterable wrapper over an OpenAI-compatible streaming response."""

    def __init__(self, *, sdk_stream: Any, session: OpenAIStreamSession) -> None:
        self._sdk_stream = sdk_stream
        self.session = session
        self._dispatcher = RuleDispatcher(
            ruleset=session.ruleset,
            workspace_prior=session.workspace_prior,
            stager=session.stager,
        )
        self._next_block_idx = 0
        self._open_by_kind: dict[str, int] = {}
        self._open_text: dict[int, str] = {}
        self._tool_by_index: dict[int, OpenAIToolCall] = {}

    def events(self) -> Iterator[Event]:
        for chunk in self._sdk_stream:
            t_ms = now_ms() - self.session.started_at_ms
            self.session.raw_events.append({"t_ms": t_ms})
            choices = getattr(chunk, "choices", None)
            if not choices:
                continue
            delta = getattr(choices[0], "delta", None)
            if delta is None:
                continue

            reasoning = (getattr(delta, "reasoning", None)
                         or getattr(delta, "reasoning_content", None))
            if reasoning:
                yield from self._emit_text("thinking", reasoning, t_ms)

            content = getattr(delta, "content", None)
            if content:
                yield from self._emit_text("text", content, t_ms)

            for tc in getattr(delta, "tool_calls", None) or []:
                yield from self._handle_tool_call(tc, t_ms)

        for idx in list(self._open_text):
            yield Event(type="content_block_stop", index=idx)
        for call in self._tool_by_index.values():
            yield Event(type="content_block_stop", index=call.block_idx)
        yield Event(type="message_stop")
        self._finalize_thinking_blocks()

    def _emit_text(self, kind: str, text: str, t_ms: float) -> Iterator[Event]:
        idx = self._open_by_kind.get(kind)
        if idx is None:
            idx = self._next_block_idx
            self._next_block_idx += 1
            self._open_by_kind[kind] = idx
            self._open_text[idx] = ""
            if kind == "thinking" and self.session.first_thinking_chunk_at_ms is None:
                self.session.first_thinking_chunk_at_ms = t_ms
            yield Event(type="content_block_start", index=idx,
                        content_block=ContentBlock(type=kind))
        self._open_text[idx] += text

        if kind == "thinking":
            self._dispatcher.fire(self._open_text[idx], t_ms,
                                  self.session.fired_rule_names)
            yield Event(type="content_block_delta", index=idx,
                        delta=Delta(type="thinking_delta", thinking=text))
        else:
            yield Event(type="content_block_delta", index=idx,
                        delta=Delta(type="text_delta", text=text))

    def _handle_tool_call(self, tc: Any, t_ms: float) -> Iterator[Event]:
        pos = getattr(tc, "index", 0) or 0
        call = self._tool_by_index.get(pos)
        if call is None:
            idx = self._next_block_idx
            self._next_block_idx += 1
            fn = getattr(tc, "function", None)
            call = OpenAIToolCall(
                name=getattr(fn, "name", "") or "",
                id=getattr(tc, "id", "") or f"oaifn_{idx}_{int(t_ms)}",
                block_idx=idx,
            )
            self._tool_by_index[pos] = call
            self.session.tool_calls.append(call)
            if self.session.first_tool_use_at_ms is None:
                self.session.first_tool_use_at_ms = t_ms
            yield Event(type="content_block_start", index=idx,
                        content_block=ContentBlock(type="tool_use",
                                                   name=call.name, id=call.id))

        fn = getattr(tc, "function", None)
        if fn is not None:
            if not call.name and getattr(fn, "name", None):
                call.name = fn.name
            fragment = getattr(fn, "arguments", None)
            if fragment:
                call.input_json += fragment
                yield Event(type="content_block_delta", index=call.block_idx,
                            delta=Delta(type="input_json_delta",
                                        partial_json=fragment))

    def _finalize_thinking_blocks(self) -> None:
        idx = self._open_by_kind.get("thinking")
        if idx is None:
            return
        self.session.thinking_blocks.append(StreamBlock(
            type="thinking",
            t_first=self.session.first_thinking_chunk_at_ms or 0.0,
            t_stop=now_ms() - self.session.started_at_ms,
            text=self._open_text.get(idx, ""),
            chunks=1,
        ))
