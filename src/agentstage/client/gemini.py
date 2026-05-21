"""Gemini client wrapper — live streaming + thinking + tool_use capture.

Companion to client/anthropic.py. Wraps google-genai's
``generate_content_stream`` so the rest of the system can consume a
unified event stream regardless of LLM vendor.

Emits per-chunk events on the StreamSession:
  - "thinking_delta"     — model emits a "thought" part
  - "text_delta"         — model emits visible text
  - "tool_use_start"     — model emits a function_call (one per call)
  - "tool_use_end"       — function_call complete (we synthesize this
                            at the end of the stream)
  - "message_stop"       — stream finished

These match the conceptual shape of Anthropic's content_block events so
the downstream detector dispatch path can be source-agnostic.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Iterator

from google import genai
from google.genai import types

from agentstage.detector.engine import StreamBlock, run_detector
from agentstage.detector.rules import RuleSet
from agentstage.stager import DataHint, Stager, now_ms


@dataclass
class GeminiToolCall:
    """One function_call block emitted by the model."""

    name: str
    args: dict
    id: str = ""  # Gemini doesn't have stable tool_call ids in the same way
    block_idx: int = 0
    input_json: str = ""  # mirror AnthropicClient's ToolCall for path_b compat


@dataclass
class GeminiStreamSession:
    """Mirror of StreamSession for Gemini. Same shape so path_b can
    swap clients without changing the per-turn bookkeeping."""

    started_at_ms: float
    workspace_prior: dict[str, tuple[str, ...] | list[str]]
    ruleset: RuleSet
    stager: Stager | None = None
    thinking_blocks: list[StreamBlock] = field(default_factory=list)
    fired_rule_names: set[str] = field(default_factory=set)
    tool_calls: list[GeminiToolCall] = field(default_factory=list)
    first_thinking_chunk_at_ms: float | None = None
    first_tool_use_at_ms: float | None = None
    raw_events: list[dict] = field(default_factory=list)

    @property
    def slack_ms(self) -> float | None:
        if (self.first_thinking_chunk_at_ms is None
            or self.first_tool_use_at_ms is None):
            return None
        return self.first_tool_use_at_ms - self.first_thinking_chunk_at_ms


class GeminiClient:
    """Live AgentStage wrapper around google.genai.Client.

    Usage parallels AnthropicClient:
        client = GeminiClient(api_key=..., workspace_prior=..., ruleset=...)
        response = client.stream(model="gemini-2.5-flash", messages=...,
                                 max_tokens=8192, thinking_budget=4096, ...)
        for event in response.events():
            ...
    """

    def __init__(
        self,
        *,
        api_key: str,
        stager: Stager | None = None,
        workspace_prior: dict[str, tuple[str, ...] | list[str]] | None = None,
        ruleset: RuleSet | None = None,
    ) -> None:
        self._api_key = api_key
        self._stager = stager
        self._workspace_prior = workspace_prior
        self._ruleset = ruleset
        self._sdk = genai.Client(api_key=api_key)

    def stream(
        self,
        *,
        model: str,
        messages: list[dict],
        max_tokens: int = 8192,
        thinking_budget: int = 4096,
        temperature: float = 1.0,
        system: str | None = None,
        extra_body: dict | None = None,
    ) -> "GeminiStreamingResponse":
        """Start a streaming generate_content call.

        `messages` and `extra_body["tools"]` use the Anthropic format that
        path_b_multiturn already produces; we translate to Gemini's
        Content/Part/Tool/FunctionDeclaration types internally.
        """
        # Translate messages (Anthropic-flavored dicts) → Gemini Contents
        contents = _translate_messages_to_gemini(messages)

        # Translate tool list (Anthropic-flavored) → Gemini Tool
        tools = []
        if extra_body and "tools" in extra_body:
            decls = []
            for t in extra_body["tools"]:
                decls.append(types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description", ""),
                    parameters=t.get("input_schema") or {},
                ))
            if decls:
                tools.append(types.Tool(function_declarations=decls))

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            tools=tools or None,
            thinking_config=types.ThinkingConfig(
                thinking_budget=thinking_budget,
                include_thoughts=True,
            ),
            system_instruction=system if system else None,
        )

        session = GeminiStreamSession(
            started_at_ms=now_ms(),
            workspace_prior=self._workspace_prior or {},
            ruleset=self._ruleset,  # type: ignore[arg-type]
            stager=self._stager,
        )

        sdk_stream = self._sdk.models.generate_content_stream(
            model=model,
            contents=contents,
            config=config,
        )
        return GeminiStreamingResponse(sdk_stream=sdk_stream, session=session)


def _translate_messages_to_gemini(messages: list[dict]) -> list[types.Content]:
    """Translate Anthropic-flavored message list to Gemini Contents.

    Input message dict shape (Anthropic-flavored, what path_b builds):
        {"role": "user", "content": "..." OR [{"type": "tool_result",
                                               "tool_use_id": ...,
                                               "content": "..."}, ...]}
        {"role": "assistant", "content": [{"type": "thinking",
                                            "thinking": "..."},
                                           {"type": "text", "text": "..."},
                                           {"type": "tool_use", "name": ...,
                                            "id": ..., "input": ...}]}

    Output Gemini Content shape:
        Content(role="user", parts=[Part(text=...) or
                                     Part(function_response=...)])
        Content(role="model", parts=[Part(text=..., thought=True/False) or
                                      Part(function_call=...)])
    """
    out: list[types.Content] = []
    for msg in messages:
        role = msg["role"]
        gemini_role = "model" if role == "assistant" else "user"
        content = msg.get("content")
        parts: list[types.Part] = []

        if isinstance(content, str):
            parts.append(types.Part(text=content))
        elif isinstance(content, list):
            for cb in content:
                if not isinstance(cb, dict):
                    continue
                cb_type = cb.get("type")
                if cb_type == "thinking":
                    # Gemini doesn't accept us replaying thought parts back
                    # in the conversation history (they're model-internal),
                    # so we skip them. Text remains.
                    continue
                if cb_type == "text":
                    parts.append(types.Part(text=cb.get("text", "")))
                elif cb_type == "tool_use":
                    parts.append(types.Part(
                        function_call=types.FunctionCall(
                            name=cb.get("name", ""),
                            args=cb.get("input") or {},
                        )
                    ))
                elif cb_type == "tool_result":
                    tool_use_id = cb.get("tool_use_id", "")
                    result_content = cb.get("content", "")
                    if isinstance(result_content, list):
                        result_content = "\n".join(
                            sub.get("text", "")
                            for sub in result_content
                            if isinstance(sub, dict) and sub.get("type") == "text"
                        )
                    # Gemini's FunctionResponse needs a name (the function
                    # that was called) — we look it up by tool_use_id in
                    # the preceding assistant message's function_calls.
                    name = _find_tool_use_name(messages, tool_use_id)
                    parts.append(types.Part(
                        function_response=types.FunctionResponse(
                            name=name,
                            response={"output": str(result_content)},
                        )
                    ))
        if parts:
            out.append(types.Content(role=gemini_role, parts=parts))
    return out


def _find_tool_use_name(messages: list[dict], tool_use_id: str) -> str:
    """Look up which function was called for a given tool_use_id."""
    for m in messages:
        if m.get("role") != "assistant":
            continue
        c = m.get("content")
        if not isinstance(c, list):
            continue
        for cb in c:
            if isinstance(cb, dict) and cb.get("type") == "tool_use" \
               and cb.get("id") == tool_use_id:
                return cb.get("name", "")
    return ""


class _Event:
    """Lightweight namespace mimicking the Anthropic SDK event shape.

    Allows path_b_multiturn's existing event-handling code to consume
    Gemini events without rewriting the dispatch loop.
    """

    __slots__ = ("type", "index", "content_block", "delta")

    def __init__(self, type: str, index: int = -1,
                 content_block: Any = None, delta: Any = None) -> None:
        self.type = type
        self.index = index
        self.content_block = content_block
        self.delta = delta


class _ContentBlock:
    __slots__ = ("type", "name", "id")

    def __init__(self, type: str, name: str = "", id: str = "") -> None:
        self.type = type
        self.name = name
        self.id = id


class _Delta:
    __slots__ = ("type", "thinking", "text", "partial_json", "signature")

    def __init__(self, type: str, thinking: str = "", text: str = "",
                 partial_json: str = "", signature: str = "") -> None:
        self.type = type
        self.thinking = thinking
        self.text = text
        self.partial_json = partial_json
        self.signature = signature


class GeminiStreamingResponse:
    """Iterable wrapper over a Gemini streaming response.

    Yields Anthropic-SDK-shaped events (via _Event/_ContentBlock/_Delta
    namespaces) so path_b_multiturn's existing dispatch loop can consume
    both providers without branching.
    """

    def __init__(self, *, sdk_stream: Any, session: GeminiStreamSession) -> None:
        self._sdk_stream = sdk_stream
        self.session = session
        # We assign synthetic block indices: increment when we see a NEW
        # part type/identity. Within one chunk, multiple parts may exist
        # of different types; we emit per-part deltas under a per-type idx.
        self._next_block_idx = 0
        self._open_blocks: dict[int, dict] = {}  # idx → {kind, accumulated text}
        # Track the currently-open block per kind so consecutive deltas
        # of the same kind accumulate into the same block index.
        self._open_by_kind: dict[str, int] = {}

    def events(self) -> Iterator[_Event]:
        for chunk in self._sdk_stream:
            t_ms = now_ms() - self.session.started_at_ms
            self.session.raw_events.append({"t_ms": t_ms})
            if not chunk.candidates:
                continue
            for cand in chunk.candidates:
                if not cand.content or not cand.content.parts:
                    continue
                for part in cand.content.parts:
                    yield from self._handle_part(part, t_ms)
        # Emit close events for any still-open blocks + message_stop
        for idx in list(self._open_blocks.keys()):
            yield _Event(type="content_block_stop", index=idx)
        yield _Event(type="message_stop")

    def _handle_part(self, part: Any, t_ms: float) -> Iterator[_Event]:
        is_thought = getattr(part, "thought", False)
        text = getattr(part, "text", None)
        fcall = getattr(part, "function_call", None)

        if fcall is not None:
            idx = self._next_block_idx
            self._next_block_idx += 1
            args = dict(fcall.args) if fcall.args else {}
            # Gemini doesn't issue stable tool_use ids; synthesize one
            # so tool_result correlation works.
            synth_id = f"gemfn_{idx}_{int(t_ms)}"
            tc = GeminiToolCall(
                name=fcall.name or "",
                args=args,
                id=synth_id,
                block_idx=idx,
                input_json=json.dumps(args),
            )
            self.session.tool_calls.append(tc)
            if self.session.first_tool_use_at_ms is None:
                self.session.first_tool_use_at_ms = t_ms
            yield _Event(
                type="content_block_start", index=idx,
                content_block=_ContentBlock(type="tool_use", name=tc.name, id=synth_id),
            )
            yield _Event(
                type="content_block_delta", index=idx,
                delta=_Delta(type="input_json_delta", partial_json=tc.input_json),
            )
            yield _Event(type="content_block_stop", index=idx)
            return

        if text:
            kind = "thinking" if is_thought else "text"
            idx = self._open_by_kind.get(kind)
            if idx is None:
                idx = self._next_block_idx
                self._next_block_idx += 1
                self._open_blocks[idx] = {"kind": kind, "text": ""}
                self._open_by_kind[kind] = idx
                if kind == "thinking" and self.session.first_thinking_chunk_at_ms is None:
                    self.session.first_thinking_chunk_at_ms = t_ms
                yield _Event(
                    type="content_block_start", index=idx,
                    content_block=_ContentBlock(type=kind),
                )
            self._open_blocks[idx]["text"] += text
            if kind == "thinking":
                yield _Event(
                    type="content_block_delta", index=idx,
                    delta=_Delta(type="thinking_delta", thinking=text),
                )
            else:
                yield _Event(
                    type="content_block_delta", index=idx,
                    delta=_Delta(type="text_delta", text=text),
                )
