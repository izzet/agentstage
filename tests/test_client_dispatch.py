"""Tests for the shared live-dispatch path used by every client wrapper.

Covers `dispatch.RuleDispatcher` directly, then the OpenAI wrapper end to end
against a fake SDK stream, so provider plumbing is exercised without network.
"""

from __future__ import annotations

import pytest

from agentstage.client.dispatch import RuleDispatcher, tier_for_size
from agentstage.client.openai import OpenAIStreamingResponse, OpenAIStreamSession
from agentstage.stager import now_ms


class _Rule:
    def __init__(self, name: str, pattern: str, target_keys: tuple[str, ...]):
        self.name = name
        self.pattern = pattern
        self.target_keys = target_keys


class _RuleSet:
    def __init__(self, rules):
        self.rules = rules


class _RecordingStager:
    """Captures prefetch calls instead of touching the filesystem."""

    def __init__(self):
        self.hints = []

    def prefetch(self, hint):
        self.hints.append(hint)


PRIOR = {
    "sample_a": ("/cold/a1.bam", "/cold/a2.bam"),
    "everything": tuple(f"/cold/bulk/{i}.dat" for i in range(50)),
}


def _dispatcher(stager=None, ruleset=None):
    return RuleDispatcher(
        ruleset=ruleset if ruleset is not None else _RuleSet([
            _Rule("sample_a", r"sample\s*a", ("sample_a",)),
            _Rule("everything", r"all files", ("everything",)),
        ]),
        workspace_prior=PRIOR,
        stager=stager if stager is not None else _RecordingStager(),
    )


class TestTierForSize:
    def test_boundaries(self):
        assert tier_for_size(0) == 1
        assert tier_for_size(10) == 1
        assert tier_for_size(11) == 2
        assert tier_for_size(200) == 2
        assert tier_for_size(201) == 3


class TestRuleDispatcher:
    def test_disabled_without_stager_or_ruleset(self):
        assert not RuleDispatcher(ruleset=None, workspace_prior=PRIOR,
                                  stager=_RecordingStager()).enabled
        assert not RuleDispatcher(ruleset=_RuleSet([]), workspace_prior=PRIOR,
                                  stager=None).enabled

    def test_tier1_match_dispatches_the_bucket(self):
        stager = _RecordingStager()
        d = _dispatcher(stager)
        fired: set[str] = set()

        newly = d.fire("I should read sample A first", 12.0, fired)

        assert newly == ["sample_a"]
        assert fired == {"sample_a"}
        assert len(stager.hints) == 1
        hint = stager.hints[0]
        assert hint.detected_files == PRIOR["sample_a"]
        assert hint.tier == 1
        assert hint.fired_at_ms == 12.0
        assert hint.rule_id == "sample_a"

    def test_rule_fires_at_most_once(self):
        stager = _RecordingStager()
        d = _dispatcher(stager)
        fired: set[str] = set()

        d.fire("sample a", 1.0, fired)
        d.fire("sample a, and again sample a", 2.0, fired)

        assert len(stager.hints) == 1

    def test_broad_rule_is_recorded_but_not_prefetched(self):
        """Tier-2/3 rules must not dispatch: they can name thousands of
        files and starve the streaming loop."""
        stager = _RecordingStager()
        d = _dispatcher(stager)
        fired: set[str] = set()

        newly = d.fire("let me scan all files in the workspace", 5.0, fired)

        assert newly == ["everything"]      # recorded as fired
        assert stager.hints == []           # but nothing staged

    def test_no_match_is_a_no_op(self):
        stager = _RecordingStager()
        _dispatcher(stager).fire("nothing relevant here", 1.0, set())
        assert stager.hints == []

    def test_empty_text_is_a_no_op(self):
        stager = _RecordingStager()
        _dispatcher(stager).fire("", 1.0, set())
        assert stager.hints == []

    def test_matching_is_case_insensitive(self):
        stager = _RecordingStager()
        _dispatcher(stager).fire("SAMPLE A", 1.0, set())
        assert len(stager.hints) == 1


# --------------------------------------------------------------------------
# OpenAI wrapper, driven by a fake SDK stream (no network).
# --------------------------------------------------------------------------

class _FakeFn:
    def __init__(self, name="", arguments=""):
        self.name = name
        self.arguments = arguments


class _FakeToolCall:
    def __init__(self, index=0, id="", name="", arguments=""):
        self.index = index
        self.id = id
        self.function = _FakeFn(name, arguments)


class _FakeDelta:
    def __init__(self, reasoning=None, reasoning_content=None,
                 content=None, tool_calls=None):
        self.reasoning = reasoning
        self.reasoning_content = reasoning_content
        self.content = content
        self.tool_calls = tool_calls


class _FakeChunk:
    def __init__(self, delta):
        self.choices = [type("C", (), {"delta": delta})()]


def _response(chunks, stager=None):
    session = OpenAIStreamSession(
        started_at_ms=now_ms(),
        workspace_prior=PRIOR,
        ruleset=_RuleSet([_Rule("sample_a", r"sample\s*a", ("sample_a",))]),
        stager=stager,
    )
    return OpenAIStreamingResponse(sdk_stream=iter(chunks), session=session)


class TestOpenAIStreaming:
    def test_reasoning_delta_fires_the_detector(self):
        stager = _RecordingStager()
        r = _response([_FakeChunk(_FakeDelta(reasoning="read sample a now"))],
                      stager=stager)
        list(r.events())
        assert [h.rule_id for h in stager.hints] == ["sample_a"]

    def test_legacy_reasoning_content_field_also_works(self):
        """Older vLLM emits reasoning_content instead of reasoning."""
        stager = _RecordingStager()
        r = _response([_FakeChunk(_FakeDelta(reasoning_content="sample a"))],
                      stager=stager)
        list(r.events())
        assert len(stager.hints) == 1

    def test_server_without_reasoning_streams_but_stages_nothing(self):
        """OpenAI's own API returns no reasoning text, so there is nothing to
        detect against. The stream must still pass through cleanly."""
        stager = _RecordingStager()
        r = _response([_FakeChunk(_FakeDelta(content="here is the answer"))],
                      stager=stager)
        types = [e.type for e in r.events()]
        assert stager.hints == []
        assert "content_block_delta" in types
        assert types[-1] == "message_stop"

    def test_events_use_the_anthropic_shape(self):
        r = _response([_FakeChunk(_FakeDelta(reasoning="thinking..."))])
        events = list(r.events())
        start = next(e for e in events if e.type == "content_block_start")
        delta = next(e for e in events if e.type == "content_block_delta")
        assert start.content_block.type == "thinking"
        assert delta.delta.type == "thinking_delta"
        assert delta.delta.thinking == "thinking..."

    def test_tool_call_fragments_are_reassembled(self):
        r = _response([
            _FakeChunk(_FakeDelta(tool_calls=[
                _FakeToolCall(index=0, id="call_1", name="run_shell",
                              arguments='{"cmd":')])),
            _FakeChunk(_FakeDelta(tool_calls=[
                _FakeToolCall(index=0, arguments=' "ls"}')])),
        ])
        list(r.events())
        assert len(r.session.tool_calls) == 1
        call = r.session.tool_calls[0]
        assert call.name == "run_shell"
        assert call.id == "call_1"
        assert call.args == {"cmd": "ls"}

    def test_session_records_thinking_and_tool_timing(self):
        r = _response([
            _FakeChunk(_FakeDelta(reasoning="plan")),
            _FakeChunk(_FakeDelta(tool_calls=[
                _FakeToolCall(index=0, id="c", name="run_shell",
                              arguments="{}")])),
        ])
        list(r.events())
        s = r.session
        assert s.first_thinking_chunk_at_ms is not None
        assert s.first_tool_use_at_ms is not None
        assert s.slack_ms == pytest.approx(
            s.first_tool_use_at_ms - s.first_thinking_chunk_at_ms)
        assert s.thinking_blocks[0].text == "plan"

    def test_stream_without_a_stager_is_transparent(self):
        r = _response([_FakeChunk(_FakeDelta(reasoning="sample a"))], stager=None)
        assert [e.type for e in r.events()][-1] == "message_stop"


# --------------------------------------------------------------------------
# Gemini wrapper: regression test for the stager it used to accept and ignore.
# --------------------------------------------------------------------------

class _FakePart:
    def __init__(self, text=None, thought=False):
        self.text = text
        self.thought = thought
        self.function_call = None


class TestGeminiDispatch:
    def test_thinking_part_dispatches_to_the_stager(self):
        """GeminiClient accepted a stager and never called it. Reasoning
        deltas must now fire the detector like every other provider."""
        from agentstage.client.gemini import (
            GeminiStreamingResponse,
            GeminiStreamSession,
        )

        stager = _RecordingStager()
        session = GeminiStreamSession(
            started_at_ms=now_ms(),
            workspace_prior=PRIOR,
            ruleset=_RuleSet([_Rule("sample_a", r"sample\s*a", ("sample_a",))]),
            stager=stager,
        )
        r = GeminiStreamingResponse(sdk_stream=iter([]), session=session)

        list(r._handle_part(_FakePart(text="read sample a", thought=True), 7.0))

        assert [h.rule_id for h in stager.hints] == ["sample_a"]
        assert stager.hints[0].detected_files == PRIOR["sample_a"]

    def test_visible_text_does_not_dispatch(self):
        from agentstage.client.gemini import (
            GeminiStreamingResponse,
            GeminiStreamSession,
        )

        stager = _RecordingStager()
        session = GeminiStreamSession(
            started_at_ms=now_ms(),
            workspace_prior=PRIOR,
            ruleset=_RuleSet([_Rule("sample_a", r"sample\s*a", ("sample_a",))]),
            stager=stager,
        )
        r = GeminiStreamingResponse(sdk_stream=iter([]), session=session)

        list(r._handle_part(_FakePart(text="read sample a", thought=False), 7.0))

        assert stager.hints == []
