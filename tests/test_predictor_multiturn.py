"""Tests for the multi-turn / tool_result-aware predictor extensions.

These cover:
  - run_predictor scanning tool_result blocks alongside thinking
  - StreamBlock.turn / RuleActivation.source / RuleActivation.turn
  - blocks_from_messages() parsing a multi-turn Anthropic message history
"""

from __future__ import annotations

import re

from agentstage.predictor.engine import (
    StreamBlock,
    blocks_from_messages,
    run_predictor,
)
from agentstage.predictor.rules import Rule, RuleSet


def _ruleset_with(pattern: str, target_keys: tuple[str, ...] = ("k",)) -> RuleSet:
    return RuleSet(
        workload="test",
        rules=(
            Rule(
                name="rule_x",
                pattern=pattern,
                target_keys=target_keys,
                origin="general",
            ),
        ),
    )


def test_predictor_fires_on_tool_result_when_thinking_silent():
    """If the only place the pattern appears is the tool_result text,
    the predictor must still fire — and report source='tool_result'."""
    blocks = [
        StreamBlock(
            type="thinking", t_first=0.0, t_stop=100.0,
            text="Let me look at the workspace first.", chunks=1, turn=0,
        ),
        StreamBlock(
            type="tool_result", t_first=200.0, t_stop=200.0,
            text="/data/foo/bar_C08.nc\n/data/foo/bar_C09.nc\n", chunks=1, turn=1,
        ),
    ]
    prior = {"k": ("/data/foo/bar_C08.nc", "/data/foo/bar_C09.nc")}
    ruleset = _ruleset_with(r"bar_C08\.nc")
    pred = run_predictor(blocks, prior, ruleset)
    assert len(pred.activations) == 1
    act = pred.activations[0]
    assert act.source == "tool_result"
    assert act.turn == 1
    assert pred.tier_1.size == 2  # both files in target


def test_predictor_prefers_earliest_match_across_blocks():
    """If a rule pattern appears in thinking AND a later tool_result,
    the thinking activation wins (source='thinking', turn=0)."""
    blocks = [
        StreamBlock(
            type="thinking", t_first=0.0, t_stop=100.0,
            text="I need to read bar_C08.nc.", chunks=1, turn=0,
        ),
        StreamBlock(
            type="tool_result", t_first=200.0, t_stop=200.0,
            text="bar_C08.nc bar_C09.nc", chunks=1, turn=1,
        ),
    ]
    prior = {"k": ("/data/foo/bar_C08.nc",)}
    ruleset = _ruleset_with(r"bar_C08\.nc")
    pred = run_predictor(blocks, prior, ruleset)
    assert len(pred.activations) == 1
    assert pred.activations[0].source == "thinking"
    assert pred.activations[0].turn == 0


def test_text_blocks_scanned_with_text_source():
    """Visible assistant text blocks MUST be scanned for rule patterns.
    Multi-turn continuation responses often emit text-only (no thinking),
    so without this scan we'd miss the entire post-turn-0 signal."""
    blocks = [
        StreamBlock(
            type="text", t_first=0.0, t_stop=100.0,
            text="I will now read bar_C08.nc.", chunks=1, turn=1,
        ),
    ]
    prior = {"k": ("/data/foo/bar_C08.nc",)}
    ruleset = _ruleset_with(r"bar_C08\.nc")
    pred = run_predictor(blocks, prior, ruleset)
    assert len(pred.activations) == 1
    act = pred.activations[0]
    assert act.source == "text"
    assert act.turn == 1


def test_blocks_from_messages_chronological_order():
    """Multi-turn message history → blocks in chronological order with
    correct turn indices."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me explore /data."},
                {"type": "tool_use", "name": "list_dir", "input": {"path": "/data"}},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result",
                 "content": "bar_C08.nc bar_C09.nc bar_C10.nc"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Found the bands. Reading C08 first."},
                {"type": "tool_use", "name": "open_file",
                 "input": {"path": "/data/bar_C08.nc"}},
            ],
        },
    ]
    blocks = blocks_from_messages(messages)
    # Expect: thinking(turn=0), tool_use(0), tool_result(1), thinking(1), tool_use(1)
    types_turns = [(b.type, b.turn) for b in blocks]
    assert types_turns == [
        ("thinking", 0),
        ("tool_use", 0),
        ("tool_result", 1),
        ("thinking", 1),
        ("tool_use", 1),
    ]


def test_session_predictor_returns_only_new_activations_per_turn():
    """SessionPredictor.feed_turn must only return rules that fired
    on THIS call, not the cumulative set."""
    from agentstage.predictor.session import SessionPredictor
    prior = {"k": ("/data/foo/bar_C08.nc",)}
    ruleset = _ruleset_with(r"bar_C08\.nc")
    sp = SessionPredictor(prior=prior, ruleset=ruleset)

    # Turn 0: thinking does NOT mention bar_C08
    new_t0 = sp.feed_turn([
        StreamBlock(
            type="thinking", t_first=0.0, t_stop=100.0,
            text="Let me see what we have.", chunks=1,
        ),
    ])
    assert new_t0 == []

    # tool_result with the magic filename
    new_tr = sp.feed_tool_results([
        StreamBlock(
            type="tool_result", t_first=200.0, t_stop=200.0,
            text="bar_C08.nc found", chunks=1,
        ),
    ])
    assert len(new_tr) == 1
    assert new_tr[0].source == "tool_result"

    # Turn 1: thinking re-mentions the file — rule already fired, no new
    # activation should be reported
    new_t1 = sp.feed_turn([
        StreamBlock(
            type="thinking", t_first=300.0, t_stop=400.0,
            text="Reading bar_C08.nc now.", chunks=1,
        ),
    ])
    assert new_t1 == []
    assert len(sp.activations) == 1


def test_session_predictor_turn_indices_correct():
    """Turn counter must increment across feed_turn calls; tool_results
    stamped with the NEXT turn's index."""
    from agentstage.predictor.session import SessionPredictor
    prior = {"k": ("/data/a.nc",)}
    ruleset = _ruleset_with(r"a\.nc")
    sp = SessionPredictor(prior=prior, ruleset=ruleset)

    sp.feed_turn([
        StreamBlock(type="thinking", t_first=0.0, t_stop=100.0,
                    text="exploring", chunks=1),
    ])
    assert sp.current_turn == 1
    new_tr = sp.feed_tool_results([
        StreamBlock(type="tool_result", t_first=150.0, t_stop=150.0,
                    text="a.nc", chunks=1),
    ])
    assert new_tr[0].turn == 1  # tool_results belong to upcoming turn
    sp.feed_turn([
        StreamBlock(type="thinking", t_first=200.0, t_stop=300.0,
                    text="reading", chunks=1),
    ])
    assert sp.current_turn == 2


def test_multi_turn_replay_through_run_predictor():
    """End-to-end: build a multi-turn message history, parse into blocks,
    run predictor. Confirm rule fires on the right turn and source."""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Need to see what's in /data."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "content": "raw/C08.nc raw/C09.nc"},
            ],
        },
    ]
    prior = {"all_bands": ("raw/C08.nc", "raw/C09.nc")}
    ruleset = _ruleset_with(r"C08\.nc", target_keys=("all_bands",))
    blocks = blocks_from_messages(messages)
    pred = run_predictor(blocks, prior, ruleset)
    assert len(pred.activations) == 1
    act = pred.activations[0]
    assert act.source == "tool_result"
    assert act.turn == 1
