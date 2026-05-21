"""SessionDetector — multi-turn stateful detector wrapper.

Single-turn `run_detector` (engine.py) scans a flat list of blocks and
returns a Detection snapshot. For multi-turn agent sessions we need:

  - Activations accumulated across turns (turn-2 thinking may complete
    a pattern that turn-1 partially matched, etc.)
  - Per-turn deltas: "which NEW rules fired on this turn" so the
    AnthropicClient can dispatch only the new tier-1 hints to the stager
  - A per-turn record of which signal (thinking vs. tool_result) caused
    each activation, for the threats-to-validity analysis in the paper.

Usage:
    session = SessionDetector(prior=workload.workspace_prior, ruleset=ruleset)

    # Turn 1
    new_acts_1 = session.feed_turn(blocks_turn1)  # thinking + (maybe) tool_use
    for act in new_acts_1:
        if _tier_for_size(len(act.detected_files)) == 1:
            stager.prefetch(DataHint(...))

    # Between turns: tool execution happens, results captured
    new_acts_tr = session.feed_tool_results([...])  # tool_result blocks

    # Turn 2
    new_acts_2 = session.feed_turn(blocks_turn2)

    # At any time, the cumulative detection is available via:
    session.cumulative_detection()
"""

from __future__ import annotations

from dataclasses import dataclass, field

from agentstage.detector.engine import (
    Detection,
    RuleActivation,
    StreamBlock,
    _tier_for_size,
    run_detector,
)
from agentstage.detector.rules import RuleSet


@dataclass
class SessionDetector:
    """Stateful multi-turn detector.

    Maintains:
      - `all_blocks`: chronological list of all StreamBlocks fed so far
        (thinking + tool_result; we ignore "text" and "tool_use" because
        run_detector never scanned them)
      - `activations`: cumulative list of RuleActivation across turns
      - `fired_rule_names`: set of already-fired rule names (so feed_*
        returns only deltas)
      - `current_turn`: monotonically increasing turn counter
    """

    prior: dict[str, tuple[str, ...] | list[str]]
    ruleset: RuleSet
    all_blocks: list[StreamBlock] = field(default_factory=list)
    activations: list[RuleActivation] = field(default_factory=list)
    fired_rule_names: set[str] = field(default_factory=set)
    current_turn: int = 0

    def feed_blocks(self, new_blocks: list[StreamBlock]) -> list[RuleActivation]:
        """Append `new_blocks` to the session and return newly-fired rules.

        Each block's `turn` attribute is preserved if set; otherwise we
        stamp it with `self.current_turn`.

        Returns the list of activations that fired due to this call,
        with `turn` and `source` already populated by run_detector.
        """
        stamped: list[StreamBlock] = []
        for b in new_blocks:
            if b.turn == 0 and self.current_turn != 0:
                # Caller didn't stamp the turn — apply current_turn
                stamped.append(StreamBlock(
                    type=b.type, t_first=b.t_first, t_stop=b.t_stop,
                    text=b.text, chunks=b.chunks, tool_name=b.tool_name,
                    turn=self.current_turn,
                ))
            else:
                stamped.append(b)
        self.all_blocks.extend(stamped)

        # Re-run the detector on the full accumulated block list. This
        # is O(N · M) per call where N = total text so far, M = rules.
        # For ≤ 50 KB total text and ~100 rules this is sub-millisecond.
        pred = run_detector(self.all_blocks, self.prior, self.ruleset)

        # Diff against already-fired rules to find what's new
        new_acts: list[RuleActivation] = []
        for act in pred.activations:
            if act.rule_name not in self.fired_rule_names:
                new_acts.append(act)
                self.fired_rule_names.add(act.rule_name)
        # Replace cumulative activations with the fresh authoritative list
        self.activations = list(pred.activations)
        return new_acts

    def feed_turn(self, assistant_blocks: list[StreamBlock]) -> list[RuleActivation]:
        """Feed one assistant turn (thinking + text + tool_use blocks)
        and advance the turn counter."""
        # Stamp turn=current_turn on any unmarked blocks
        marked: list[StreamBlock] = []
        for b in assistant_blocks:
            if b.turn != self.current_turn:
                marked.append(StreamBlock(
                    type=b.type, t_first=b.t_first, t_stop=b.t_stop,
                    text=b.text, chunks=b.chunks, tool_name=b.tool_name,
                    turn=self.current_turn,
                ))
            else:
                marked.append(b)
        new_acts = self.feed_blocks(marked)
        self.current_turn += 1
        return new_acts

    def feed_tool_results(
        self, tool_result_blocks: list[StreamBlock]
    ) -> list[RuleActivation]:
        """Feed user-side tool_result blocks (between turns). These are
        stamped with `current_turn` so they appear as belonging to the
        next assistant turn's input context."""
        marked = [
            StreamBlock(
                type="tool_result", t_first=b.t_first, t_stop=b.t_stop,
                text=b.text, chunks=b.chunks, turn=self.current_turn,
            )
            for b in tool_result_blocks
        ]
        return self.feed_blocks(marked)

    def cumulative_detection(self) -> Detection:
        """Return a Detection snapshot for all blocks seen so far."""
        return run_detector(self.all_blocks, self.prior, self.ruleset)

    def tier1_activations(self) -> list[RuleActivation]:
        """Convenience: filter activations to those producing a tier-1
        detection (≤ 10 files). These are the ones we'd auto-dispatch
        to the stager."""
        return [
            a for a in self.activations
            if _tier_for_size(len(set(a.detected_files))) == 1
        ]


# Re-export for type-checking convenience
__all__ = ["SessionDetector"]
