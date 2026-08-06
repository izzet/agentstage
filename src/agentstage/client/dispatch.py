"""Provider-agnostic pieces of the live capture path.

Every client wrapper does the same thing once a thinking delta arrives:
match not-yet-fired rules against the accumulated thinking text and hand
tier-1 hits to the stager. That logic lives here so the Anthropic, Gemini,
and OpenAI wrappers share one implementation rather than three.

Also holds the event shims the non-Anthropic wrappers use to present
Anthropic-SDK-shaped events, so a harness can swap providers without
rewriting its dispatch loop.
"""

from __future__ import annotations

import re
from typing import Any

from agentstage.stager import DataHint, Stager


def tier_for_size(n: int) -> int:
    """Tier a rule by the size of the file set it resolves to.

    Note this is deliberately tighter than the offline classifier in
    `detector.engine`, which admits up to 20 files to tier-1. The live path
    dispatches eagerly on the LLM critical path, so it stays conservative.
    """
    if n <= 10:
        return 1
    if n <= 200:
        return 2
    return 3


class RuleDispatcher:
    """Compiles a RuleSet once, then fires unfired rules against accumulated
    thinking text and dispatches tier-1 matches to the stager.

    Detection-equivalent to `engine.run_detector`: a rule fires iff its regex
    matches the accumulated thinking text. The difference is cost. This runs
    one `regex.search` per not-yet-fired rule per chunk, O(N*M), where the
    offline engine re-evaluates every character prefix to interpolate an exact
    activation timestamp, O(N^2*M). On the live path that interpolation is
    unnecessary: the activation time is the chunk's arrival time, which is the
    moment we learned the file was needed.
    """

    def __init__(
        self,
        *,
        ruleset: Any | None,
        workspace_prior: dict[str, tuple[str, ...] | list[str]] | None,
        stager: Stager | None,
    ) -> None:
        self._ruleset = ruleset
        self._workspace_prior = workspace_prior or {}
        self._stager = stager
        self._compiled: list[tuple[Any, re.Pattern[str]]] | None = None

    @property
    def enabled(self) -> bool:
        """False when there is nothing to match against or nowhere to stage."""
        return self._ruleset is not None and self._stager is not None

    def fire(self, accumulated: str, t_ms: float, fired: set[str]) -> list[str]:
        """Match `accumulated` against every rule not already in `fired`.

        Tier-1 hits are dispatched to the stager. Tier-2/3 hits are recorded
        as fired but not prefetched: a broad rule can name thousands of files
        and starve the streaming loop.

        `fired` is mutated with the newly-fired rule names, which are also
        returned.
        """
        if not self.enabled or not accumulated:
            return []
        if self._compiled is None:
            self._compiled = [
                (r, re.compile(r.pattern, flags=re.IGNORECASE))
                for r in self._ruleset.rules  # type: ignore[union-attr]
            ]

        newly: list[str] = []
        for rule, regex in self._compiled:
            if rule.name in fired:
                continue
            if regex.search(accumulated) is None:
                continue
            fired.add(rule.name)
            newly.append(rule.name)

            detected = tuple(
                p
                for k in rule.target_keys
                for p in self._workspace_prior.get(k, ())
            )
            if tier_for_size(len(detected)) > 1:
                continue
            self._stager.prefetch(DataHint(  # type: ignore[union-attr]
                detected_files=detected,
                tier=1,
                fired_at_ms=t_ms,
                rule_id=rule.name,
                byte_estimate=0,
            ))
        return newly


class Event:
    """Lightweight namespace mimicking the Anthropic SDK event shape.

    Lets a harness consume any provider's stream without rewriting its
    event-handling code.
    """

    __slots__ = ("type", "index", "content_block", "delta")

    def __init__(self, type: str, index: int = -1,
                 content_block: Any = None, delta: Any = None) -> None:
        self.type = type
        self.index = index
        self.content_block = content_block
        self.delta = delta


class ContentBlock:
    __slots__ = ("type", "name", "id")

    def __init__(self, type: str, name: str = "", id: str = "") -> None:
        self.type = type
        self.name = name
        self.id = id


class Delta:
    __slots__ = ("type", "thinking", "text", "partial_json", "signature")

    def __init__(self, type: str, thinking: str = "", text: str = "",
                 partial_json: str = "", signature: str = "") -> None:
        self.type = type
        self.thinking = thinking
        self.text = text
        self.partial_json = partial_json
        self.signature = signature
