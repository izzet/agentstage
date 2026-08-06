"""Freeze contract for the AgentStage semantic-class rule library.

These tests pin `RULE_LIBRARY_HASH`, `RULE_LIBRARY_VERSION`, per-workload
rule counts, and origin distribution. The library is FROZEN — any change
to any rule's name, pattern, target_keys, or origin bumps the hash, fails
this test, and forces a deliberate `RULE_LIBRARY_VERSION` bump documented
in the commit message.

Without this gate, silent rule edits would invalidate the cross-corpus
genericity claim without anyone noticing. The
paper_evals test `test_h6_frozen_rules_crosscorpus.py` consumes
`RULE_LIBRARY_VERSION` via `--rule-library-version`; this unit test pins
the bytes underneath that version string.

To bump deliberately:
  1. Edit `src/agentstage/detector/rules.py`.
  2. Bump `RULE_LIBRARY_VERSION` (e.g. v1 → v2).
  3. Run this test, copy the new hash from the failure message.
  4. Update `EXPECTED_HASH` and `EXPECTED_VERSION` below.
  5. Commit with a message documenting the change and the genericity-
     impact assessment.
"""

from __future__ import annotations

from agentstage.detector import (
    ALL_RULESETS,
    RULE_LIBRARY_HASH,
    RULE_LIBRARY_VERSION,
    Rule,
    RuleSet,
    rule_count_by_origin,
)


# ---------------------------------------------------------------------------
# THE FREEZE PINS — DO NOT EDIT WITHOUT BUMPING RULE_LIBRARY_VERSION
# ---------------------------------------------------------------------------

EXPECTED_VERSION = "v1"
EXPECTED_HASH = "e8d8dfc6dda24b01b191b6fd894be920814a36148154f11355eeea6cb1da63c7"

EXPECTED_RULE_COUNTS = {
    "aiob_101": 8,
    "aiob_104": 58,   # 50 per-sample mention rules + 8 semantic
    "aiob_107": 10,
    "aiob_110": 16,   # 10 per-subject mention rules + 6 semantic
    "code_repo": 13,
}

EXPECTED_ORIGIN_DISTRIBUTION = {
    "aiob_101": 8,
    "aiob_104": 55,
    "aiob_107": 7,
    "aiob_110": 13,
    "code_repo": 6,
    "general": 16,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_rule_library_version_pinned():
    """The version string is the public name of the freeze."""
    assert RULE_LIBRARY_VERSION == EXPECTED_VERSION, (
        f"RULE_LIBRARY_VERSION drifted from {EXPECTED_VERSION!r} to "
        f"{RULE_LIBRARY_VERSION!r}. Update EXPECTED_VERSION here and document "
        "the bump in the commit message."
    )


def test_rule_library_hash_pinned():
    """SHA-256 over the canonical rule serialization. Bumps on ANY rule edit."""
    assert RULE_LIBRARY_HASH == EXPECTED_HASH, (
        f"RULE_LIBRARY_HASH drifted.\n"
        f"  expected: {EXPECTED_HASH}\n"
        f"  actual:   {RULE_LIBRARY_HASH}\n"
        f"\nThe rule library changed. If this was intentional:\n"
        f"  1. Bump RULE_LIBRARY_VERSION in src/agentstage/detector/rules.py\n"
        f"  2. Update EXPECTED_HASH above to: {RULE_LIBRARY_HASH}\n"
        f"  3. Document the change and its cross-corpus-genericity impact "
        f"in the commit message.\n"
        f"If it was unintentional, revert the rule edit."
    )


def test_per_workload_rule_counts():
    """Lock the number of rules per workload. Catches accidental rule
    additions or deletions that wouldn't necessarily change the hash
    if they're additions of rules with already-present fingerprints
    (unlikely but defensive)."""
    actual = {w: len(rs) for w, rs in ALL_RULESETS.items()}
    assert actual == EXPECTED_RULE_COUNTS, (
        f"Per-workload rule counts drifted:\n"
        f"  expected: {EXPECTED_RULE_COUNTS}\n"
        f"  actual:   {actual}"
    )


def test_origin_distribution_pinned():
    """Lock how many rules each origin tag covers. Critical for the
    leave-one-out (E3) genericity test: changing a rule's origin from
    'aiob_104' to 'general' (or vice versa) changes which rules
    survive each held-out fold."""
    actual = rule_count_by_origin()
    assert actual == EXPECTED_ORIGIN_DISTRIBUTION, (
        f"Origin distribution drifted:\n"
        f"  expected: {EXPECTED_ORIGIN_DISTRIBUTION}\n"
        f"  actual:   {actual}"
    )


def test_rule_names_unique_within_workload():
    """A rule's `name` is the activation key in `detection.json`.
    Duplicates within a workload would silently drop activations."""
    for workload, rs in ALL_RULESETS.items():
        names = [r.name for r in rs.rules]
        assert len(names) == len(set(names)), (
            f"Duplicate rule name in {workload}: {sorted(n for n in names if names.count(n) > 1)}"
        )


def test_rules_are_immutable():
    """Rule and RuleSet are frozen dataclasses — mutation is a programming
    error. Verify the frozen contract at runtime as a smoke test."""
    sample = next(iter(ALL_RULESETS.values())).rules[0]
    import dataclasses
    try:
        dataclasses.replace(sample, name="mutated")  # this is allowed; creates a new instance
    except Exception as exc:
        raise AssertionError(f"dataclasses.replace failed: {exc}") from exc
    try:
        sample.name = "mutated"  # this MUST fail on a frozen dataclass
    except dataclasses.FrozenInstanceError:
        pass
    else:
        raise AssertionError("Rule is not frozen — direct attribute assignment succeeded")


def test_all_rules_use_valid_origins():
    """Defensive: an unrecognized origin string would bypass leave-one-out
    filtering (since `excl` is a closed set of names from CAMPAIGN.md).
    Verify every rule's origin is in the known set."""
    valid_origins = {"aiob_101", "aiob_104", "aiob_107", "aiob_110", "code_repo", "general"}
    for workload, rs in ALL_RULESETS.items():
        for r in rs.rules:
            assert r.origin in valid_origins, (
                f"Rule {workload}.{r.name} has unknown origin {r.origin!r}; "
                f"must be one of {sorted(valid_origins)}"
            )


def test_leave_one_out_filter_drops_target_origin_keeps_general():
    """Sanity-check the leave-one-out filter: excluding aiob_104 drops
    aiob_104-tagged rules but keeps 'general' rules and rules from
    other workloads."""
    aiob_104 = ALL_RULESETS["aiob_104"]
    filtered = aiob_104.filter_by_origin({"aiob_104"})
    # All rules with origin == "aiob_104" should be gone
    for r in filtered.rules:
        assert r.origin != "aiob_104", f"Filter failed to drop {r.name}"
    # At least some "general" rules from aiob_104 should remain (first_inspect, all_samples_signal, report_out)
    general_remaining = {r.name for r in filtered.rules if r.origin == "general"}
    assert "first_inspect" in general_remaining
    assert "all_samples_signal" in general_remaining
    assert "report_out" in general_remaining


def test_ruleset_iteration_order_is_stable():
    """RuleSet iteration order is part of the freeze contract — first-fire
    wins on identical char offsets. Verify iteration matches construction
    order."""
    for workload, rs in ALL_RULESETS.items():
        seen = []
        for r in rs:
            assert isinstance(r, Rule)
            seen.append(r.name)
        assert seen == [r.name for r in rs.rules]
