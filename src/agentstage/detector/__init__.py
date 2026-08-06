"""AgentStage's intent detector: turns streaming thinking text into a
tiered file working-set detection.

The detector is composed of three layers:

1. Workspace prior — files known to exist at the agent's current point
   in the task (loaded per-workload from `agentstage.workloads`).
2. HOT scan — substring search across thinking text for literal file
   paths from the workspace prior (high-precision, low-recall).
3. Semantic rules — regex-defined activations mapping thinking text
   content to subsets of the workspace prior, tiered by target-set
   size (specific ≤20, medium ≤200, broad >200).

The rule library is FROZEN — see `RULE_LIBRARY_VERSION` and
`RULE_LIBRARY_HASH` for the freeze contract enforced by
`tests/test_rules_freeze.py`.
"""

from agentstage.detector.rules import (
    AIOB_104_SAMPLES,
    AIOB_110_SUBJECTS,
    ALL_RULESETS,
    RULE_LIBRARY_HASH,
    RULE_LIBRARY_VERSION,
    RULES_AIOB_101,
    RULES_AIOB_104,
    RULES_AIOB_107,
    RULES_AIOB_110,
    RULES_CODE_REPO,
    Rule,
    RuleOrigin,
    RuleSet,
    get_ruleset,
    rule_count_by_origin,
)

__all__ = [
    "AIOB_104_SAMPLES",
    "AIOB_110_SUBJECTS",
    "ALL_RULESETS",
    "RULE_LIBRARY_HASH",
    "RULE_LIBRARY_VERSION",
    "RULES_AIOB_101",
    "RULES_AIOB_104",
    "RULES_AIOB_107",
    "RULES_AIOB_110",
    "RULES_CODE_REPO",
    "Rule",
    "RuleOrigin",
    "RuleSet",
    "get_ruleset",
    "rule_count_by_origin",
]
