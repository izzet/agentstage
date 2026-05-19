"""H6: The frozen rule library generalizes across corpora.

Applying the FROZEN rule library (zero per-task tuning) to traces from
externally-published benchmarks — ScienceAgentBench (Chen et al. ICLR 2025)
and SWE-bench Lite (Jimenez et al. ICLR 2024) — preserves tier-1 byte
recall ≥ 0.70. This is the L2 level of the §11.6 genericity defense and
the strongest argument that the predictor architecture is corpus-agnostic.

Serves: L2 genericity (E9, E10)
Origin: AGENTSTAGE.md §11.6 (three-level genericity verification)
Required data: trace-only on SAB + SWE-bench corpora, captured via the
proxy on Day 8-10. Does NOT exist yet.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h6


class TestFrozenRulesOnScienceAgentBench:
    """E9: ScienceAgentBench end-to-end with frozen rules."""

    def test_sab_tier1_byte_recall_threshold(
        self, outputs_root, io_report_root, rule_library_version, report
    ):
        """Tier-1 byte recall ≥ 0.70 on each SAB task in the 3-5 task subset.

        Pass threshold for E9 per §11.6 ("modest degradation from AIOB 0.85+
        is acceptable; below 0.70 is a genericity failure").
        """
        pytest.skip(
            "H6.sab_tier1_recall: pending — needs SAB integration on Day 8-9 "
            "(proxy routing + ground-truth extraction from SAB task spec). "
            "Submodule lives at external/benchmarks/scienceagentbench."
        )


class TestFrozenRulesOnSWEbenchLite:
    """E10: SWE-bench Lite end-to-end with frozen rules."""

    def test_swebench_tier1_byte_recall_threshold(
        self, outputs_root, io_report_root, rule_library_version, report
    ):
        """Tier-1 byte recall ≥ 0.70 on each SWE-bench Lite instance in the
        representative-repo set.

        Pass threshold for E10. Drop to 1 instance only as risk-mitigation
        per §11.9.
        """
        pytest.skip(
            "H6.swebench_tier1_recall: pending — needs SWE-bench Lite "
            "integration on Day 9-10 (container harness routing). "
            "Submodule lives at external/benchmarks/swebench."
        )


class TestRuleLibraryFreezeContract:
    """The rule library used here is the one that was frozen on Day 1 —
    no per-corpus tuning. Defended by version pin."""

    def test_rule_library_version_matches_pin(
        self, rule_library_version, report
    ):
        """If --rule-library-version is provided, the loaded rule library's
        version (hash or semver) must match. This is the on-the-wire check
        that we're not silently tuning rules between corpus runs.
        """
        pytest.skip(
            "H6.freeze_contract: pending — depends on Day-1 rule freeze "
            "(agentstage.predictor.rules.RULE_LIBRARY_VERSION). The same "
            "version hash gets unit-tested in tests/test_rules_freeze.py."
        )
