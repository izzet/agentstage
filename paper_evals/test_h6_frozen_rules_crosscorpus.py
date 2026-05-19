"""H6: The frozen rule library generalizes across corpora.

Applying the FROZEN rule library (zero per-task tuning) to traces from
externally-released benchmarks — ScienceAgentBench (Chen et al., ICLR
2025) and KramaBench (Lai et al., 2025; MIT DB Lab preprint) — preserves
tier-1 byte recall ≥ 0.70. This is the L2 level of the §11.6 genericity
defense and the strongest argument that the predictor architecture is
corpus-agnostic.

KramaBench's preprint (not peer-reviewed) status is acknowledged in the
paper's external-benchmarks footnote; it was chosen over SWE-bench Lite
because its multi-domain raw-data-pipeline I/O profile (1.7 GB across
1764 files spanning 6 domains) is far closer to AgentStage's
scientific-HPC use case than SWE-bench's small-Python-file repos.

Serves: L2 genericity (E9, E10)
Origin: AGENTSTAGE.md §11.6 (three-level genericity verification)
Required data: end-to-end runs on SAB + KramaBench, captured via the
client library (monkey-patch of openai/anthropic SDKs) on Days 8-10.
Does NOT exist yet.
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


class TestFrozenRulesOnKramaBench:
    """E10: KramaBench end-to-end with frozen rules."""

    @pytest.mark.parametrize("domain", ["astronomy", "biomedical", "wildfire"])
    def test_kramabench_tier1_byte_recall_threshold(
        self, domain, outputs_root, io_report_root, rule_library_version, report
    ):
        """Tier-1 byte recall ≥ 0.70 on each KramaBench task in the 3-task
        subset (one each from Astronomy 1556 files / 486 MB, Biomedical
        7 files / 175 MB, Wildfire 23 files / 1 GB).

        Pass threshold for E10 per §11.6. Drop a domain only as
        risk-mitigation per §11.9.
        """
        pytest.skip(
            f"H6.kramabench_tier1_recall[{domain}]: pending — needs "
            "KramaBench integration on Day 9-10 (openai SDK monkey-patch in "
            "the KramaBench harness). Submodule lives at "
            "external/benchmarks/kramabench."
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
