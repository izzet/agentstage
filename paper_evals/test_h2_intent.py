"""H2: Streaming thinking content reveals file-access intent.

Models commit either to literal file paths (HOT scan layer) or to semantic
classes (file format, dataset region, processing stage) that map onto the
workspace prior via regex rules. The HOT layer is intentionally high-
precision and low-recall; the semantic-class rules carry the load.

Serves: C2, C5
Origin: AGENTSTAGE.md §3, §6.5 (HOT scan)
Required data: trace-only (--trace-root) + ground-truth (--ground-truth-root)
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h2


class TestThinkingContentPresence:
    """Most thinking-enabled probes do produce thinking content."""

    def test_thinking_seed_fraction(self, outputs_root, report):
        """At least 80% of probes produce non-empty thinking content.

        §5.5 reports 30 of 39 matrix probes produced thinking; §TLDR puts the
        well-defined subset at 47 turn-1 thinking seeds out of ~53. The
        denominator here is "probes where thinking was enabled at all."
        """
        pytest.skip(
            "H2.thinking_seed_fraction: pending — needs a way to read "
            "summary.json fields for 'thinking_present' and aggregate."
        )


class TestHotScanLayer:
    """The literal-path HOT scan is high-precision, low-recall by design."""

    def test_hot_precision_on_input_paths(
        self, outputs_root, io_report_root, report
    ):
        """When HOT fires (output paths excluded, unique-basename gate), it
        always names a file the agent will actually read.

        §6.5: HOT byte overfetch ≤ 1.5× on 100% of 30 thinking seeds.
        """
        pytest.skip(
            "H2.hot_precision: pending — depends on byte_metrics.json HOT "
            "block and per-seed ground truth (immediate-need set)."
        )

    def test_hot_recall_is_low_by_design(self, outputs_root, io_report_root):
        """HOT recall is low — corroborates C5 (literal-path commitment is
        unreliable). Threshold: < 0.35 byte recall on average across
        scientific workloads (it does fire reliably on code_repo).

        §6.5 reports overall HOT recall ≥ 0.85 only 17% of seeds (5/30).
        """
        pytest.skip(
            "H2.hot_recall_low: pending — requires per-workload HOT recall "
            "rollup so the code_repo exception is visible."
        )
