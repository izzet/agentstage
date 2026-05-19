"""H10: The capture proxy adds negligible LLM-side latency, and the
no-thinking pathway is indistinguishable from the no-proxy baseline.

The proxy terminates the LLM SSE stream, parses events, runs the predictor,
and forwards everything to the agent harness unchanged. Two distinct claims:

  (E4) Proxy overhead: p99 LLM-side latency with proxy vs without ≤ 1%.
  (E7) Graceful degradation: when the model emits no thinking content (or
       the proxy is disabled), end-to-end latency is identical to the
       no-AgentStage baseline.

If either fails, the proxy is unshippable in front of production LLM traffic.

Serves: E4, E7
Origin: AGENTSTAGE.md §11.5, §4.1 (sub-1% requirement)
Required data: proxy microbench output under --staging-root (or a dedicated
--proxy-bench-root). Land alongside the proxy build on Day 3-4.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h10


class TestProxyOverhead:
    """E4: proxy overhead on the LLM critical path."""

    def test_p99_latency_overhead_below_1pct(self, staging_root, report):
        """p99 LLM-side latency with proxy / without proxy ≤ 1.01.

        §4.1 sets the requirement: "sub-1% on the LLM critical path. The
        proxy must not buffer; it forwards each event as it arrives."

        Records: `table_proxy_overhead` (mean, p50, p95, p99 with/without).
        """
        if staging_root is None:
            pytest.skip("H10 requires --staging-root (proxy microbench output)")
        pytest.skip(
            "H10.p99_overhead: pending — Day 4 proxy microbench."
        )

    def test_no_buffering_event_pacing(self, staging_root):
        """SSE event arrival timestamps inside the proxy match upstream
        timestamps to within 5 ms — the proxy is not silently buffering
        and then bursting events.
        """
        if staging_root is None:
            pytest.skip("H10 requires --staging-root")
        pytest.skip(
            "H10.event_pacing: pending — needs proxy-side timestamp capture."
        )


class TestGracefulDegradation:
    """E7: when no thinking is emitted, AgentStage is invisible."""

    def test_no_thinking_pathway_latency_identical(
        self, staging_root, report
    ):
        """For runs where the model emits zero thinking content, end-to-end
        latency CDF with AgentStage and without AgentStage are statistically
        indistinguishable (KS p > 0.05).
        """
        if staging_root is None:
            pytest.skip("H10 requires --staging-root")
        pytest.skip(
            "H10.no_thinking_baseline: pending — needs paired no-thinking "
            "runs (e.g. Anthropic turn-2 from §6.8) with and without proxy."
        )

    def test_no_data_corruption_when_predictor_disabled(self, staging_root):
        """With --predictor-disabled, the proxy is a pure SSE pass-through.
        Every event forwarded byte-identical to upstream.
        """
        if staging_root is None:
            pytest.skip("H10 requires --staging-root")
        pytest.skip(
            "H10.passthrough_byte_identity: pending — needs --predictor-disabled "
            "flag in the proxy CLI."
        )
