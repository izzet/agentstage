"""H10: The capture proxy adds negligible LLM-side latency, and the
no-thinking pathway is indistinguishable from the no-proxy baseline.

The proxy terminates the LLM SSE stream, parses events, runs the detector,
and forwards everything to the agent harness unchanged. Three claims:

  (E4)  Proxy overhead: p99 LLM-side latency with proxy vs without ≤ 1%.
  (E4b) Auto-rule decision cost: each detector-fire dispatch decision
        completes in ≤ 1 ms p95 — the cost stays inside the slack window.
  (E7)  Graceful degradation: when the model emits no thinking content
        (or the detector is disabled), end-to-end latency is identical
        to the no-AgentStage baseline.

If any fails, the proxy is unshippable in front of production LLM traffic.

Serves: E4, E7
Origin: sub-1% LLM-critical-path requirement
Required data:
  - outputs/microbench/auto_rules_cost.json (E-032) — auto-rule cost data
  - outputs/microbench/proxy_overhead_*.json — proxy LLM-side latency
    (LANDS WHEN PROXY MICROBENCH IS WRITTEN; currently pending)
  - outputs/no_thinking_pairs/ — paired runs with/without proxy for E7
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.h10


def _load_microbench(outputs_root: Path, name: str) -> dict | None:
    p = outputs_root / "microbench" / f"{name}.json"
    if not p.is_file():
        return None
    try:
        return json.loads(p.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def _load_proxy_overhead(outputs_root: Path) -> dict | None:
    """Locate the proxy LLM-side overhead microbench artifact. Convention:
    outputs/microbench/proxy_overhead.json or proxy_overhead_<ts>.json (latest).
    """
    mb = outputs_root / "microbench"
    if not mb.is_dir():
        return None
    cands = sorted(mb.glob("proxy_overhead*.json"))
    if not cands:
        return None
    try:
        return json.loads(cands[-1].read_text())
    except (OSError, json.JSONDecodeError):
        return None


class TestProxyOverhead:
    """E4: proxy overhead on the LLM critical path."""

    def test_p99_latency_overhead_below_1pct(self, outputs_root, report):
        """p99 LLM-side latency with proxy / without proxy ≤ 1.01.

        §4.1: "sub-1% on the LLM critical path. The proxy must not buffer;
        it forwards each event as it arrives."

        Records: `table_proxy_overhead` (mean, p50, p95, p99 with/without).
        """
        d = _load_proxy_overhead(outputs_root)
        if d is None:
            pytest.skip(
                "H10.p99_overhead: outputs/microbench/proxy_overhead*.json "
                "missing — write a proxy microbench that times paired "
                "SSE-pass-through with vs without the detector callback."
            )
        try:
            p99_with = d["with_proxy"]["p99_ms"]
            p99_without = d["without_proxy"]["p99_ms"]
        except (KeyError, TypeError):
            pytest.skip("H10.p99_overhead: artifact missing expected keys")
        ratio = p99_with / p99_without if p99_without > 0 else float("inf")
        report.record("table_proxy_overhead", {
            "p99_ms_with": p99_with,
            "p99_ms_without": p99_without,
            "p99_overhead_ratio": round(ratio, 4),
            "n_events": d.get("n_events"),
        })
        assert ratio <= 1.01, (
            f"H10.p99_overhead: proxy adds {(ratio - 1) * 100:+.2f}% to p99 "
            f"LLM-side latency (target: ≤ 1%). "
            f"with={p99_with:.2f} ms, without={p99_without:.2f} ms."
        )

    def test_auto_rule_decision_cost_under_1ms(self, outputs_root, report):
        """The detector's per-event decision cost (auto-rules generation +
        regex scan) stays within 1 ms p95 — it must fit inside the slack
        window without competing for budget. From E-032 (5 workloads ×
        1000 samples each), the campaign-wide p95 is ~297 µs.
        """
        d = _load_microbench(outputs_root, "auto_rules_cost")
        if d is None:
            pytest.skip(
                "H10.rule_cost: outputs/microbench/auto_rules_cost.json missing"
            )
        p95 = d.get("p95_us_across_workloads")
        if p95 is None:
            pytest.skip("H10.rule_cost: artifact lacks p95_us_across_workloads")
        report.record("h10_auto_rule_decision_cost", {
            "p50_us": d.get("p50_us_across_workloads"),
            "p95_us": p95,
            "max_us": d.get("max_us_observed"),
            "n_workloads": d.get("n_workloads"),
            "n_samples_per_workload": d.get("n_samples_per_workload"),
        })
        assert p95 <= 1000.0, (
            f"H10.rule_cost: detector p95 decision cost is {p95:.1f} µs — "
            f"exceeds the 1 ms slack-window budget."
        )

    def test_no_buffering_event_pacing(self, outputs_root, report):
        """SSE event arrival timestamps inside the proxy match upstream
        timestamps to within 5 ms — the proxy is not silently buffering
        and then bursting events.
        """
        d = _load_proxy_overhead(outputs_root)
        if d is None:
            pytest.skip("H10.event_pacing: proxy_overhead artifact missing")
        try:
            max_skew_ms = d["event_pacing"]["max_skew_ms"]
        except (KeyError, TypeError):
            pytest.skip(
                "H10.event_pacing: artifact lacks event_pacing.max_skew_ms — "
                "the microbench must record per-event upstream/proxy timestamps."
            )
        report.record("h10_proxy_event_pacing", {
            "max_skew_ms": max_skew_ms,
            "n_events": d.get("n_events"),
        })
        assert max_skew_ms <= 5.0, (
            f"H10.event_pacing: max upstream→proxy event skew is "
            f"{max_skew_ms:.2f} ms — the proxy is buffering."
        )


class TestGracefulDegradation:
    """E7: when no thinking is emitted, AgentStage is invisible."""

    def test_no_thinking_pathway_latency_identical(self, outputs_root, report):
        """For runs where the model emits zero thinking content, end-to-end
        latency CDF with AgentStage and without AgentStage are statistically
        indistinguishable (KS p > 0.05 OR p99 ratio ≤ 1.01).
        """
        d = _load_microbench(outputs_root, "no_thinking_pairs")
        if d is None:
            pytest.skip(
                "H10.no_thinking_baseline: outputs/microbench/no_thinking_pairs.json "
                "missing — needs paired no-thinking runs (e.g., Anthropic turn-2 "
                "from §6.8) with and without the proxy."
            )
        try:
            ks_p = d.get("ks_p_value")
            p99_ratio = d["p99_ratio_with_over_without"]
        except (KeyError, TypeError):
            pytest.skip("H10.no_thinking_baseline: artifact missing keys")
        report.record("h10_no_thinking_latency_parity", {
            "ks_p_value": ks_p,
            "p99_ratio": round(p99_ratio, 4),
            "n_paired_runs": d.get("n_paired_runs"),
        })
        assert (ks_p is not None and ks_p > 0.05) or p99_ratio <= 1.01, (
            f"H10.no_thinking_baseline: distributions differ (KS p={ks_p}, "
            f"p99 ratio={p99_ratio:.4f}) — graceful-degradation broken."
        )

    def test_no_data_corruption_when_detector_disabled(
        self, outputs_root, report
    ):
        """With --detector-disabled, the proxy is a pure SSE pass-through.
        Every event forwarded byte-identical to upstream.
        """
        d = _load_microbench(outputs_root, "detector_disabled_byte_identity")
        if d is None:
            pytest.skip(
                "H10.passthrough_byte_identity: outputs/microbench/"
                "detector_disabled_byte_identity.json missing — needs a "
                "byte-comparison test against an upstream SSE capture."
            )
        try:
            n_events = d["n_events_compared"]
            n_diffs = d["n_byte_differences"]
        except (KeyError, TypeError):
            pytest.skip("H10.passthrough: artifact missing keys")
        report.record("h10_passthrough_byte_identity", {
            "n_events_compared": n_events,
            "n_byte_differences": n_diffs,
        })
        assert n_diffs == 0, (
            f"H10.passthrough_byte_identity: {n_diffs} of {n_events} SSE "
            f"events differ byte-for-byte when detector is disabled — the "
            f"proxy is mutating traffic."
        )
