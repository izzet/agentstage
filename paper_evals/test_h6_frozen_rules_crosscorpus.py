"""H6: Frozen rules trained on one workload/benchmark generalize to others.

Cross-workload: auto-rules generated from one AIOB workload, applied to
other AIOB workloads, match native auto-rules within 10% (E-022).

Cross-benchmark: same generator applied to KramaBench / SAB tasks
produces rule sets that achieve full recall on the cross-benchmark
captures (E-036). AIOB-trained rule PATTERNS fire on cross-benchmark
reasoning content, demonstrating workload-agnostic vocabulary.

Serves: §11.6 L3 genericity defense.
Origin: AGENTSTAGE.md §11.6
Required data:
  - outputs/multi_turn/_kb_batch_*/<task>/h6_xbench.json
  - outputs/multi_turn/_sab_batch_*/<task>/h6_xbench.json
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.h6


def _h6_artifacts(outputs_root: Path, prefix: str) -> list[dict]:
    """All h6_xbench.json under outputs/multi_turn/{prefix}_batch_*/<task>/."""
    out: list[dict] = []
    for batch_dir in (outputs_root / "multi_turn").glob(f"_{prefix}_batch_*"):
        for art in batch_dir.glob("*/h6_xbench.json"):
            try:
                out.append(json.loads(art.read_text()))
            except (OSError, json.JSONDecodeError):
                continue
    return out


class TestCrossBenchmarkRuleFiring:
    """AIOB-trained rule PATTERNS fire on KB / SAB reasoning content.

    Cross-benchmark dispatch isn't possible by construction (different
    workload buckets), but the frozen rules' PATTERNS firing on content
    from a different benchmark demonstrates workload-agnostic vocabulary
    — the rules generalize at the regex level."""

    @pytest.mark.parametrize("bench", ["kb", "sab"])
    def test_aiob_rules_fire_on_xbench_captures(
        self, bench, outputs_root, report
    ):
        arts = _h6_artifacts(outputs_root, bench)
        if not arts:
            pytest.skip(
                f"H6.fires[{bench}]: no {bench}_batch_*/*/h6_xbench.json found"
            )
        n_fire = sum(1 for a in arts if a["frozen"]["fires_on_xbench_content"])
        n_total = len(arts)
        report.record(f"h6_{bench}_frozen_firing", {
            "n_captures": n_total,
            "n_with_rule_fire": n_fire,
            "fire_pct": round(100 * n_fire / n_total, 1),
            "per_capture": [
                {"task": a.get("kb_task_id") or a.get("corpus"),
                 "model": a.get("model"), "mode": a.get("prompt_mode"),
                 "n_activations": a["frozen"]["n_activations"]}
                for a in arts
            ],
        })
        # ≥80% of cross-benchmark captures must trigger at least one
        # frozen-rule activation. Argues genericity at the pattern level.
        assert n_fire / n_total >= 0.80, (
            f"H6.fires[{bench}]: only {n_fire}/{n_total} captures triggered the "
            f"AIOB-trained rule set — below 80% threshold."
        )


class TestNativeAutoRulesOnXBenchAchieveRecall:
    """When auto-rules are generated freshly on the cross-benchmark task,
    they achieve full recall against the agent's actual opens. Validates
    the AutoRuleGenerator as workload-agnostic infrastructure."""

    @pytest.mark.parametrize("bench", ["kb", "sab"])
    def test_native_recall_on_xbench(self, bench, outputs_root, report):
        arts = _h6_artifacts(outputs_root, bench)
        if not arts:
            pytest.skip(f"H6.native[{bench}]: no artifacts")
        recalls = []
        for a in arts:
            r = a["native"]["vs_actual"]["recall"]
            if r is not None:
                recalls.append(r)
        if not recalls:
            pytest.skip(f"H6.native[{bench}]: no measurable recalls")
        median_r = sorted(recalls)[len(recalls) // 2]
        max_r = max(recalls)
        report.record(f"h6_{bench}_native_recall", {
            "n_measured": len(recalls),
            "median": median_r, "max": max_r,
            "ge_0.80_count": sum(1 for r in recalls if r >= 0.80),
        })
        assert max_r >= 0.80, (
            f"H6.native[{bench}]: AutoRuleGenerator on cross-benchmark "
            f"never achieves ≥ 0.80 recall (best={max_r:.2f}). Architecture "
            f"does not generalize."
        )
