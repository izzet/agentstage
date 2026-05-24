"""H7: No single auto-generated rule is load-bearing.

Leave-one-out: drop each rule from the auto-generated ruleset in turn
and re-replay a captured multi-turn corpus. A ruleset is robust if no
single rule's removal drops recall by ≥ 10pp.

Tested on captures where the full ruleset already achieves recall ≥ 0.80
(precondition for LOO to be meaningful — if the full set has 0 recall,
dropping a rule changes nothing and the test is uninformative).

Serves: §11.6 (no-single-point-of-failure)
Origin: AGENTSTAGE.md §11.6
Required data: outputs/multi_turn/<corpus>/h7_loo.json from E-034.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.h7


def _h7_artifacts(outputs_root: Path) -> list[dict]:
    out: list[dict] = []
    for art in (outputs_root / "multi_turn").glob("*/h7_loo.json"):
        try:
            d = json.loads(art.read_text())
            d["_artifact_path"] = str(art)
            out.append(d)
        except (OSError, json.JSONDecodeError):
            continue
    return out


class TestNoSingleRuleLoadBearing:
    """For every corpus where the full ruleset achieves ≥ 0.80 recall,
    LOO shows no single rule's removal drops recall by ≥ 10pp."""

    def test_loo_robust_on_passing_corpora(self, outputs_root, report):
        arts = _h7_artifacts(outputs_root)
        if not arts:
            pytest.skip("H7: no h7_loo.json artifacts under outputs/multi_turn/")
        passing = [a for a in arts
                   if a.get("full_ruleset_recall", 0) >= 0.80
                   and a.get("n_agent_opens", 0) > 0]
        if not passing:
            pytest.skip(
                "H7: no corpora with full_ruleset_recall ≥ 0.80 — "
                "LOO is uninformative on 0-recall captures."
            )
        violations = []
        for a in passing:
            for r in a.get("per_rule", []):
                if r["delta_from_full"] >= 0.10:
                    violations.append({
                        "corpus": Path(a["corpus"]).name,
                        "rule": r["dropped"],
                        "delta": r["delta_from_full"],
                    })
        report.record("h7_loo_summary", {
            "n_artifacts_total": len(arts),
            "n_passing_full_recall": len(passing),
            "n_load_bearing_violations": len(violations),
            "violations": violations[:10],
        })
        assert not violations, (
            f"H7: {len(violations)} rules are load-bearing across "
            f"{len(passing)} passing corpora (top: {violations[:3]})"
        )

    def test_ruleset_robust_min_recall(self, outputs_root, report):
        """Every passing corpus's worst-case LOO recall stays ≥ 0.80."""
        arts = _h7_artifacts(outputs_root)
        passing = [a for a in arts
                   if a.get("full_ruleset_recall", 0) >= 0.80
                   and a.get("n_agent_opens", 0) > 0]
        if not passing:
            pytest.skip("H7.min_recall: no passing corpora")
        weak = [a for a in passing if a["min_recall_after_drop"] < 0.80]
        report.append("h7_min_recall_after_drop", [
            {"corpus": Path(a["corpus"]).name,
             "min_recall_after_drop": a["min_recall_after_drop"]}
            for a in passing
        ])
        assert not weak, (
            f"H7: {len(weak)} corpora drop below 0.80 recall when ANY single "
            f"rule is removed."
        )
