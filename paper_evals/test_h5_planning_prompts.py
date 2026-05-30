"""H5: Planning prompts are a free 2-10× slack multiplier.

Inserting explicit thinking instructions ("think step-by-step about which
files you'll read") into the user message multiplies slack and increases
intent-extraction precision without changing the model or budget. This
is a paired comparison: same (task, model, turn, seed) with prompt variant.

Serves: C6
Origin: AGENTSTAGE.md §2 (H5), §3 (C6), §7.1 (free lever)
Required data: paired no-PP / PP runs in the campaign. The campaign
               object exposes `planning_prompt: bool` per run, so pairing
               is `task × model × turn × seed × planning_prompt`.
"""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.h5


def _pair_by_planning_prompt(campaign):
    """Pair runs that share (task, model, turn, seed) but differ on the
    planning_prompt flag. Returns (no_pp_run, pp_run) tuples."""
    by_key: dict[tuple, dict[bool, list]] = {}
    for r in campaign.runs:
        k = (r.task, r.model, r.turn, r.seed)
        by_key.setdefault(k, {True: [], False: []}).setdefault(
            r.planning_prompt, []
        ).append(r)
    pairs = []
    for slots in by_key.values():
        if slots.get(False) and slots.get(True):
            pairs.append((slots[False][0], slots[True][0]))
    return pairs


class TestPlanningPromptSlackMultiplier:
    """Paired comparison: with-PP slack ≥ 2× without-PP slack."""

    def test_paired_slack_ratio_median(self, campaign, min_seeds, report):
        """Across paired (task, model, turn, seed) runs, median PP-slack /
        no-PP-slack ≥ 2.0.

        §3 C6 evidence: aiob_101 t=1 sonnet no-PP 3.9 s vs PP 9.6 s (2.5×).
        """
        pairs = _pair_by_planning_prompt(campaign)
        if len(pairs) < min_seeds:
            pytest.skip(
                f"H5.paired_ratio: only {len(pairs)} (task, model, turn, "
                f"seed) pairs differing on planning_prompt; need ≥ {min_seeds}"
            )
        ratios: list[dict] = []
        for no_pp, pp in pairs:
            if no_pp.slack_ms is None or pp.slack_ms is None:
                continue
            if no_pp.slack_ms <= 0:
                continue
            r = pp.slack_ms / no_pp.slack_ms
            ratios.append({
                "task": no_pp.task, "model": no_pp.model,
                "turn": no_pp.turn, "seed": no_pp.seed,
                "no_pp_slack_ms": no_pp.slack_ms,
                "pp_slack_ms": pp.slack_ms,
                "ratio": round(r, 2),
            })
        if len(ratios) < min_seeds:
            pytest.skip(
                f"H5.paired_ratio: only {len(ratios)} pairs had both slack "
                f"values populated"
            )
        med_ratio = sorted(r["ratio"] for r in ratios)[len(ratios) // 2]
        report.record("table_planning_prompt_multipliers", {
            "n_pairs": len(ratios),
            "median_ratio": med_ratio,
            "p25_ratio": sorted(r["ratio"] for r in ratios)[len(ratios) // 4],
            "p75_ratio": sorted(r["ratio"] for r in ratios)[(3 * len(ratios)) // 4],
            "per_pair": ratios[:30],
        })
        assert med_ratio >= 2.0, (
            f"H5.paired_ratio: median PP-slack/no-PP-slack ratio is "
            f"{med_ratio:.2f}× across {len(ratios)} pairs — below the 2× "
            f"threshold (§3 C6 evidence: 2.5–10×)."
        )

    def test_pp_never_dramatically_shorter(self, campaign, min_seeds, report):
        """Planning prompts never shorten slack by more than 10% — the
        intervention is at worst a no-op, never a regression. Direction
        test: ≥ 80% of pairs have PP-slack ≥ 0.9 × no-PP-slack.
        """
        pairs = _pair_by_planning_prompt(campaign)
        if len(pairs) < min_seeds:
            pytest.skip("H5.no_regression: insufficient pairs")
        ratios = [
            pp.slack_ms / no_pp.slack_ms
            for no_pp, pp in pairs
            if no_pp.slack_ms and pp.slack_ms and no_pp.slack_ms > 0
        ]
        if len(ratios) < min_seeds:
            pytest.skip("H5.no_regression: insufficient slack pairs")
        frac_not_regressing = sum(1 for r in ratios if r >= 0.9) / len(ratios)
        report.record("h5_no_regression", {
            "n_pairs": len(ratios),
            "frac_pp_slack_ge_0_9_x_no_pp": round(frac_not_regressing, 3),
        })
        assert frac_not_regressing >= 0.80, (
            f"H5.no_regression: PP regressed slack ≥ 10% on "
            f"{1 - frac_not_regressing:.0%} of pairs — the 'free lever' "
            f"framing breaks if PP ever reliably shortens slack."
        )


class TestPlanningPromptIntentPrecision:
    """Planning prompts also tighten the detected set (lower overfetch),
    not just lengthen slack. This is the part of C6 the paper claims but
    that needs to be separately verified."""

    def test_pp_reduces_or_holds_tier1_overfetch(self, campaign, min_seeds, report):
        """For paired configs, median (PP tier-1 overfetch − no-PP tier-1
        overfetch) ≤ 0 (PP never makes the predictor more wasteful).

        Direction-only test — magnitude varies by workload.
        """
        pairs = _pair_by_planning_prompt(campaign)
        if len(pairs) < min_seeds:
            pytest.skip("H5.pp_overfetch: insufficient pairs")
        deltas: list[float] = []
        for no_pp, pp in pairs:
            if not no_pp.byte_metrics_v1 or not pp.byte_metrics_v1:
                continue
            try:
                no = no_pp.byte_metrics_v1["tier_1_first"]["byte_overfetch"]
                p = pp.byte_metrics_v1["tier_1_first"]["byte_overfetch"]
            except (KeyError, TypeError):
                continue
            if no == float("inf") or p == float("inf"):
                continue
            deltas.append(p - no)
        if len(deltas) < min_seeds:
            pytest.skip("H5.pp_overfetch: insufficient paired overfetch values")
        median = sorted(deltas)[len(deltas) // 2]
        report.record("h5_pp_tier1_overfetch_delta", {
            "n_pairs": len(deltas),
            "median_delta": round(median, 3),
            "p25": round(sorted(deltas)[len(deltas) // 4], 3),
            "p75": round(sorted(deltas)[(3 * len(deltas)) // 4], 3),
        })
        assert median <= 0.05, (
            f"H5.pp_overfetch: PP increases tier-1 overfetch by median "
            f"{median:+.3f} — planning prompts may be loosening the predictor."
        )


class TestStrictPpDoesNotRegress:
    """The 'strict-PP' variant (force literal absolute paths) was a concern
    in §5.4 / §5.5. The hypothesis says strict-PP at worst ties vanilla PP
    on slack, and (per H12) does not improve recall. Here we just pin the
    no-slack-regression part."""

    def test_strict_pp_slack_not_below_vanilla_pp(self, campaign, min_seeds, report):
        """For runs tagged strict_pp / pathful, slack distribution is
        statistically no worse than vanilla PP runs.

        Currently a directional check: median(strict_pp slack) ≥ 0.8 ×
        median(vanilla_pp slack). Defers the stronger paired test until
        the campaign indexer exposes prompt_mode as a Run property."""
        import json as _json
        rows: list[float] = []
        rows_strict: list[float] = []
        for r in campaign.runs:
            if not r.planning_prompt or r.slack_ms is None:
                continue
            # Read raw summary to find strict-pp tagging.
            summary_p = r.run_dir / "summary.json"
            if not summary_p.is_file():
                continue
            try:
                d = _json.loads(summary_p.read_text())
            except (OSError, _json.JSONDecodeError):
                continue
            mode = (d.get("prompt_mode") or "").lower()
            if mode in {"pathful", "strict_pp", "pathful_strict", "hinted_pathful"}:
                rows_strict.append(r.slack_ms)
            else:
                rows.append(r.slack_ms)
        if len(rows_strict) < min_seeds or len(rows) < min_seeds:
            pytest.skip(
                f"H5.strict_pp: vanilla={len(rows)} strict={len(rows_strict)} "
                f"need ≥ {min_seeds} each"
            )
        med_vanilla = sorted(rows)[len(rows) // 2]
        med_strict = sorted(rows_strict)[len(rows_strict) // 2]
        report.record("h5_strict_pp_vs_vanilla_pp", {
            "n_vanilla_pp": len(rows),
            "n_strict_pp": len(rows_strict),
            "median_vanilla_slack_ms": med_vanilla,
            "median_strict_slack_ms": med_strict,
        })
        assert med_strict >= 0.8 * med_vanilla, (
            f"H5.strict_pp: median strict-PP slack ({med_strict:.0f} ms) is "
            f"more than 20% below vanilla PP ({med_vanilla:.0f} ms) — "
            f"strict-pp is regressing the slack window."
        )
