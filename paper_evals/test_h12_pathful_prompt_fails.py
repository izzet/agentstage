"""H12: Forcing the agent to emit literal file paths via prompting does
NOT improve prediction recall — auto-rules over semantic intent are required.

The naïve approach to the path-extraction challenge (C1 in §1 P4) is to
add an instruction like "before any tool call, enumerate the absolute
paths you intend to read." The hypothesis under test: this does not work.
Models either ignore the instruction or paraphrase paths rather than
emitting them literally; in either case, the literal-path HOT scan does
not gain meaningful recall, while the semantic-class rules (auto-rules)
remain the load-bearing predictor.

Paper role: this is the empirical justification for the auto-rule
generator architecture (§4 predictor). If pathful prompting worked, the
predictor would not need rules at all.

Serves: C1 (path extraction), justifies §4 auto-rule generator
Origin: HOT scan, pathful-prompt experiment
Required data: paired runs differing only in prompt_mode ∈ {default, hinted,
               pathful, sparse}. Pathful directories under
               outputs/multi_turn/e020_multiturn_hinted_pathful_* contain
               the strict-PP variants.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.h12


_PATHFUL_VARIANTS = {"pathful", "hinted_pathful", "strict_pp", "pathful_strict"}
_BASELINE_VARIANTS = {"default", "hinted", "none", "sparse"}


def _prompt_mode(run) -> str | None:
    """Read prompt_mode from the raw summary, since RunResult doesn't
    expose it as a property (yet)."""
    summary_path = run.run_dir / "summary.json"
    if not summary_path.is_file():
        return None
    try:
        d = json.loads(summary_path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return d.get("prompt_mode") or d.get("prompt_variant") or None


def _pair_by_prompt_mode(campaign, baseline_set, pathful_set):
    """Pair runs that share (task, model, turn, seed) but differ in
    prompt_mode. Returns list of (baseline_run, pathful_run) tuples."""
    by_key: dict[tuple, dict[str, list]] = {}
    for r in campaign.runs:
        mode = _prompt_mode(r)
        if mode is None:
            continue
        bucket = "baseline" if mode in baseline_set else (
            "pathful" if mode in pathful_set else None
        )
        if bucket is None:
            continue
        k = (r.task, r.model, r.turn, r.seed)
        by_key.setdefault(k, {"baseline": [], "pathful": []}).setdefault(
            bucket, []
        ).append(r)

    pairs: list[tuple] = []
    for k, slots in by_key.items():
        if slots.get("baseline") and slots.get("pathful"):
            pairs.append((slots["baseline"][0], slots["pathful"][0]))
    return pairs


class TestPathfulPromptDoesNotImproveHotRecall:
    """If pathful prompting worked, the HOT (literal-path) layer would
    gain recall when the prompt switches from default→pathful. We expect
    no improvement (or a regression) across the paired comparison."""

    def test_paired_hot_recall_no_improvement(self, campaign, min_seeds, report):
        pairs = _pair_by_prompt_mode(campaign, _BASELINE_VARIANTS,
                                     _PATHFUL_VARIANTS)
        if len(pairs) < min_seeds:
            pytest.skip(
                f"H12.hot_paired: only {len(pairs)} matched (task, model, "
                f"turn, seed) pairs differing on prompt_mode; need ≥ {min_seeds}. "
                f"Pathful variants live under outputs/multi_turn/e020_*pathful*."
            )

        deltas: list[dict] = []
        for b, p in pairs:
            if not b.byte_metrics_v1 or not p.byte_metrics_v1:
                continue
            try:
                # byte_metrics_v1 keys the HOT layer as "hot_first"/"hot_full"
                # (immediate-need vs full working-set GT), mirroring
                # tier_1_first below. The bare "hot" key never existed.
                b_hot = b.byte_metrics_v1["hot_first"]["byte_recall"]
                p_hot = p.byte_metrics_v1["hot_first"]["byte_recall"]
            except (KeyError, TypeError):
                continue
            deltas.append({
                "task": b.task, "model": b.model, "seed": b.seed,
                "hot_recall_baseline": b_hot,
                "hot_recall_pathful": p_hot,
                "delta": p_hot - b_hot,
            })

        if len(deltas) < min_seeds:
            pytest.skip(
                f"H12.hot_paired: {len(deltas)} pairs had byte_metrics_v1/hot "
                f"populated, need ≥ {min_seeds}"
            )

        median_delta = sorted(d["delta"] for d in deltas)[len(deltas) // 2]
        n_improved = sum(1 for d in deltas if d["delta"] > 0.05)
        report.record("h12_paired_hot_recall_deltas", {
            "n_pairs": len(deltas),
            "median_delta": round(median_delta, 4),
            "n_with_improvement_gt_5pp": n_improved,
            "frac_improved": round(n_improved / len(deltas), 3),
            "per_pair": deltas[:30],
        })

        # The hypothesis: median pathful-induced HOT-recall lift is ≤ 5pp.
        # A lift > 5pp would falsify "prompting doesn't work" and weaken
        # the auto-rules motivation.
        assert median_delta <= 0.05, (
            f"H12.hot_paired: median HOT-recall lift from pathful prompting "
            f"is {median_delta:+.3f} (> 0.05) across {len(deltas)} pairs — "
            f"prompting MAY actually work; reconsider the auto-rules-only "
            f"motivation."
        )


class TestPathfulPromptDoesNotImproveTier1Recall:
    """If pathful prompting only helped HOT but not tier-1 (the whole
    predictor), the headline story still holds. But if pathful prompting
    also lifts tier-1 recall, then auto-rules are over-engineered. This
    test pins the no-lift outcome at the tier-1 level too."""

    def test_paired_tier1_recall_no_improvement(self, campaign, min_seeds, report):
        pairs = _pair_by_prompt_mode(campaign, _BASELINE_VARIANTS,
                                     _PATHFUL_VARIANTS)
        if len(pairs) < min_seeds:
            pytest.skip(
                f"H12.tier1_paired: only {len(pairs)} matched pairs; "
                f"need ≥ {min_seeds}"
            )
        deltas: list[float] = []
        for b, p in pairs:
            if not b.byte_metrics_v1 or not p.byte_metrics_v1:
                continue
            try:
                b_t1 = b.byte_metrics_v1["tier_1_first"]["byte_recall"]
                p_t1 = p.byte_metrics_v1["tier_1_first"]["byte_recall"]
            except (KeyError, TypeError):
                continue
            deltas.append(p_t1 - b_t1)
        if len(deltas) < min_seeds:
            pytest.skip("H12.tier1_paired: insufficient paired byte_metrics_v1")

        median = sorted(deltas)[len(deltas) // 2]
        report.record("h12_paired_tier1_recall_deltas", {
            "n_pairs": len(deltas),
            "median_delta": round(median, 4),
            "p25": round(sorted(deltas)[len(deltas) // 4], 4),
            "p75": round(sorted(deltas)[(3 * len(deltas)) // 4], 4),
        })
        assert median <= 0.05, (
            f"H12.tier1_paired: median tier-1-recall lift from pathful "
            f"prompting is {median:+.3f} — auto-rules motivation needs "
            f"rethinking."
        )


class TestPathfulPromptCostNotJustified:
    """Pathful prompts cost extra tokens (the instruction itself + the
    longer reasoning needed to enumerate paths). If they don't improve
    recall (per the two tests above), the cost is wasted. This test
    documents the cost so the §6 paper text can quote it."""

    def test_pathful_prompt_token_overhead_recorded(
        self, campaign, min_seeds, report
    ):
        pairs = _pair_by_prompt_mode(campaign, _BASELINE_VARIANTS,
                                     _PATHFUL_VARIANTS)
        if len(pairs) < min_seeds:
            pytest.skip("H12.cost: insufficient pairs")
        rows: list[dict] = []
        for b, p in pairs:
            rows.append({
                "task": b.task, "model": b.model, "seed": b.seed,
                "baseline_thinking_chars": b.thinking_chars,
                "pathful_thinking_chars": p.thinking_chars,
                "char_overhead_pct": (
                    round(100 * (p.thinking_chars - b.thinking_chars)
                          / max(1, b.thinking_chars), 1)
                ),
            })
        if not rows:
            pytest.skip("H12.cost: thinking_chars missing on paired runs")
        med_overhead = sorted(r["char_overhead_pct"] for r in rows)[len(rows) // 2]
        report.record("h12_pathful_prompt_token_overhead", {
            "n_pairs": len(rows),
            "median_thinking_char_overhead_pct": med_overhead,
            "per_pair": rows[:30],
        })
        # No assertion on magnitude — this is a recorded measurement,
        # used in §6 narrative text. We do assert "not negative" so a
        # downward shift (pathful makes thinking SHORTER) gets caught.
        assert med_overhead > -50, (
            f"H12.cost: pathful runs are emitting much shorter thinking "
            f"than baseline (median {med_overhead:+.1f}%) — investigate "
            f"whether pathful runs are being truncated."
        )
