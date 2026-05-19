"""Auto-rule generator — synthesizes RuleSets from a workload's task spec
+ workspace prior without human curation.

Serves the L3 genericity defense in AGENTSTAGE.md §11.6: if auto-generated
rules hit within 10% of hand-tuned recall, the predictor architecture is
task-agnostic and the hand-curation is purely an engineering optimization.

**Scaffold only — Day 1 deliverable.** Real implementation lands on Day 1
backlog / Day 2 alongside T13 (leave-one-out). The intended approach:

  1. Read the workload's task spec (TaskConfig.task_inst) + workspace
     prior bucket KEYS (semantically meaningful names like
     "input_netcdfs", "core_runner", "sample_HG00096").
  2. Extract noun phrases via simple regex / POS tagging.
  3. Generate per-class regex variants for each bucket key:
     - exact bucket-name token
     - common synonym families (NetCDF ↔ .nc ↔ netCDF4)
     - per-sample / per-subject mention rules (for buckets that look
       like enumerations)
  4. Tier-tag the generated rules and emit a RuleSet.

Evaluation: re-score the PoC corpus against auto-generated rules and
compare tier-1/tier-3 byte recall against hand-tuned. Hits within 10%
of hand-tuned → L3 PASS.
"""

from __future__ import annotations

from agentstage.predictor.rules import RuleSet


class AutoRuleGenerator:
    """Auto-derives a RuleSet from workload metadata. NOT YET IMPLEMENTED.

    Constructor takes the same inputs a human author would have when
    writing rules by hand: the task instruction text, the workspace
    prior bucket structure, and optionally a few example thinking
    excerpts (transcripts of how models talk about this workload).
    """

    def __init__(
        self,
        workload_id: str,
        task_instruction: str,
        workspace_prior_keys: tuple[str, ...],
        thinking_excerpts: tuple[str, ...] = (),
    ) -> None:
        self.workload_id = workload_id
        self.task_instruction = task_instruction
        self.workspace_prior_keys = workspace_prior_keys
        self.thinking_excerpts = thinking_excerpts

    def generate(self) -> RuleSet:
        """Synthesize a RuleSet for the workload. NOT YET IMPLEMENTED.

        Returns an empty RuleSet so callers can wire up the integration
        before the algorithm lands. The H6/H7 paper_evals tests that
        consume auto_rules will see "0 rules → 0 byte recall" until
        real generation lands.
        """
        return RuleSet(workload=self.workload_id, rules=())
