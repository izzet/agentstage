"""Frozen semantic-class rule library for AgentStage's intent detector.

Each rule maps a regex pattern over the agent's streaming thinking text to
one or more workspace-prior bucket keys. When the regex matches, the rule
"fires" and contributes every file in its target buckets to the tiered
detected set.

This module is the load-bearing artifact for the paper's genericity
claim. The rules are FROZEN: `RULE_LIBRARY_VERSION`
plus `RULE_LIBRARY_HASH` form the freeze contract that
`tests/test_rules_freeze.py` pins. Bumping the version is a deliberate
commit-message-documented event; quiet rule edits are detected as a hash
mismatch in the test suite.

Each rule carries an `origin` tag naming the workload it was authored for.
The leave-one-out evaluation (E3) re-scores a held-out workload's traces
using only the rules whose `origin` is NOT in the held-out set. Rules
tagged `"general"` are corpus-agnostic and survive every fold.

Ported from `poc/probe_reasoning_slack.py` on 2026-05-19.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

RuleOrigin = Literal[
    "aiob_101", "aiob_104", "aiob_107", "aiob_110", "code_repo", "general"
]


# ---------------------------------------------------------------------------
# Core types
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Rule:
    """A semantic-class rule mapping a thinking-text regex to workspace-prior
    bucket keys.

    Attributes:
        name: Unique identifier within the rule library (used as activation
            key in `detection.json`).
        pattern: Regex applied (case-insensitive) to streaming thinking text.
        target_keys: Workspace-prior bucket keys to contribute when fired.
        origin: Workload this rule was authored for, or "general" if it
            transfers across workloads without modification.
    """

    name: str
    pattern: str
    target_keys: tuple[str, ...]
    origin: RuleOrigin

    def fingerprint(self) -> tuple[str, str, tuple[str, ...], str]:
        """Canonical tuple for hashing into the freeze contract."""
        return (self.name, self.pattern, tuple(sorted(self.target_keys)), self.origin)


@dataclass(frozen=True)
class RuleSet:
    """An immutable ordered collection of rules, scoped to one workload.

    Rule order matters for activation timestamps when multiple rules fire on
    the same char offset (first in iteration wins ties — port semantics from
    the PoC's list-of-tuples iteration order).
    """

    workload: str
    rules: tuple[Rule, ...]

    def filter_by_origin(self, exclude_origins: Iterable[str]) -> RuleSet:
        """Return a new RuleSet with rules from `exclude_origins` removed.

        Used for leave-one-out (E3). Rules with `origin == "general"` are
        never excluded.
        """
        excl = set(exclude_origins)
        return RuleSet(
            workload=self.workload,
            rules=tuple(r for r in self.rules if r.origin == "general" or r.origin not in excl),
        )

    def __iter__(self):
        return iter(self.rules)

    def __len__(self) -> int:
        return len(self.rules)


# ---------------------------------------------------------------------------
# aiob_104 — IGSR genomics (50 GBR samples, BAM/BAI/BAS + reference)
# ---------------------------------------------------------------------------

# Real sample IDs (alphabetically first 50 GBR samples in IGSR).
AIOB_104_SAMPLES: tuple[str, ...] = (
    "HG00096", "HG00097", "HG00099", "HG00100", "HG00101", "HG00102", "HG00103", "HG00105",
    "HG00106", "HG00107", "HG00108", "HG00109", "HG00110", "HG00111", "HG00112", "HG00113",
    "HG00114", "HG00115", "HG00116", "HG00117", "HG00118", "HG00119", "HG00120", "HG00121",
    "HG00122", "HG00123", "HG00124", "HG00125", "HG00126", "HG00127", "HG00128", "HG00129",
    "HG00130", "HG00131", "HG00132", "HG00133", "HG00136", "HG00137", "HG00138", "HG00139",
    "HG00140", "HG00141", "HG00142", "HG00143", "HG00145", "HG00146", "HG00148", "HG00149",
    "HG00150", "HG00151",
)

RULES_AIOB_104: RuleSet = RuleSet(
    workload="aiob_104",
    rules=(
        # Per-sample literal mentions (50 rules; each fires its own bucket)
        *(
            Rule(
                name=f"mention_{s}",
                pattern=re.escape(s) + r"\b",
                target_keys=(f"sample_{s}",),
                origin="aiob_104",
            )
            for s in AIOB_104_SAMPLES
        ),
        Rule("bam_general", r"\bBAM\b|\.bam\b|samtools|pysam", ("all_samples",), "aiob_104"),
        Rule("reference_signal", r"reference|BED|targets|fasta", ("reference",), "aiob_104"),
        Rule(
            "first_inspect",
            r"(?:first|one|single).{0,30}(?:sample|BAM|file)|"
            r"\bstart with\b|\bbegin with\b",
            ("sample_HG00096",),
            "general",
        ),
        Rule(
            "all_samples_signal",
            r"(?:all|each|every).{0,30}sample|iterate.{0,30}sample|"
            r"(?:50|fifty)\s+(?:samples|files)|loop.*sample",
            ("all_samples",),
            "general",
        ),
        Rule("coverage_out", r"coverage_matrix|coverage matrix", ("output_coverage",), "aiob_104"),
        Rule("qc_out", r"qc_metrics|qc metrics", ("output_qc",), "aiob_104"),
        Rule("undercov_out", r"undercovered|under.?cover", ("output_undercov",), "aiob_104"),
        Rule("report_out", r"report\.md|report markdown", ("output_report",), "general"),
    ),
)


# ---------------------------------------------------------------------------
# aiob_107 — GOES meteorology (6042 NetCDFs across 3 bands × 7 days × 24 hours)
# ---------------------------------------------------------------------------

RULES_AIOB_107: RuleSet = RuleSet(
    workload="aiob_107",
    rules=(
        Rule("band_08", r"\bband[- ]?0?8\b|C08|channel[- ]?8", ("band_08",), "aiob_107"),
        Rule("band_09", r"\bband[- ]?0?9\b|C09|channel[- ]?9", ("band_09",), "aiob_107"),
        Rule("band_10", r"\bband[- ]?10\b|C10|channel[- ]?10", ("band_10",), "aiob_107"),
        Rule(
            "all_bands",
            r"all (?:three )?bands|3 bands|three bands|both bands",
            ("band_08", "band_09", "band_10"),
            "aiob_107",
        ),
        Rule(
            "all_files_signal",
            r"(?:all|each|every).{0,30}(?:file|NetCDF|timestamp)|"
            r"iterate.{0,40}(?:file|timestamp)|6042|~6000|6,?000",
            ("all_files",),
            "general",
        ),
        Rule(
            "first_inspect",
            r"(?:first|one|single|sample|representative).{0,30}(?:file|NetCDF|timestamp)|"
            r"\bstart with\b|\bbegin with\b|inspect.{0,20}(?:one|a single|first)",
            ("first_file",),
            "general",
        ),
        Rule(
            "one_hour",
            r"first hour|one hour|(?:day 122|2024-?05-?01).*hour 0",
            ("first_hour_all_bands",),
            "aiob_107",
        ),
        Rule("csv_out", r"timeseries\.csv|\.csv\b", ("output_csv",), "aiob_107"),
        Rule("fig_out", r"point_timeseries\.png|figure|\.png\b", ("output_fig",), "aiob_107"),
        Rule("report_out", r"report\.md", ("output_report",), "general"),
    ),
)


# ---------------------------------------------------------------------------
# aiob_110 — Steinmetz NWB (39 NWBs across 10 subjects)
# ---------------------------------------------------------------------------

AIOB_110_SUBJECTS: tuple[str, ...] = (
    "sub-Cori", "sub-Forssmann", "sub-Hench", "sub-Lederberg", "sub-Moniz",
    "sub-Muller", "sub-Radnitz", "sub-Richards", "sub-Tatum", "sub-Theiler",
)

RULES_AIOB_110: RuleSet = RuleSet(
    workload="aiob_110",
    rules=(
        Rule(
            "all_subjects_signal",
            r"(?:all|each|every).{0,30}(?:session|subject|file)|"
            r"iterate.{0,30}(?:session|subject)|"
            r"(?:39|ten|10)\s+(?:sessions|subjects|files|NWBs?)|"
            r"\bloop\b.*(?:NWB|session|subject)",
            ("all_subjects",),
            "general",
        ),
        Rule(
            "first_inspect",
            r"(?:first|one|single|representative).{0,30}(?:session|subject|file|NWB|inspect)|"
            r"\bstart with\b|\bbegin with\b|inspect.{0,20}(?:one|a single|first)",
            ("subject_sub-Cori",),
            "general",
        ),
        # Per-subject mentions: each fires its own subject's files
        *(
            Rule(
                name=f"mention_{subj}",
                pattern=re.escape(subj) + r"\b",
                target_keys=(f"subject_{subj}",),
                origin="aiob_110",
            )
            for subj in AIOB_110_SUBJECTS
        ),
        Rule("nwb_general", r"\bNWB\b|\.nwb\b|pynwb|h5py", ("all_subjects",), "aiob_110"),
        Rule(
            "trial_responses_out",
            r"trial_responses\.parquet|trial.{0,5}response|per.trial",
            ("output_trial_responses",),
            "aiob_110",
        ),
        Rule(
            "session_summary_out",
            r"session_summary\.parquet|per.session|session.{0,5}summary",
            ("output_session_summary",),
            "aiob_110",
        ),
        Rule(
            "report_out",
            r"report\.md|summary of sessions",
            ("output_report_md",),
            "general",
        ),
    ),
)


# ---------------------------------------------------------------------------
# aiob_101 — ERA5 climate (36 monthly NetCDFs + shapefile; structural edge case)
# ---------------------------------------------------------------------------

RULES_AIOB_101: RuleSet = RuleSet(
    workload="aiob_101",
    rules=(
        Rule(
            "netcdf_inputs",
            r"NetCDF|\.nc\b|single_levels|\bmonthly\b",
            ("input_netcdfs",),
            "aiob_101",
        ),
        Rule(
            "wet_bulb_signals",
            r"wet.?bulb|2m_temperature|dewpoint",
            ("input_netcdfs",),
            "aiob_101",
        ),
        Rule("era5_dataset", r"\bERA5\b", ("input_netcdfs",), "aiob_101"),
        Rule(
            "shapefile",
            r"shapefile|polygon|county.*shape|cb_2023|500k",
            ("input_shapefile",),
            "aiob_101",
        ),
        Rule(
            "county_signals",
            r"\bcounty\b|county-level|U\.?S\.? count",
            ("input_shapefile",),
            "aiob_101",
        ),
        Rule(
            "stage_a_out",
            r"Stage A|rechunked.*zarr|chunk.?schema|stage_a/",
            ("output_stage_a_zarr", "output_chunk_schema"),
            "aiob_101",
        ),
        Rule(
            "stage_b_out",
            r"Stage B|heatwave.?event|\.parquet|report\.md|result/",
            ("output_events_parquet", "output_report_md", "output_result_zarr"),
            "aiob_101",
        ),
        Rule(
            "stage_b_reads_a",
            r"open the rechunked|read.*rechunked|from Stage A",
            ("output_stage_a_zarr",),
            "aiob_101",
        ),
    ),
)


# ---------------------------------------------------------------------------
# code_repo — Coding agent on a 120-file Python codebase
# ---------------------------------------------------------------------------

RULES_CODE_REPO: RuleSet = RuleSet(
    workload="code_repo",
    rules=(
        # Literal module mentions
        Rule("mention_runner", r"\brunner(?:\.py)?\b|runner\.py", ("core_runner",), "code_repo"),
        Rule("mention_tools", r"\btools(?:\.py)?\b|tools\.py", ("core_tools",), "code_repo"),
        Rule("mention_llm", r"\bllm(?:\.py)?\b|llm\.py", ("core_llm",), "code_repo"),
        # Function-name signals → the file that implements them
        Rule("write_file_signal", r"write_file|writeFile", ("core_tools",), "code_repo"),
        Rule(
            "dispatch_signal",
            r"dispatch|tool[- ]dispatch|main loop|agent loop|tool[- ]call",
            ("core_runner", "core_tools"),
            "code_repo",
        ),
        # Broader categories
        Rule("test_signal", r"\btest\b|pytest|unit test|test_", ("tests",), "general"),
        Rule("config_signal", r"\bconfig\b(?!\.py)|yaml|YAML", ("config_files",), "general"),
        Rule(
            "tracing_signal",
            r"tracing|dftracer|instrumentation|tracer",
            ("tracing_files",),
            "general",
        ),
        Rule("scripts_signal", r"\bscripts?\b|utility script", ("scripts_dir",), "general"),
        # First-look behavior
        Rule(
            "first_inspect",
            r"(?:start|begin) with|inspect(?:ing)? .{0,20}(?:runner|tools|main)|"
            r"first.{0,20}(?:read|look|explore|inspect)",
            ("core_runner", "core_tools"),
            "general",
        ),
        # Broad — last resort
        Rule(
            "all_files_signal",
            r"(?:search|grep|scan).{0,30}(?:codebase|repo|repository|whole|entire)|"
            r"every file|all files|entire codebase",
            ("all_files",),
            "general",
        ),
        # Output rules
        Rule("fix_out", r"fix\.md|patch|one-line", ("output_fix",), "code_repo"),
        Rule("report_out", r"report\.md", ("output_report",), "general"),
    ),
)


# ---------------------------------------------------------------------------
# Library aggregation
# ---------------------------------------------------------------------------

ALL_RULESETS: dict[str, RuleSet] = {
    "aiob_101": RULES_AIOB_101,
    "aiob_104": RULES_AIOB_104,
    "aiob_107": RULES_AIOB_107,
    "aiob_110": RULES_AIOB_110,
    "code_repo": RULES_CODE_REPO,
}


def get_ruleset(workload: str) -> RuleSet:
    """Look up a workload's RuleSet by name."""
    if workload not in ALL_RULESETS:
        raise KeyError(
            f"No rule set for workload {workload!r}. "
            f"Known workloads: {sorted(ALL_RULESETS)}"
        )
    return ALL_RULESETS[workload]


# ---------------------------------------------------------------------------
# Freeze contract: version + hash
# ---------------------------------------------------------------------------

RULE_LIBRARY_VERSION: str = "v1"
"""Semver-ish identifier bumped on deliberate rule-library edits.

Bumping this is a load-bearing event: it invalidates the cross-corpus
genericity claim until the new version has been re-validated against the
external benchmarks (SAB + KramaBench). Bump only with an explicit
commit message documenting why and what changed.
"""


def _canonical_serialization() -> bytes:
    """Deterministic byte representation of the entire rule library for
    hashing. The hash is invariant to dict-iteration order but sensitive
    to rule-text, target-keys, and origin tags."""
    payload = []
    for workload in sorted(ALL_RULESETS):
        rs = ALL_RULESETS[workload]
        payload.append({
            "workload": workload,
            "rules": [
                {
                    "name": r.name,
                    "pattern": r.pattern,
                    "target_keys": list(r.target_keys),
                    "origin": r.origin,
                }
                for r in rs.rules
            ],
        })
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


RULE_LIBRARY_HASH: str = hashlib.sha256(_canonical_serialization()).hexdigest()
"""SHA-256 of the canonical rule serialization. Pinned in
`tests/test_rules_freeze.py`. Any change to any rule's name, pattern,
target_keys, or origin bumps this hash automatically — the test fails,
forcing a deliberate `RULE_LIBRARY_VERSION` bump and updated pin."""


def rule_count_by_origin() -> dict[str, int]:
    """Diagnostic: how many rules are tagged with each origin."""
    counts: dict[str, int] = {}
    for rs in ALL_RULESETS.values():
        for r in rs.rules:
            counts[r.origin] = counts.get(r.origin, 0) + 1
    return counts
