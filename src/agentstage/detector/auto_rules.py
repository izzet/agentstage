"""Auto-rule generator — synthesizes RuleSets from a workload's task spec
+ workspace prior without human curation.

Serves the L3 genericity defense in AGENTSTAGE.md §11.6: if auto-generated
rules hit within 10% of hand-tuned recall, the detector architecture is
task-agnostic and the hand-curation is purely an engineering optimization.

Algorithm (Day 1 deliverable, now real):

  1. For each workspace-prior bucket key, mechanically derive a regex
     pattern from the key's structure:
       - `band_08`        → pattern matches "band 08", "band 8", "band_08"
       - `sample_HG00096` → pattern matches "HG00096"
       - `subject_sub-Cori` → pattern matches "sub-Cori" or bare "Cori"
       - `day_122`        → pattern matches "day 122" or "/122/"
     Emit one Rule per bucket, target_keys=(key,), origin="auto".

  2. Emit general rules (workload-agnostic): first_inspect, all_signal,
     report_out. The file-format token list is extracted from the
     task_instruction (e.g. spot "NetCDF" / "BAM" / "NWB" / "VCF" /
     "CSV" / "Parquet" mentions).

  3. Skip buckets where a general rule already covers the function:
       - `output_*`  → matched by report_out / generic-output rules
       - `all_*`     → matched by all_X_signal
       - `first_*`   → matched by first_inspect
       - `reference` → matched by reference_signal (auto-emitted if seen)

The output is a RuleSet that doesn't require domain knowledge — just the
task spec and the bucket structure. The "C08 ↔ band 08" synonym mapping
that hand-tuned rules have is the main gap; we close some of that via
task-instruction scanning (step 2 above).

Evaluation: re-score the PoC corpus against auto-generated rules and
compare tier-1/tier-3 byte recall against hand-tuned. Hits within 10%
of hand-tuned → L3 PASS.
"""

from __future__ import annotations

import re
from collections import OrderedDict

from agentstage.detector.rules import Rule, RuleSet

# Class-prefix → optional "class label" for regex construction
# Order matters: longer/more-specific prefixes first.
_CLASS_PREFIXES: tuple[tuple[str, str], ...] = (
    ("sample_", "sample"),
    ("subject_", "subject"),
    ("band_", "band"),
    ("channel_", "channel"),
    ("day_", "day"),
    ("month_", "month"),
    ("year_", "year"),
    ("session_", "session"),
    ("trial_", "trial"),
)

# Known scientific file-format tokens we look for in task_instruction
# to seed the general rules' "noun" alternation.
_FORMAT_TOKENS: tuple[str, ...] = (
    "NetCDF", "netCDF", "netcdf", "nc",
    "BAM", "bam", "BAI", "bai", "BAS", "bas",
    "NWB", "nwb",
    "VCF", "vcf",
    "CSV", "csv",
    "Parquet", "parquet",
    "HDF5", "hdf5", "h5",
    "Zarr", "zarr",
    "FITS", "fits",
    "GeoTIFF", "tiff", "tif",
    "JSON", "json",
    "NPY", "npy",
)


def _split_class_prefix(key: str) -> tuple[str, str]:
    """Return (class_label, instance) for a bucket key.
    If no recognized prefix, returns ('', key)."""
    for prefix, label in _CLASS_PREFIXES:
        if key.startswith(prefix):
            return label, key[len(prefix):]
    return "", key


def _instance_regex(class_label: str, instance: str) -> str:
    """Build a regex matching the instance, with optional class-label-aware
    variants. Examples:
       ("band", "08")  →  \\b(?:band[- ]?0?8\\b|\\b08\\b)
       ("sample", "HG00096")  →  \\bHG00096\\b
       ("subject", "sub-Cori")  →  \\b(?:sub-Cori\\b|\\bCori\\b)
       ("day", "122")  →  \\b(?:day[- ]?122\\b|/122/)
    """
    escaped = re.escape(instance)
    parts: list[str] = []
    # 1. class_label + instance (e.g. "band 08", "band-08", "band08")
    if class_label:
        # If instance is a number, also allow stripped-leading-zero variant
        if instance.isdigit() and instance.startswith("0") and len(instance) > 1:
            stripped = instance.lstrip("0") or "0"
            parts.append(rf"\b{class_label}[- _]?0?{stripped}\b")
        else:
            parts.append(rf"\b{class_label}[- _]?{escaped}\b")
    # 2. bare instance with word boundary (catches identifiers like HG00096,
    #    sub-Cori, mentioned without the class label)
    # For multi-token names with hyphens (sub-Cori), also emit a "suffix only"
    # variant (just "Cori") because LLMs often drop the prefix.
    if "-" in instance and not instance[0].isdigit():
        suffix = instance.split("-", 1)[1]
        if suffix and len(suffix) >= 3:  # avoid trivial suffixes
            parts.append(rf"\b(?:{escaped}|{re.escape(suffix)})\b")
        else:
            parts.append(rf"\b{escaped}\b")
    else:
        parts.append(rf"\b{escaped}\b")
    # 3. path-component form (for time/date components like /122/ in paths)
    if class_label in ("day", "month", "year", "session") and instance.isdigit():
        parts.append(rf"/{escaped}/")
    return "|".join(parts)


def _format_alternation(task_instruction: str) -> str:
    """Build a regex alternation of file-format tokens that appear in
    the task instruction. Used to seed first_inspect / all_signal rules
    with workload-relevant nouns."""
    seen: list[str] = []
    for tok in _FORMAT_TOKENS:
        if tok in task_instruction and tok.lower() not in [s.lower() for s in seen]:
            seen.append(tok)
    if not seen:
        return "file"
    # Always include the generic "file" too
    seen.append("file")
    return "|".join(re.escape(t) for t in seen)


def _extract_output_filenames(task_instruction: str) -> list[str]:
    """Find any explicit output filenames mentioned in the task spec.
    E.g. 'result/report.md', 'goes_cmi_timeseries.csv'."""
    # Match path-like tokens with a recognized extension
    pat = re.compile(
        r"[A-Za-z0-9_\-./]+\.(?:csv|md|png|parquet|json|txt|tsv|h5|nc)\b",
        flags=re.IGNORECASE,
    )
    return list(OrderedDict.fromkeys(pat.findall(task_instruction)))


class AutoRuleGenerator:
    """Auto-derives a RuleSet from workload metadata.

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
        self.workspace_prior_keys = tuple(workspace_prior_keys)
        self.thinking_excerpts = thinking_excerpts

    def generate(self) -> RuleSet:
        """Synthesize a RuleSet for the workload.

        Strategy:
          A. Per-bucket rules (one per non-aggregate key)
          B. General rules (first_inspect, all_signal, report_out)
          C. Output-file rules (one per detected output filename)
        """
        rules: list[Rule] = []
        format_alt = _format_alternation(self.task_instruction)
        output_files = _extract_output_filenames(self.task_instruction)

        # ─── B. General rules (emit first so they precede per-instance) ────
        rules.append(Rule(
            name="first_inspect",
            pattern=rf"(?:first|one|single|sample|representative).{{0,30}}"
                    rf"(?:{format_alt})|"
                    rf"\bstart with\b|\bbegin with\b|"
                    rf"inspect.{{0,20}}(?:one|a single|first)",
            target_keys=("first_file",) if "first_file" in self.workspace_prior_keys
                        else ("all_files",) if "all_files" in self.workspace_prior_keys
                        else self._first_aggregate_key(),
            origin="auto",
        ))

        # Pick a plural aggregate to point at
        all_key = self._find_aggregate_key()
        if all_key is not None:
            rules.append(Rule(
                name="all_signal",
                pattern=rf"(?:all|each|every).{{0,30}}(?:{format_alt})|"
                        rf"iterate.{{0,40}}(?:{format_alt})|"
                        rf"(?:the )?(?:entire|whole|complete) (?:dataset|workspace|corpus)",
                target_keys=(all_key,),
                origin="auto",
            ))

        # ─── A. Per-instance rules ────────────────────────────────────────
        seen_targets: set[str] = set()
        for key in self.workspace_prior_keys:
            if key.startswith(("output_", "all_", "first_")):
                continue  # covered by general rules
            if key in seen_targets:
                continue
            seen_targets.add(key)
            class_label, instance = _split_class_prefix(key)
            if not instance:
                continue
            pattern = _instance_regex(class_label, instance)
            rules.append(Rule(
                name=key,
                pattern=pattern,
                target_keys=(key,),
                origin="auto",
            ))

        # ─── C. Output-file rules ─────────────────────────────────────────
        # For each output_* key, try to find a regex matching the output file
        for key in self.workspace_prior_keys:
            if not key.startswith("output_"):
                continue
            # Match the output bucket key to a filename mentioned in task_inst
            # Heuristic: filename stem contains the key's suffix (e.g.
            # output_csv → "*.csv", output_report → "report.md").
            suffix = key[len("output_"):]
            matched: list[str] = []
            for fname in output_files:
                if suffix in fname.lower() or fname.lower().endswith("." + suffix):
                    matched.append(fname)
            if matched:
                pattern = "|".join(re.escape(f) for f in matched)
            else:
                # Fallback: match the key's suffix as a file extension
                pattern = rf"\.{re.escape(suffix)}\b"
            rules.append(Rule(
                name=f"{key}_signal",
                pattern=pattern,
                target_keys=(key,),
                origin="auto",
            ))

        # Always emit a generic report rule (covers report.md across workloads)
        rules.append(Rule(
            name="report_out",
            pattern=r"report\.md|report markdown",
            target_keys=tuple(k for k in self.workspace_prior_keys
                              if k.startswith("output_report"))
                        or ("output_report",) if False else (),
            origin="auto",
        ))

        return RuleSet(workload=self.workload_id, rules=tuple(rules))

    # ────────────────────────────────────────────────────────────────────
    # Helpers
    # ────────────────────────────────────────────────────────────────────

    def _find_aggregate_key(self) -> str | None:
        """Pick the most-likely 'all the data' bucket name."""
        for candidate in ("all_files", "all_samples", "all_subjects",
                          "all_bands", "all_sessions"):
            if candidate in self.workspace_prior_keys:
                return candidate
        # Fallback: first key starting with "all_"
        for k in self.workspace_prior_keys:
            if k.startswith("all_"):
                return k
        return None

    def _first_aggregate_key(self) -> tuple[str, ...]:
        ak = self._find_aggregate_key()
        return (ak,) if ak else ()
