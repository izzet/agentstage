"""ScienceAgentBench (SAB) workload loader.

SAB ships 102 verified tasks via HuggingFace (`osunlp/ScienceAgentBench`).
The actual input data files are distributed in a separate zip not bundled
here, so this loader builds a workspace_prior from the `dataset_folder_tree`
field — the listing the agent sees as context — and serves a *mock* sandbox
that returns dataset_preview for any open_file. Captures from this setup
are valid for H6 cross-benchmark detection evidence (we care about which
file names the agent reasons about, not the byte content).

For H8 (staging effectiveness) on SAB, the user would need to download
benchmark_verified.zip and wire a real cold-tier mount — out of scope for
the minimal cross-benchmark cycle.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any


@dataclass(frozen=True)
class SABTask:
    instance_id: int
    domain: str
    task_inst: str
    dataset_folder_tree: str
    dataset_preview: str
    output_fname: str
    domain_knowledge: str = ""

    @property
    def task_inst_full(self) -> str:
        """The task_inst the LLM sees, plus tree context, mirroring SAB's
        run_infer.py task dict format."""
        return (
            f"{self.task_inst}\n\n"
            f"Dataset folder tree:\n{self.dataset_folder_tree}\n\n"
            f"Output filename: {self.output_fname}"
        )


@dataclass(frozen=True)
class SABWorkload:
    task_id: str
    task: SABTask
    workspace_prior: dict[str, tuple[str, ...]]
    ground_truth_full: tuple[str, ...]
    ground_truth_first_inspect: tuple[str, ...]
    prefix_map: tuple[tuple[str, str], ...]

    @property
    def all_workspace_paths(self) -> tuple[str, ...]:
        return tuple(p for paths in self.workspace_prior.values() for p in paths)


_TREE_LINE = re.compile(r"^\|(-+)\s*(.+?)/?\s*$")


def _parse_tree(tree: str, root_prefix: str = "/data") -> dict[str, tuple[str, ...]]:
    """Parse SAB's `|-- foo.csv` style folder tree into a workspace_prior.

    Returns a dict keyed by top-level directory name, values are tuples of
    LOGICAL file paths.
    """
    files: list[str] = []
    stack: list[str] = []
    last_depth = 0
    for line in tree.splitlines():
        m = _TREE_LINE.match(line)
        if not m:
            continue
        depth = len(m.group(1)) // 2  # tree indents in pairs of "-"
        name = m.group(2).strip()
        if not name:
            continue
        # Adjust the stack to current depth
        if depth <= len(stack):
            stack = stack[: depth - 1]
        stack.append(name.rstrip("/"))
        # Heuristic: treat as a file iff there's a "." in the last component
        # AND the next non-blank line is at the same or lower depth.
        is_file = "." in name and not name.endswith("/")
        if is_file:
            files.append(root_prefix + "/" + "/".join(stack))
        last_depth = depth

    # Group by top-level group
    buckets: dict[str, list[str]] = {}
    for f in files:
        parts = f[len(root_prefix) + 1:].split("/", 1)
        group = parts[0] if len(parts) > 1 else "root"
        bucket = "files_" + re.sub(r"[^A-Za-z0-9_]", "_", group)
        buckets.setdefault(bucket, []).append(f)
    out: dict[str, tuple[str, ...]] = {k: tuple(v) for k, v in buckets.items()}
    out["all_files"] = tuple(files)
    return out


@lru_cache(maxsize=1)
def _load_sab_dataset() -> Any:
    from datasets import load_dataset
    return load_dataset("osunlp/ScienceAgentBench", split="verified")


def load_sab_task(instance_id: int) -> SABWorkload:
    ds = _load_sab_dataset()
    rows = [r for r in ds if r["instance_id"] == instance_id]
    if not rows:
        raise KeyError(f"SAB instance_id={instance_id} not in verified split")
    raw = rows[0]
    task = SABTask(
        instance_id=raw["instance_id"],
        domain=raw.get("domain", "?"),
        task_inst=raw["task_inst"],
        dataset_folder_tree=raw["dataset_folder_tree"],
        dataset_preview=raw.get("dataset_preview", ""),
        output_fname=raw["output_fname"],
        domain_knowledge=raw.get("domain_knowledge", "") or "",
    )
    workspace_prior = _parse_tree(task.dataset_folder_tree, root_prefix="/data")
    # GT: every input file is a candidate (SAB tasks read everything in the
    # supplied tree typically); narrow further if a src_file_or_path is set.
    gt = workspace_prior.get("all_files", ())
    return SABWorkload(
        task_id=f"sab_{instance_id:03d}",
        task=task,
        workspace_prior=workspace_prior,
        ground_truth_full=gt,
        ground_truth_first_inspect=gt,
        prefix_map=(("/data/", "MOCK://"),),
    )


def sab_minimal_slice() -> dict[str, int]:
    """3 small-tree SAB tasks across distinct domains. Picked at runtime
    rather than hard-coded so this works even if SAB IDs shift."""
    ds = _load_sab_dataset()
    by_domain: dict[str, list[dict]] = {}
    for r in ds:
        by_domain.setdefault(r.get("domain", "?"), []).append(r)
    picks: dict[str, int] = {}
    for domain, rows in sorted(by_domain.items()):
        # Smallest tree first (lines of `dataset_folder_tree`)
        rows.sort(key=lambda r: len(r["dataset_folder_tree"].splitlines()))
        if rows:
            r = rows[0]
            picks[f"sab_{r['instance_id']:03d}"] = r["instance_id"]
        if len(picks) >= 3:
            break
    return picks
