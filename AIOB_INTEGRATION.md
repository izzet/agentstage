# AgentIOBench Integration

How AgentStage observes and augments AgentIOBench (AIOB) runs without
forking it. Companion to `CAMPAIGN.md` (which says *what* to run) and
`STAGER_DESIGN.md` (which covers the staging-daemon side).

Source-of-truth code lives on the `feat/agentstage-integration` branch
of `git@github.com:grc-iit/agentiobench.git`. Our submodule pin
(`external/benchmarks/agentiobench/`) tracks that branch. Four commits
constitute the integration as of 2026-05-19:

- `f4d5723` Add public API surface + run_agentic turn-hook integration points
- `4b78210` Make agentiobench public-API re-exports lazy
- `813d477` Verify run_agentic hooks fire correctly under mocked LLM + tools
- `9f07f9b` gitignore: .venv-test/

The contract this document describes is **verified by AIOB's own
pytest suite** — see §3.5 below for the five behavioral tests that
exercise the hooks under mocked LLM + tools, and §6 for the property
table mapping each guarantee to the test that pins it.

---

## 1. The problem

AgentStage runs an agent loop. AIOB also runs an agent loop. They are
**the same loop** for the AIOB workloads (aiob_104, 107, 110, plus
code_repo) — there is no value in writing a second agentic harness when
AIOB's `run_agentic` already drives `chat_completion`, dispatches
tools, captures trajectories, writes `verdict.json`, and wraps the
whole thing in DFTracer + judge spans.

What AgentStage needs to add to that loop:

1. **Per-turn observability.** Before the LLM call: snapshot the
   prompt that's about to be sent. After the LLM call: read the
   streaming response's thinking content, run the predictor live,
   dispatch prefetch to the stager.
2. **Per-run lifecycle hooks.** Before the run: cold-cache eviction +
   temperature snapshot. After the run: post-run temperature, write
   AgentStage-side artifacts (`prediction.json`, `byte_metrics.json`,
   `staging_report.json`, `cost.json`).
3. **A public import surface.** Use AIOB's `TaskConfig`,
   `evict_dataset`, `measure_temperature` without reaching into private
   modules — so a future AIOB refactor that renames an internal module
   doesn't silently break AgentStage.

What AgentStage explicitly does **not** need:

- A separate copy of AIOB's `chat_completion` / tool dispatch /
  trajectory bookkeeping. Those stay in AIOB; AgentStage observes them.
- The ability to change the agent's behavior (replace prompts, inject
  tool calls). Read-only observation is sufficient for every paper
  claim except the soft-stop, which is an out-of-band cooperative
  signal not a hook (see `CAMPAIGN.md` §15).

---

## 2. Design choices and alternatives

I considered four patterns. Here's what I picked and why.

### 2.1 What I picked: keyword-only callback hooks on `run_agentic`

```python
def run_agentic(
    cfg: BenchmarkConfig,
    *,
    pre_turn_hook:  Optional[PreTurnHook]  = None,   # (turn_idx, messages)
    post_turn_hook: Optional[PostTurnHook] = None,   # (turn_idx, response, trajectory)
) -> Dict[str, Any]: ...
```

Two callbacks, invoked at well-defined points in the agentic loop.
Backwards-compatible by construction (new params default to `None`,
all-existing-callers unaffected). Hooks are wrapped in a
`_safe_call_hook` helper that catches exceptions so a broken AgentStage
hook can't corrupt benchmark results.

### 2.2 Alternative A — monkey-patch `chat_completion`

```python
# AgentStage side, before calling AIOB:
import agentiobench.llm
real_chat_completion = agentiobench.llm.chat_completion
def wrapped(messages, **kw):
    response = real_chat_completion(messages, **kw)
    predictor.feed(response.streaming_events)
    return response
agentiobench.llm.chat_completion = wrapped
```

**Why I rejected it:** monkey-patching is invisible to the AIOB
codebase. A reviewer reading AIOB has no idea AgentStage exists, and
a future AIOB refactor that renames or relocates `chat_completion`
silently breaks AgentStage with no compile-time signal. Also,
monkey-patching only sees the LLM call — not the turn boundary, which
is the right unit of cache-temperature logging.

### 2.3 Alternative B — subclass `BenchmarkConfig` and dispatch via methods

```python
class AgentStageConfig(BenchmarkConfig):
    def on_pre_turn(self, turn, messages): ...
    def on_post_turn(self, turn, response, trajectory): ...
```

Then AIOB would check `if hasattr(cfg, "on_pre_turn"): cfg.on_pre_turn(...)`.

**Why I rejected it:** Pydantic `BaseModel` subclasses don't carry
methods naturally, and putting behavior on a config object conflates
"description of the run" with "callbacks during the run." The type
hierarchy gets weird fast. Also, you'd be passing AgentStage-specific
methods through AIOB's config validation pipeline.

### 2.4 Alternative C — global hook registry

```python
# agentiobench/runner.py
_HOOKS = {"pre_turn": [], "post_turn": []}

def register_pre_turn(fn): _HOOKS["pre_turn"].append(fn)
def register_post_turn(fn): _HOOKS["post_turn"].append(fn)
```

AgentStage would `register_pre_turn(predictor.observe)` at import time.

**Why I rejected it:** global mutable state is a footgun. Two
`run_agentic` invocations in the same process would share hooks,
which is wrong: AgentStage's hooks for run-A shouldn't fire on run-B
that some other code path is running concurrently. Per-call params
make the lifetime explicit.

### 2.5 Alternative D — replace `run_agentic` entirely with our own

Write `agentstage.runners.aiob_runner.run_aiob_task(...)` that imports
AIOB's primitives (TaskConfig, cache, tracing) but implements the
agent loop itself.

**Why I rejected it (with caveat):** duplicates AIOB's loop, judge
phase, trajectory bookkeeping, verdict-writing, and tracing-span
hierarchy. Every AIOB upstream change to any of those becomes a manual
re-port. The maintenance burden compounds over the AIOB-companion-paper
timeline.

The caveat: there's still an AgentStage-side `aiob_runner.py` (T23/T24
on Days 4-5), but it's a **thin wrapper** that calls `run_agentic` with
hooks set, not a reimplementation.

---

## 3. The two hooks: signatures and semantics

```python
PreTurnHook  = Callable[[int, List[Dict[str, Any]]], None]
PostTurnHook = Callable[[int, Any,                List[Dict[str, Any]]], None]
#                       turn_idx  response_or_None  trajectory_snapshot
```

### 3.1 `pre_turn_hook(turn_idx, messages)`

**Invoked:** at the very top of each `for turn in range(max_turns):`
iteration, **before** the `agent_step.plan` span opens or any LLM call
fires.

**Receives:**
- `turn_idx: int` — zero-based, increments per iteration up to `max_turns - 1`
- `messages: List[Dict[str, Any]]` — the message list that's about to
  be sent to the LLM. Includes system + user prompts on turn 0; grows
  with assistant messages + tool results on later turns.

**AgentStage's use:** record `t_turn_start_ms` (monotonic), snapshot
the message list for offline replay, prime per-turn predictor state.

**Hook contract:** read-only. The hook *may* call into the LLM client
or stager, but it **must not mutate** `messages` — AIOB doesn't make
defensive copies and a mutation here would change what the model sees.

### 3.2 `post_turn_hook(turn_idx, response, trajectory)`

**Invoked:** at the end of each `for turn` iteration body. Fires
**twice-via-one-call-site**:

1. **Normal path** — after all tool dispatches for the turn complete,
   just before the for loop wraps to the next iteration
2. **Early-break path** — after the LLM emits no tool calls and AIOB
   appends a `final_message` entry to trajectory and `break`s out of
   the loop

In both cases the hook fires *exactly once per turn*, never zero or
twice.

**Receives:**
- `turn_idx: int` — same idx as the matching `pre_turn_hook` call
- `response: Any` — the raw LLM response object (provider-shaped, as
  returned by `chat_completion`). Useful for tokens-used + thinking-
  block extraction.
- `trajectory: List[Dict[str, Any]]` — the trajectory **as it stands
  right now**, including any tool-call entries this turn appended.
  Snapshot semantics: AIOB appends to this list as the run progresses,
  so the hook sees the latest state but the list will continue
  growing on subsequent turns.

**AgentStage's use:** parse the streaming response for thinking-block
deltas, run the predictor against the prior, emit `DataHint`(s) to
the stager, write per-turn entries into `prediction.json` /
`cost.json`, sample cache temperature post-tool-call.

**Hook contract:** read-only with respect to `messages` (not passed
here, so trivially honored) and `trajectory` (must not mutate; AIOB
continues appending to it next iteration).

### 3.3 Insertion sites in `agentiobench/runner.py`

There are three call sites, all inside `run_agentic`:

```python
with workflow.run(args=wf_args):
    turn = -1
    for turn in range(max_turns):
        _safe_call_hook(pre_turn_hook, "pre_turn_hook", log, turn, messages)
        #             ┑ SITE 1 — top of every iteration
        log.debug(f"[agent] turn {turn}/{max_turns}")

        # ... agent_step.plan + llm_tracer.call + chat_completion ...

        tool_calls = extract_tool_calls(response)
        text       = extract_text(response)

        if not tool_calls:
            trajectory.append({"turn": turn, "type": "final_message", ...})
            _safe_call_hook(post_turn_hook, "post_turn_hook", log,
                            turn, response, trajectory)
            #          ┑ SITE 2 — early-break path
            break

        # ... append assistant_msg, dispatch tools, append to trajectory ...

        _safe_call_hook(post_turn_hook, "post_turn_hook", log,
                        turn, response, trajectory)
        #          ┑ SITE 3 — normal end of turn

    judge_span_id = new_id()       # ← outside the for-turn loop; judge phase
    with judge.evaluate(...):
        # ... verdict assembly ...
```

Both post-hook sites pass identical args. The duplication is
deliberate — a single trailing hook after the for-loop would not see
the LLM response for the final turn (which is in scope only inside
the loop body), and would also fire after `judge.evaluate` rather than
after the agent's last tool execution.

### 3.4 `_safe_call_hook` — the exception isolation contract

```python
def _safe_call_hook(hook, name, log, *args):
    if hook is None:
        return
    try:
        hook(*args)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        log.warning(f"{name} raised {type(exc).__name__}: {exc}")
```

A broken AgentStage hook must not corrupt AIOB benchmark results.
Without this guard, a misbehaving predictor could throw mid-loop and
leave the run in a half-validated state — failing `verdict.json` to
write at all, breaking the entire campaign's reproducibility.

The trade-off is loud-failure-vs-quiet-degradation. We accept quiet
degradation here because:

1. AIOB's `log.warning` makes the failure visible in run logs.
2. AgentStage's own side-channel artifacts (`prediction.json` etc.)
   will be incomplete or missing for that turn — the AgentStage layer
   notices its own breakage independently.
3. The eScience submission has hard data deadlines; one bad hook
   shouldn't waste a $0.40 multi-turn run.

If AgentStage development ever wants to *enforce* hook correctness,
the right place is a `--strict-hooks` flag on AgentStage's runner,
not in AIOB.

### 3.5 Contract verification — `tests/test_run_agentic_hooks.py`

Each property in §3.1–3.4 is pinned by a behavioral test on the AIOB
branch (committed in `813d477`). The tests mock `chat_completion`,
`extract_tool_calls`, `extract_text`, `extract_llm_metrics`,
`dispatch_tool`, the tracing runtime, `is_netns_isolated_runtime`,
`set_active_llm_accumulator`, `build_network_block`, `record_replay`,
`structured_validate`, `collect_output_files`, and
`tracing_metadata_dict` so that only the hook invocation logic
runs through real code.

| Test | Pins |
|---|---|
| `test_hooks_fire_per_turn_normal_path` | 3-turn run, every turn returns tool calls. Pre fires 3× with `turn=0,1,2` in order; post fires 3× with matching response object; `messages` grows monotonically across turns; `trajectory` snapshot grows monotonically. |
| `test_post_turn_hook_fires_on_early_break_path` | Turn 2 returns empty `tool_calls`. Pre fires for turns 0/1/2; post **also** fires for turn 2 (the early-break path) and sees the `{"type": "final_message"}` entry that AIOB just appended; `verdict["total_turns"] == 3`. |
| `test_broken_pre_hook_does_not_abort_run` | `pre_turn_hook` raises `RuntimeError` on every turn. Run still completes; post hook still fires; `verdict.json` still written. |
| `test_broken_post_hook_does_not_corrupt_verdict` | `post_turn_hook` raises `ValueError` on every turn. All turns still execute; `verdict["total_turns"]` correct; `verdict.json` round-trips through `json.load`. |
| `test_run_agentic_without_hooks_is_unchanged` | `run_agentic(cfg)` with no hook kwargs runs to completion exactly as before the integration patch — the backwards-compat guarantee the submodule-pin policy depends on. |

**How to run them locally:**

```bash
cd external/benchmarks/agentiobench
uv venv .venv-test --python 3.12
uv pip install --python .venv-test/bin/python -e '.[science]' pytest
.venv-test/bin/python -m pytest tests/test_run_agentic_hooks.py -v
```

Expected: `5 passed`. The `.venv-test/` is gitignored by `9f07f9b`.

**Regression scope verified:** running the runner-adjacent suite
(`test_runner.py + test_run_agentic_hooks.py + test_cache.py +
test_llm.py + test_tracing.py`) gives **50/50 pass** with the patched
`runner.py`. The pre-existing `test_run_oneshot_uses_single_step_…`
continues to pass — the `run_oneshot` sibling code path is untouched.

**Pre-existing failures observed in the broader AIOB suite** (23
tests in `test_validation.py`, `test_tools.py`, `test_orchestrate.py`)
are all environment issues (`pyarrow`, `fastparquet`, `structlog`
missing from the `[science]` extras set) — none touch `run_agentic`,
the public API surface, or the hook plumbing. They fail equivalently
on AIOB main.

---

## 4. The public API re-export — `agentiobench/__init__.py`

Before the integration branch, `agentiobench/__init__.py` was an empty
docstring. Importing `agentiobench` did nothing. To use anything
useful, callers had to reach into submodules:

```python
from agentiobench.config        import TaskConfig
from agentiobench.utils.cache   import evict_dataset, measure_temperature
from agentiobench.runner        import run_agentic
from agentiobench.tracing       import configure_dftracer_env, tracing_from_environ
```

This was fine for AIOB's internal callers, but it forces every
downstream wrapper to know AIOB's private module layout. A refactor
that moves `cache.py` from `utils/` to `runtime/` breaks every
downstream silently.

### 4.1 The fix — lazy `__getattr__` re-exports

```python
__all__ = [
    "BenchmarkConfig", "TaskConfig",
    "configure_dftracer_env", "evict_dataset", "measure_temperature",
    "run_agentic", "tracing_from_environ",
]

_LAZY_EXPORTS: dict[str, str] = {
    "BenchmarkConfig":         "agentiobench.config",
    "TaskConfig":              "agentiobench.config",
    "configure_dftracer_env":  "agentiobench.tracing",
    "evict_dataset":           "agentiobench.utils.cache",
    "measure_temperature":     "agentiobench.utils.cache",
    "run_agentic":             "agentiobench.runner",
    "tracing_from_environ":    "agentiobench.tracing",
}

def __getattr__(name: str) -> Any:
    module_path = _LAZY_EXPORTS.get(name)
    if module_path is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib
    module = importlib.import_module(module_path)
    value = getattr(module, name)
    globals()[name] = value  # cache after first access
    return value

if TYPE_CHECKING:
    from agentiobench.config import BenchmarkConfig, TaskConfig
    from agentiobench.runner import run_agentic
    # ... etc; only seen by mypy/pyright
```

### 4.2 Why lazy and not eager

The first version of this commit (`f4d5723`) was **eager**: it did
`from agentiobench.runner import run_agentic` at the top of `__init__.py`.

That broke AgentStage's trace-only path. Reason: `agentiobench.runner`
imports `agentiobench.validation`, which imports `numpy` at module
load. AgentStage in trace-only mode does not run the AIOB runner —
it only loads task YAMLs (`TaskConfig`) and calls `evict_dataset`.
But the eager `__init__.py` pulled the whole chain on `import
agentiobench`, requiring numpy at every callsite — including in CI
smoke tests, linters, and IDE auto-completion.

The lazy version (`4b78210`) preserves the convenience of
`from agentiobench import TaskConfig` while making `import agentiobench`
itself essentially free. `run_agentic` only triggers the heavy import
chain on the line where it's accessed; anyone who never touches it
never pays for numpy.

The `TYPE_CHECKING` branch matters because static analyzers don't
follow `__getattr__`. Without the conditional imports, mypy would
report `agentiobench` as having no `TaskConfig` attribute, and IDE
autocomplete would be empty. The `if TYPE_CHECKING:` guard is a
runtime-noop / static-analyzer-only declaration.

### 4.3 Caching after first access

Each `__getattr__` resolution writes the value back to `globals()`.
This means subsequent lookups skip `importlib.import_module` (which
itself is cheap but not free) and resolve directly through the
normal module attribute lookup. Hot paths like `predictor.observe`
calling `from agentiobench import TaskConfig` in a loop pay the
lookup cost exactly once.

---

## 5. End-to-end usage from AgentStage (T23/T24, Days 4-5)

This is the code path AgentStage will use to drive an AIOB workload.
It does not exist yet — the snippet below is a forward reference for
when T23/T24 land. Keep this section in sync with the real
`src/agentstage/runners/aiob_runner.py` once it's written.

```python
# src/agentstage/runners/aiob_runner.py  (not yet written)

from pathlib import Path
from typing import Any

from agentiobench import (
    BenchmarkConfig, TaskConfig,
    evict_dataset, measure_temperature,
    run_agentic,
)

from agentstage.client import AnthropicClient
from agentstage.predictor import load_frozen_rules, Predictor
from agentstage.stager import Stager  # None for trace-only


def run_aiob_task(
    task_yaml: Path,
    *,
    model: str,
    seed: int,
    output_root: Path,
    enable_stager: bool,
) -> dict[str, Any]:
    """Drive one AIOB run through AIOB's run_agentic with AgentStage
    observation hooks. Writes outputs/<task>_<model>_e2e_s<seed>/."""
    task = TaskConfig.from_yaml(task_yaml)
    out_dir = output_root / f"{task.name}_{model}_e2e_s{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Pre-run cold cache (per CAMPAIGN.md §"Cold cache enforcement")
    evict_dataset(task.dataset_root)
    pre_temp = measure_temperature(task.dataset_root)
    (out_dir / "verdict_pre.json").write_text(json.dumps({"pre_run_temperature": pre_temp}))

    # AgentStage runtime state for this run
    predictor = Predictor(rules=load_frozen_rules(), workspace=task)
    stager = Stager(out_dir=out_dir) if enable_stager else None
    client = AnthropicClient(model=model, predictor=predictor, stager=stager)

    # The hooks AIOB will call:
    def on_pre_turn(turn: int, messages: list) -> None:
        predictor.on_turn_start(turn, messages)

    def on_post_turn(turn: int, response: Any, trajectory: list) -> None:
        # response.usage gives token counts; response.content[*].thinking
        # gives thinking-block text. Predictor was already fed live via
        # the AnthropicClient SSE intercept; here we just close out the
        # turn and dump per-turn artifacts.
        predictor.on_turn_complete(turn, response, trajectory)
        client.flush_cost_record(out_dir / "cost.json")

    # Build the AIOB config, then hand control to run_agentic
    cfg = BenchmarkConfig(
        task=task,
        model=ModelConfig(name=model, ...),
        # ... mode, provider, etc.
        output_dir=str(out_dir),
        dataset_dir=str(task.dataset_root),
    )

    verdict = run_agentic(cfg, pre_turn_hook=on_pre_turn, post_turn_hook=on_post_turn)

    # Post-run temperature, written into the same verdict.json AIOB created
    post_temp = measure_temperature(task.dataset_root)
    update_json(out_dir / "verdict.json", {"post_run_temperature": post_temp})

    return verdict
```

Notes:

- `chat_completion` inside AIOB calls the *upstream* LLM via the
  standard provider SDK by default. To route through AgentStage's
  client, we **also** monkey-patch `agentiobench.llm.chat_completion`
  in this runner (or set `cfg.provider.base_url` to point at the local
  client's HTTP endpoint when the proxy fallback is in use). The
  monkey-patch is per-run and undone at function exit, avoiding the
  global-state hazard from §2.4.
- The hooks are observers; the *actual* streaming intercept happens
  inside `AnthropicClient`, which sits in front of `chat_completion`.
  The post-turn hook just consolidates per-turn artifacts that the
  client emitted live.
- For the AgentStage trace-only path (Campaign A), pass
  `enable_stager=False` and a no-op `Stager` (or `None`). The
  predictor still observes; only the prefetch dispatch goes silent.

---

## 6. Robustness properties

| Property | Mechanism | Verified by | Why it matters |
|---|---|---|---|
| Backwards-compatible signature | Keyword-only params with `None` defaults | `test_run_agentic_without_hooks_is_unchanged` | Existing AIOB callers (AIOB CLI, companion-paper experiments) keep working unchanged |
| Hook failures don't kill the run | `_safe_call_hook` try/except + log.warning | `test_broken_pre_hook_does_not_abort_run` + `test_broken_post_hook_does_not_corrupt_verdict` | AgentStage development can't cost AIOB benchmark data |
| Hook fires exactly once per turn | Three explicit call sites covering top + normal-end + early-break paths | `test_hooks_fire_per_turn_normal_path` + `test_post_turn_hook_fires_on_early_break_path` | No double-counting in cost / prediction accounting; no missing turn in the predictor's state |
| `pre` receives the *outgoing* messages, `post` receives the *response* | Hook insertion sites bracket the LLM call | `test_hooks_fire_per_turn_normal_path` asserts `n_messages` growth + `response_step` identity | Predictor sees the same prompt the model saw; post-hook can extract tokens/thinking from the same response object AIOB used |
| Top-level `import agentiobench` is cheap | Lazy `__getattr__` in `__init__.py` | Live smoke test (`import agentiobench` in the agentstage venv without `[science]` extras succeeds) | AgentStage's trace-only path, linters, and IDEs work without `science` extras |
| Static analyzers see the surface | `if TYPE_CHECKING:` eager-import branch | (not pytest-verified; relies on mypy/pyright behavior) | mypy / pyright / IDE auto-complete remain useful |
| Submodule pin lockstep | AgentStage commit `2e5ff93` pins AIOB `9f07f9b` (including the tests) | `git submodule status` | Reviewers re-running the campaign get the same hooks AND the test suite that pins their behavior |

---

## 7. Trade-offs and limitations

Things this design deliberately **does not** do, in case a future
session is tempted to "fix" them:

1. **No streaming hook.** The hooks fire on turn boundaries, not on
   each SSE event. Per-SSE-event observation happens inside
   `AnthropicClient` (the client lib intercept), not in AIOB. Pushing
   it into AIOB would require AIOB to know about streaming providers'
   event shapes, which couples them tightly.

2. **No tool-dispatch hook.** AgentStage doesn't need per-tool-call
   observation from AIOB's side — DFTracer already captures per-tool
   I/O at the syscall layer, and the stager (when enabled) sees
   prefetch decisions from its own log. Adding a `pre_tool` /
   `post_tool` hook would be more surface area without a downstream
   consumer right now.

3. **No verdict-rewrite hook.** AgentStage cannot mutate `verdict.json`
   from inside AIOB's loop. AgentStage's post-run code writes its own
   side files (`prediction.json`, `staging_report.json`, etc.) and
   patches `verdict.json` with `pre_run_temperature` /
   `post_run_temperature` keys *after* `run_agentic` returns. This
   keeps AIOB's verdict structure under AIOB's control.

4. **No retry/retry-count on hook failure.** `_safe_call_hook` logs
   and moves on. If AgentStage wants retry semantics on a transient
   stager failure, that retry logic lives inside the hook itself.

5. **No hook ordering / priority.** Only one `pre_turn_hook` and one
   `post_turn_hook` per `run_agentic` call. If AgentStage ever needs
   multiple observers, the AgentStage-side runner can compose them
   into a single callable.

6. **No `dftracer_context` re-export.** The earlier CAMPAIGN.md draft
   listed it; that symbol doesn't actually exist in
   `agentiobench/tracing.py`. The real public surface is
   `configure_dftracer_env` (sets env vars for DFTracer subprocess
   instrumentation) and `tracing_from_environ` (reads them back).
   Tracing-runtime setup inside `run_agentic` uses a private helper
   (`_get_tracing_runtime`) that we intentionally do NOT re-export
   — it's an implementation detail.

---

## 8. Maintenance and evolution

### 8.1 When to bump the submodule pin

Bump the AgentStage submodule pin (`external/benchmarks/agentiobench`)
when:

- A new commit lands on `feat/agentstage-integration` that AgentStage
  needs (e.g. a new hook, a re-export addition, a bug fix in the loop
  that affects predictor input).
- AIOB main absorbs the integration commits (eventual merge) — at
  that point the submodule URL stays the same but the branch tracking
  changes from `feat/agentstage-integration` to `main`.

Bumping procedure:

```bash
cd external/benchmarks/agentiobench
git fetch origin
git checkout <new-sha>          # or git pull on the branch
cd ../../..
git add external/benchmarks/agentiobench
git commit -m "Bump agentiobench pin to <new-sha>"
```

### 8.2 When to expand the public API re-export

Add a new symbol to `_LAZY_EXPORTS` (and the `TYPE_CHECKING` branch
+ `__all__`) when:

- AgentStage's runner ends up reaching into a private AIOB submodule
  for something stable. The threshold is "second use" — once a
  private symbol is used in more than one AgentStage file, promote
  it to the public surface.
- An AIOB upstream rename or move would break AgentStage. Adding the
  re-export at the new location with an alias preserves the contract.

Do **not** re-export every AIOB symbol. The current 7-symbol surface
is intentional: it covers the use cases enumerated in §1 (per-turn +
per-run + import-surface) and nothing else.

### 8.3 When to add another hook

Resist. The cost of a new hook is a new call site, a new type alias,
new docs, and a new contract every downstream observer must respect.

Before adding `pre_tool_hook` or `pre_llm_hook` or
`post_judge_hook`, ask:

1. Can the use case be served by parsing the existing post-turn
   trajectory? (Often yes.)
2. Can it be served by a downstream observation layer (DFTracer,
   the client-lib SSE intercept)?
3. Is there *now* a concrete AgentStage callsite that needs it, or
   is it speculative?

If the answer to any of the first two is yes, or the third is
"speculative," don't add the hook.

### 8.4 Merging the branch back to AIOB main

Eventually `feat/agentstage-integration` should merge into AIOB
`main`. Timing depends on:

- The AIOB-companion-paper effort accepting the public-API stability
  promise (a §4.1 commitment is implicit in re-exports).
- A PR review (the test suite in `813d477` makes this lower-friction —
  five new tests document the contract; passing them is the merge
  criterion the reviewer can pin on).
- AgentStage's eScience submission landing, so the integration code
  is referenced from a published paper.

Once merged, our submodule pin moves from a `feat/*` SHA to a `main`
SHA, the AgentStage runner code is unchanged (the public API is
identical), and this document gets one-line-updated to note the
merge commit.

---

## 9. Open follow-ups

1. ~~**Run AIOB's pytest suite against `feat/agentstage-integration`.**~~
   **Done in `813d477`.** Five behavioral tests for the hook contract
   are now part of the AIOB branch (see §3.5). 50/50 pass on the
   runner-adjacent subset of AIOB's existing suite
   (`test_runner.py + test_cache.py + test_llm.py + test_tracing.py +
   test_run_agentic_hooks.py`). The 23 broader-suite failures are
   pre-existing environment issues (`pyarrow`/`fastparquet`/`structlog`
   missing from `[science]` extras) unrelated to the integration.

2. **Fix CAMPAIGN.md's `dftracer_context` reference.** §6 said
   `dftracer_context` is re-exported; it isn't. The real symbols are
   `configure_dftracer_env` and `tracing_from_environ`. Quick doc-only
   fix.

3. **`AnthropicClient` integration with monkey-patched
   `chat_completion`.** T19 lands the client; the per-run monkey-patch
   wrapper for AIOB's `agentiobench.llm.chat_completion` is the small
   piece that connects them. Lives in `src/agentstage/runners/aiob_runner.py`
   (T23).

4. **Per-run hook composition helper.** If AgentStage's runner ever
   needs to register multiple pre/post-turn observers (predictor
   AND stager AND cost-tracker AND temperature-logger), add a
   `compose_hooks(*fns)` utility on the AgentStage side. AIOB stays
   single-hook.

5. **Open a draft PR on `grc-iit/agentiobench` for visibility.** Not
   for immediate merge — for upstream awareness, any CI hooks the
   org has configured, and an in-progress citation when the AIOB
   companion paper drafts come together. The verified-tests state
   (§3.5) makes this lower-stakes than it would have been pre-`813d477`.

---

## 10. Quick reference

```python
# Cheap import — does NOT pull numpy etc.
import agentiobench

# Cheap names (config + cache, no heavy deps)
from agentiobench import TaskConfig, BenchmarkConfig
from agentiobench import evict_dataset, measure_temperature

# Heavy name (pulls validation → numpy chain on first access)
from agentiobench import run_agentic

# Tracing setup (env-driven, no heavy deps)
from agentiobench import configure_dftracer_env, tracing_from_environ

# Hook-driven run
verdict = run_agentic(
    cfg,
    pre_turn_hook=lambda turn, msgs:    predictor.on_turn_start(turn, msgs),
    post_turn_hook=lambda turn, resp, t: predictor.on_turn_complete(turn, resp, t),
)
```

Upstream branch: <https://github.com/grc-iit/agentiobench/tree/feat/agentstage-integration>
AgentStage submodule pin: see `external/benchmarks/agentiobench` (commit `2e5ff93` set it to AIOB `9f07f9b`, which includes the verified-tests commit `813d477`).

**Branch contents at the current pin:**
- `f4d5723` Add public API surface + `run_agentic` turn-hook integration points
- `4b78210` Make agentiobench public-API re-exports lazy
- `813d477` Verify run_agentic hooks fire correctly under mocked LLM + tools (5 tests, all pass)
- `9f07f9b` gitignore: `.venv-test/`
