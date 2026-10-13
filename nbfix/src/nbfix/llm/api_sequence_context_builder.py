"""
Context for LLM-based API call-sequence detection.

Scans the whole notebook in one LLM call rather than one cell at a time
(see api_sequence_prompts.py's docstring for the latency measurement
behind that) - but when a specific cell just changed (focus_cell), the
prompt is narrowed to that cell's dependency-connected component instead
of the entire notebook, reusing context_builder.py's own subgraph-scope
machinery (build_dependency_edges/_connected_component - the same real,
provenance-fixed dependency graph the general bug-detection path already
uses, not a separate implementation). A cell with no dependency
relationship to what just changed cannot plausibly need re-checking, so
there's no reason to keep spending prompt tokens (and, at large notebook
sizes, real prefill latency) re-showing it on every edit.

focus_cell=None (the default, used when the "Choose Active Analyses"
toggle is switched on) scans every cell - there's no "what just changed"
to narrow around at that point, a full baseline scan is the point. It's
still filtered, though: the whole notebook is partitioned into its
disjoint dependency-connected components (every genuinely independent
"sub-notebook", not just the one component focus_cell would land in),
and a component is only included at all if at least one cell in it
contains a method call (`obj.method(...)`) anywhere - see
_has_method_call. This bug class is inherently about a stateful object's
method-call order; a component built entirely from plain assignments,
builtin calls (print/len/...), or arithmetic cannot possibly violate it,
so there's no reason to spend a single token showing it to the model.
Purely syntactic (stdlib `ast`, not NBFix's own CFG/UDA machinery) and
conservative on a parse failure (assume relevant rather than silently
dropping something NBFix's own parser can't handle either).

Concurrency was considered as a way to check independent components in
parallel instead of filtering, and measured (not assumed) to not help on
this project's actual Ollama setup: 3 concurrent calls took ~2.15s total
against ~2.21s for the same 3 calls run sequentially - Ollama serializes
requests here regardless of how many arrive concurrently (confirmed via
per-call timings that grow with queue position, the signature of
queueing, not batching). True parallel inference would need
OLLAMA_NUM_PARALLEL > 1, which needs proportionally more memory per
concurrent slot - checked against this project's actual hardware (Apple
M3 Pro, 18GB unified memory, already ~9.5GB used by one loaded
qwen2.5-coder:14b instance) and judged too tight to risk. Filtering out
irrelevant components is the one purely-additive option: it never makes
a request slower, unlike concurrency, which could make things worse
under memory pressure.

A component-size cap (skip the LLM call entirely past a cell-count
threshold, to bound worst-case latency on a densely-connected chain) was
tried separately and reverted - a live user report of a misattributed
finding appeared right after that change landed, and rather than keep
debugging a discrepancy that couldn't be reproduced in isolation while
the user was blocked, the cap was undone on request. That was a
heuristic latency/coverage tradeoff; this method-call filter is not - a
component with zero method calls provably cannot contain this bug class,
so skipping it never trades away real coverage. If the size cap is
revisited, treat it as a new, separately-tested change regardless of
this one's outcome.

Includes each cell's actual *execution* status (has this cell's current
code actually run in the kernel, not just does it exist in the file) -
see IntermediateRepresentations.last_ran_code. `last_ran_code !=
cell_code` means this cell's current code has not actually executed -
either it's never been run at all (last_ran_code defaults to "") or it
was run before but has since been edited without being re-run. A
prerequisite call sitting in a cell that exists in the file but hasn't
actually run does NOT satisfy anything in a real kernel - found via a
direct user question ("doesn't cell 11 need fit to actually run first -
I can execute it directly"), not independently.
"""
import ast
from dataclasses import dataclass

from nbcore.analyses.dependency_analysis import _connected_component, build_dependency_edges


@dataclass
class ApiSequenceContext:
    cells: dict[int, str]
    not_yet_run: set[int]
    # Every cell actually examined, including ones filtered out for
    # having no method calls at all - `cells` is only what's sent to the
    # LLM, but a filtered-out cell's "no violation" answer is still
    # confidently known (proven, not guessed), so the caller can safely
    # treat it as checked too (see DetectApiSequenceEvent.checked_cells).
    considered_cells: set[int]


def _has_method_call(code: str) -> bool:
    """True if `code` contains any obj.method(...) call anywhere - the
    only shape of call this bug class cares about. A bare function call
    (print(x), len(x)) or a constructor (StandardScaler()) doesn't count
    on its own; a cell can still be relevant via a *different* cell in
    the same component that does have one, which is why filtering
    happens per-component, not per-cell (see build_api_sequence_context)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return True
    return any(
        isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        for node in ast.walk(tree)
    )


def _partition_components(edges: dict[int, set[int]], all_cells) -> list[set[int]]:
    """Splits all_cells into its disjoint dependency-connected components -
    every genuinely independent "sub-notebook", not just one target's."""
    remaining = set(all_cells)
    components = []
    while remaining:
        seed = next(iter(remaining))
        component = _connected_component(edges, seed)
        components.append(component)
        remaining -= component
    return components


def build_api_sequence_context(notebook_IR, focus_cell: int = None, dependency_edges=None) -> ApiSequenceContext:
    """dependency_edges, if given, overrides build_dependency_edges' output
    entirely - same escape hatch as DetectBugsEvent's context builders
    (see context_builder.build_cell_context's docstring), e.g. for
    analyses/type_shape_analysis.py's build_pruned_dependency_edges."""
    edges = dependency_edges if dependency_edges is not None else build_dependency_edges(notebook_IR)

    if focus_cell is None:
        considered_cells = set(notebook_IR.keys())
        relevant_cells: set[int] = set()
        for component in _partition_components(edges, notebook_IR.keys()):
            if any(_has_method_call(notebook_IR[cid].cell_code) for cid in component):
                relevant_cells |= component
    else:
        considered_cells = _connected_component(edges, focus_cell)
        if any(_has_method_call(notebook_IR[cid].cell_code) for cid in considered_cells):
            relevant_cells = considered_cells
        else:
            relevant_cells = set()

    return ApiSequenceContext(
        cells={cid: notebook_IR[cid].cell_code for cid in relevant_cells},
        not_yet_run={
            cid for cid in relevant_cells
            if notebook_IR[cid].last_ran_code != notebook_IR[cid].cell_code
        },
        considered_cells=considered_cells,
    )
