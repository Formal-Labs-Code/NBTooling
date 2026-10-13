"""
A dependency graph computed the way the NBLyzer paper (Subotic, Milikic,
Stojic - "A Static Analysis Framework for Data Science Notebooks", ICSE-SEIP
'22) actually specifies it, instead of context_builder.py's
build_dependency_edges shortcut.

The gap this closes: build_dependency_edges walks cells in a single pass
in cell-*ID* order, tracking a running last_definer map that only ever
gets updated *after* a cell is visited. That silently assumes cell ID
order is execution order - which the paper's own motivating example
(Section 1, Figure 1) explicitly rejects: "cells, regardless of their
order in the notebook, can be executed... in *any* given sequence." A
name read in cell 3 but only defined in cell 4 is a completely ordinary,
detectable dependency under that model (cell 4's post-state satisfies
cell 3's pre-condition, full stop) - but build_dependency_edges can never
see it, because by the time it reaches cell 3 in its single forward pass,
cell 4 hasn't been visited yet. That gap is exactly what NBFix's
`order_dependent` bug class lives in (see experiments.md), and it's a
limitation of that one implementation, not of dependency graphs in
general.

The paper's real model (Definition 3.1, "Cell Propagation Dependency
Graph"): an edge c_i -> c_j exists iff phi(sigma#_ci, pre_cj) - cell i's
abstract post-state satisfies cell j's pre-condition - discovered by
propagating abstract state across the notebook to a fixpoint (or a
bounded depth K), not by a single ID-ordered scan. That's exactly what
this module builds, reusing the *same* fixpoint engine
(runner.runners.Runner.inter_fixpoint_runner) StaleCellAnalysis already
uses - same shape (Analysis subclass + AbstractState subclass), just a
different abstract domain: "which names are defined so far" instead of
"how stale is this variable." Concretely:

- DependencyAS (abs_states/dependency_abs_state.py) tracks defined names
  along a propagation path (no staleness levels needed).
- phi_condition here is a plain non-empty-intersection check
  (`bool(current & pre)`), deliberately weaker than StaleCellAnalysis's
  `pre <= current` (full subset) - a cell can depend on several different
  upstream cells for different names, so satisfying *one* shared name
  with a candidate source cell is enough to justify visiting it; the
  fixpoint runner's own recursion produces the actual per-name binding via
  DependencyAS.condition when the candidate cell's own CFG is walked.

Not registered in NBFix.all_analyses/constants.py - that dict is for
analyses implementing analyze_notebook() and producing user-facing
Result/ErrorInfo findings, which this isn't (it's a graph, not a bug
flagger). It IS wired in where it matters: context_builder.py's
build_dependency_edges is a bare alias for build_fixpoint_dependency_edges
below, so every LLM context builder (DetectBugsEvent's and
DetectApiSequenceEvent's) already gets these paper-correct edges by
default, with an explicit dependency_edges override param on both events
for swapping in an alternative edge set (e.g. type_shape_analysis.py's
build_pruned_dependency_edges) without changing the default.
"""
from collections import defaultdict
from copy import deepcopy

from .abs_states.dependency_abs_state import DependencyAS
from .analysis import Analysis
from .runner.analyses_utils import AssignParserVisitor
from .runner.analysis_results import Result
from .runner.runners import Runner
from .runner.stats import Stats
from ..parser import ast_nodes as ast

# Direct-edge scope only, matching build_dependency_edges' own semantics
# (an immediate producer/consumer edge, not a transitive-closure walk -
# context_builder._connected_component already computes transitive
# reachability separately, on top of whichever edges dict it's given).
# K=2 in inter_fixpoint_runner's terms means: run the seed cell itself
# (K=2, real), then one more real hop into each candidate whose
# pre-condition the seed's post-state satisfies (K=1, real - this is
# where DependencyAS.condition actually fires and records the edge), then
# stop (any further recursion would run at K=0, which inter_fixpoint_runner
# treats as a no-op without even visiting the next cell - see runners.py).
_DIRECT_EDGE_DEPTH = 2


class DependencyAnalysis(Analysis):
    def F_transformer(self, cfg_node, a_state: DependencyAS, cell_IR):
        as_transformed = deepcopy(a_state)
        if not cfg_node.ast_node:
            return a_state
        if isinstance(cfg_node.ast_node, ast.Assign):
            assign_parser = AssignParserVisitor()
            assign_parser.parse_assign(cfg_node.ast_node)
            for name in assign_parser.def_variables:
                # Direct assignment, not setdefault: a redefinition here
                # shadows whatever provenance this name had from further
                # back in the propagation path - this cell is now the
                # correct, most-recent definer for it (see DependencyAS's
                # docstring for why per-name provenance matters once
                # K > 2 lets propagation chain through multiple cells).
                as_transformed.defined_vars[name] = cell_IR.cell_id
        return as_transformed

    def combine_states(self, states: list[DependencyAS]) -> DependencyAS:
        res_state = DependencyAS()
        for s in states:
            res_state.aug_join(s)
        return res_state

    def phi_condition(self, current: set, pre: set, cell_IR) -> bool:
        return bool(current & pre)

    def calculate_pre(self, cell_IR):
        return cell_IR.UDA.unbound_final - self.imports

    def summarize_result(self, result: Result) -> Result:
        return result

    def _find_all_imports(self, notebook_IR) -> None:
        self.imports = set()
        for cell_IR in notebook_IR.values():
            self.imports.update(cell_IR.UDA.imports)


def build_fixpoint_dependency_edges(notebook_IR) -> dict[int, set[int]]:
    """
    Seeds the fixpoint propagation from *every* cell in turn (treating
    each, one at a time, as the source whose definitions might satisfy
    some other cell's pre-condition), and merges whatever direct edges
    each seed's run discovers. Unlike a single incremental STALE-style
    run (seeded from one changed cell), this builds the *whole* graph
    from scratch, order-independently - a candidate cell earlier in
    notebook order is exactly as eligible as one later, since
    phi_condition never looks at cell IDs.
    """
    analysis = DependencyAnalysis()
    analysis.find_necessary_cells(notebook_IR)
    analysis._find_all_imports(notebook_IR)

    edges: dict[int, set[int]] = {cell_id: set() for cell_id in notebook_IR}
    for seed_cell in sorted(notebook_IR):
        # Pre-seed with this cell's own defined names, same as
        # StaleCellAnalysis._prepare_init_as does for the changed cell -
        # relying on the CFG walk to *discover* them from an empty start
        # doesn't work in general: intra_fixpoint_runner only propagates
        # past a node when its transform changes the state relative to a
        # fresh default, so a cell whose Assign is preceded by a no-op
        # sub-expression node (e.g. `data = len(data)` compiles to a Call
        # node THEN an Assign node) can silently never reach its own
        # Assign from a truly empty seed - confirmed empirically: without
        # this, `data = len(data)` produced an empty post-state.
        seed_ir = notebook_IR[seed_cell]
        init_as = DependencyAS({name: seed_cell for name in seed_ir.UDA.defined_vars})
        runner = Runner(Stats(), defaultdict(DependencyAS), notebook_IR)
        result = runner.inter_fixpoint_runner(
            analysis, seed_cell, abstract_state=init_as,
            K=_DIRECT_EDGE_DEPTH, cpath=[], results=Result(),
        )
        for path_result in result.path_results:
            target_cell = path_result.path[-1]
            # Per-name definer, not path[0] (the seed): once K > 2 lets
            # propagation chain through intermediate cells, the seed
            # isn't necessarily who actually defined the name a
            # downstream cell reads - see DependencyAS.condition, which
            # encodes the real per-name definer in each ErrorInfo's
            # `.line` field for exactly this reason.
            for err in path_result.error_infos:
                definer_cell = err.line
                if target_cell != definer_cell:
                    edges[target_cell].add(definer_cell)

    return edges


# Bare alias, not a separate implementation - kept for callers that
# migrated from an older, cell-ID-ordered build_dependency_edges (see
# this module's docstring). Lives here, in nbcore, rather than in any one
# tool's llm/context_builder.py, because both NBFix's general
# bug-detection context and NBFix's API-sequence context consume the
# same edge graph - keeping it here avoids either tool depending on the
# other's code for a plain graph utility.
build_dependency_edges = build_fixpoint_dependency_edges


def reverse_edges(edges: dict[int, set[int]]) -> dict[int, set[int]]:
    """Flips a dependency-edges graph: reverse[X] is the set of cells
    that depend on X (edges[X] is the set of cells X depends on). Shared
    graph utility - both _connected_component below and NBCompile's SCC
    detection (Kosaraju's algorithm needs a forward and a reverse DFS
    pass) need this exact operation."""
    reverse: dict[int, set[int]] = {}
    for cell, deps in edges.items():
        for dep in deps:
            reverse.setdefault(dep, set()).add(cell)
    return reverse


def _connected_component(edges: dict[int, set[int]], cell_id: int) -> set[int]:
    """Transitive closure over `edges`, following both directions (what
    cell_id depends on, and what depends on cell_id). Shared graph
    utility, same reasoning as build_dependency_edges above."""
    reverse = reverse_edges(edges)

    seen = {cell_id}
    frontier = {cell_id}
    while frontier:
        next_frontier = set()
        for cell in frontier:
            for neighbor in edges.get(cell, set()) | reverse.get(cell, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    next_frontier.add(neighbor)
        frontier = next_frontier
    return seen
