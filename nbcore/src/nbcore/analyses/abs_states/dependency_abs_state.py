from __future__ import annotations
from copy import deepcopy

from .abs_state import AbstractState
from ..runner.analysis_results import ErrorInfo, ErrorType


class DependencyAS(AbstractState):
    """
    Abstract state for DependencyAnalysis (see dependency_analysis.py):
    tracks which names are known to be defined "so far" along a
    propagation path through the notebook, and *which cell* most
    recently (along this specific path) defined each one - the
    dependency-graph analogue of CodeImpactAS.impacted_variables, minus
    the staleness levels. defined_vars maps name -> the cell_id that
    provided it; that provenance matters once K > 2 lets propagation
    chain through more than one intermediate cell (see
    dependency_analysis.py's _DIRECT_EDGE_DEPTH comment) - without it, a
    cell several hops downstream would get attributed back to the
    original seed cell for *every* name reachable along the path, not
    just the ones the seed itself actually defines. Concretely: seed
    A defines x; B reads x and defines y; C reads y. C's dependency is on
    B (which defined y), not on A - A never even appears in C's own
    UDA.unbound_final. Tracking provenance per name is what keeps that
    distinction correct instead of collapsing every hop back to the seed.
    """

    def __init__(self, defined_vars: dict[str, int] | None = None):
        self.defined_vars: dict[str, int] = deepcopy(defined_vars) if defined_vars else {}

    def __eq__(self, other: DependencyAS) -> bool:
        return self.defined_vars == other.defined_vars

    def projection(self) -> set[str]:
        return set(self.defined_vars.keys())

    def aug_join(self, other: DependencyAS) -> None:
        for var, definer in other.defined_vars.items():
            self.defined_vars.setdefault(var, definer)

    def contains(self, other: DependencyAS) -> bool:
        return other.defined_vars.items() <= self.defined_vars.items()

    def condition(self, cell_IR, node, errors) -> list:
        """
        Fires whenever this cell (cell_IR) reads (per its whole-cell
        UDA.unbound_final, same source of truth context_builder's own
        build_dependency_edges uses) a name this abstract state already
        has recorded as defined somewhere upstream in the current
        propagation path - i.e. "this cell depends on that name."
        Called once per CFG node during intra_fixpoint_runner's walk
        (matching CodeImpactAS's own convention), but unbound_final is a
        whole-cell set so the check is really whole-cell; the
        `new_error not in errors` dedup (also matching CodeImpactAS)
        collapses the resulting repeats into one entry per name.

        The line number carries the *definer* cell_id (self.defined_vars[name]),
        not a real source location - build_fixpoint_dependency_edges reads
        it directly rather than assuming path[0] is the definer, which
        would misattribute anything more than one hop away from the seed.
        """
        for name, definer_cell in self.defined_vars.items():
            if name in cell_IR.UDA.unbound_final:
                new_error = ErrorInfo(
                    cell_IR.cell_id, definer_cell, name, ErrorType.CRITICAL, f"depends on '{name}'",
                )
                if new_error not in errors:
                    errors.append(new_error)
        return errors

    def __str__(self) -> str:
        return "Defined: " + ", ".join(f"{k} (cell {v})" for k, v in sorted(self.defined_vars.items()))
