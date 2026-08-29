from dataclasses import dataclass

from nbcore.analyses.dependency_analysis import build_fixpoint_dependency_edges, reverse_edges
from nbcore.ir.intermediate_representations import IntermediateRepresentations

LOOP_TODO_COMMENT = (
    "  # TODO: this was a cycle of interactively re-run cells - there's no "
    "closed-form exit condition to recover from notebook history. Fill one "
    "in, or opt in to LLM assist to suggest one."
)


def find_start_cells(edges: dict[int, set[int]]) -> list[int]:
    """Cells with no dependencies - the natural entry points for a linear
    script, since nothing needs to happen before them."""
    return sorted(cell_id for cell_id, deps in edges.items() if not deps)


def find_sccs(edges: dict[int, set[int]]) -> list[list[int]]:
    """Kosaraju's algorithm: strongly connected components of the
    dependency graph. Any SCC with more than one cell is a genuine
    cycle - the notebook was interactively re-run, not executed once
    top-to-bottom, and has no single valid linear order. Returns SCCs in
    an arbitrary order (linearize() below topologically sorts them via
    the condensation graph); each SCC's own cell list is sorted.

    dependency_analysis.py's own test suite guarantees edges never
    contains a self-edge (test_never_produces_a_self_edge), so a
    singleton SCC here is always non-cyclic - no separate self-loop
    check needed.
    """
    order: list[int] = []
    visited: set[int] = set()

    def dfs1(node: int) -> None:
        visited.add(node)
        for neighbor in sorted(edges.get(node, set())):
            if neighbor not in visited:
                dfs1(neighbor)
        order.append(node)

    for node in sorted(edges):
        if node not in visited:
            dfs1(node)

    reverse = reverse_edges(edges)
    visited2: set[int] = set()
    sccs: list[list[int]] = []

    def dfs2(node: int, component: list[int]) -> None:
        visited2.add(node)
        component.append(node)
        for neighbor in reverse.get(node, set()):
            if neighbor not in visited2:
                dfs2(neighbor, component)

    for node in reversed(order):
        if node not in visited2:
            component: list[int] = []
            dfs2(node, component)
            sccs.append(sorted(component))

    return sccs


def _condensation(
    edges: dict[int, set[int]], sccs: list[list[int]]
) -> tuple[dict[int, int], dict[int, set[int]]]:
    """Collapses each SCC into one node. Collapsing SCCs out of any
    directed graph always yields a DAG back (a cycle spanning two
    different SCCs would mean they're actually one SCC) - that's what
    makes a uniform topological-sort algorithm possible even though the
    original cell graph might not be a DAG."""
    scc_of: dict[int, int] = {}
    for scc_id, cells in enumerate(sccs):
        for cell in cells:
            scc_of[cell] = scc_id

    condensed: dict[int, set[int]] = {scc_id: set() for scc_id in range(len(sccs))}
    for cell, deps in edges.items():
        for dep in deps:
            if scc_of[cell] != scc_of[dep]:
                condensed[scc_of[cell]].add(scc_of[dep])

    return scc_of, condensed


def _topological_order(condensed_edges: dict[int, set[int]]) -> list[int]:
    """Kahn's algorithm over the condensation graph (guaranteed acyclic -
    see _condensation's docstring). condensed_edges[X] = sccs X depends
    on; a scc becomes ready once every scc it depends on has already
    been emitted."""
    remaining = {scc: set(deps) for scc, deps in condensed_edges.items()}
    dependents: dict[int, set[int]] = {scc: set() for scc in condensed_edges}
    for scc, deps in condensed_edges.items():
        for dep in deps:
            dependents[dep].add(scc)

    ready = sorted(scc for scc, deps in remaining.items() if not deps)
    order: list[int] = []
    while ready:
        scc = ready.pop(0)
        order.append(scc)
        newly_ready = []
        for dependent in dependents.get(scc, set()):
            remaining[dependent].discard(scc)
            if not remaining[dependent] and dependent not in order and dependent not in ready:
                newly_ready.append(dependent)
        ready.extend(newly_ready)
        ready.sort()

    return order


@dataclass
class ScriptBlock:
    cell_ids: list[int]
    is_cycle: bool


def linearize(edges: dict[int, set[int]]) -> list[ScriptBlock]:
    """The core algorithm: dependency graph -> ordered list of blocks.
    Each block is either one cell (the common case) or several cells
    that form a genuine cycle, in original cell-id order, meant to be
    wrapped in a loop by render_script below.

    Doesn't try to pick a "nice" order among several independent chains
    when the graph branches - any topologically valid order is used,
    picked deterministically (lowest scc id first) rather than optimized
    for readability. Revisit once this is validated end to end."""
    sccs = find_sccs(edges)
    scc_of, condensed = _condensation(edges, sccs)
    order = _topological_order(condensed)
    return [ScriptBlock(cell_ids=sccs[scc_id], is_cycle=len(sccs[scc_id]) > 1) for scc_id in order]


def render_script(notebook_IR: dict[int, IntermediateRepresentations], blocks: list[ScriptBlock]) -> str:
    rendered: list[str] = []
    for block in blocks:
        if not block.is_cycle:
            rendered.append(notebook_IR[block.cell_ids[0]].cell_code)
            continue

        body = "\n".join(notebook_IR[cell_id].cell_code for cell_id in block.cell_ids)
        indented = "\n".join(f"    {line}" if line else "" for line in body.splitlines())
        rendered.append(f"while True:{LOOP_TODO_COMMENT}\n{indented}")

    return "\n\n".join(rendered)


def convert_to_script(notebook_IR: dict[int, IntermediateRepresentations]) -> str:
    edges = build_fixpoint_dependency_edges(notebook_IR)
    blocks = linearize(edges)
    return render_script(notebook_IR, blocks)
