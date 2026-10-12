"""
Heuristic, best-effort type/shape tagging across cell reassignment -
built to plug the specific gap `build_dependency_edges`
(src/nbfix/llm/context_builder.py) can't: it tells you a cell *depends*
on a name another cell defines, but nothing about whether a reassignment
along the way actually changed what that name *means*. That's exactly the
`cross_cell_semantic` bug shape (e.g. `data = [1,2,3]` then later
`data = len(data)` - same name, list becomes int, a downstream
`data.append(...)` crashes). Dependency edges (now flow-sensitive, so
`data = len(data)` does depend on the earlier `data`) still only say
*that* a cell depends on a name, not that its meaning changed. This fills
that gap: track a coarse type tag per name across cells, and flag it when
a reassignment changes the tag.

Not a real type inference engine - a small, closed set of tags inferred
from purely local syntax (literals, and a knowledge-base of well-known
constructors/functions with predictable return types, same style as
DataLeakAnalysis's resetKB/taintKB/etc.). Anything not recognized tags as
`None` (unknown), which must stay a real, distinct outcome - conflating
"unknown" with "unchanged" would manufacture false positives on any code
this heuristic doesn't understand.

Deliberately not wired into NBFix.all_analyses/constants.py yet - a
standalone function returning the same Result/ErrorInfo shape the other
four analyses produce, callable directly (e.g. by
scripts/benchmark_llm.py, or fed straight into
context_builder.build_full_notebook_context's deterministic_findings
param) without needing the CFG-level fixpoint/abstract-interpretation
machinery (Analysis/Runner/F_transformer) the other four - and
dependency_analysis.py - use for incremental re-analysis. Type-tag
inference doesn't need CFG-level precision (it only ever looks at a
single Assign node's RHS expression), so _compute_type_tags below uses a
lighter, purpose-fit fixpoint instead: iterate the whole-notebook set of
plain-Assign statements to a fixpoint rather than a single pass, so
`_infer_tag`'s name-to-name lookups resolve correctly regardless of
which cell's text comes first - genuinely order-independent now, not
just labeled that way (an earlier version of this module claimed
order-independence while actually doing a single ID-ordered pass -
see _compute_type_tags's docstring for why that was wrong and what
replaced it). No incremental re-run story needed yet. Promote to a full
Analysis subclass if it earns a permanent spot (CLI -a flag, live
JupyterLab diagnostics).
"""
from ..parser import ast_nodes as ast
from .dependency_analysis import build_fixpoint_dependency_edges
from .runner.analysis_results import ErrorInfo, PathResult, Result

# Call/constructor name -> the coarse tag its return value gets. Bare
# function/attribute name only (no import/qualified-name resolution,
# same tradeoff DataLeakAnalysis's own knowledge bases already make) -
# a local `def read_csv(...)` would false-positive-match pandas's, but
# that's an acceptable, well-precedented scope for a heuristic pass.
_RETURN_TYPE_KB = {
    "list": "list", "sorted": "list",
    "dict": "dict",
    "set": "set", "frozenset": "set",
    "tuple": "tuple",
    "str": "str",
    "int": "int", "len": "int",
    "float": "float",
    "bool": "bool",
    "sum": "numeric",
    "DataFrame": "DataFrame", "read_csv": "DataFrame", "read_json": "DataFrame", "read_excel": "DataFrame",
    "array": "ndarray", "zeros": "ndarray", "ones": "ndarray", "arange": "ndarray",
}

_NUMERIC_TAGS = {"int", "float", "numeric"}


def _infer_tag(node, last_tag: dict[str, str | None]) -> str | None:
    """
    Best-effort tag for a single expression node, using only what's
    visible locally (literals, a known call's return type) or already
    tracked (a name-to-name assignment carries forward the source name's
    last known tag). Returns None - a real, load-bearing "don't know",
    not a stand-in for "unchanged" - for anything not recognized.
    """
    if isinstance(node, ast.List):
        return "list"
    if isinstance(node, ast.Dict):
        return "dict"
    if isinstance(node, ast.Tuple):
        return "tuple"
    if isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, bool):  # bool is an int subclass - must check first
            return "bool"
        if isinstance(value, int):
            return "int"
        if isinstance(value, float):
            return "float"
        if isinstance(value, str):
            return "str"
        if value is None:
            return "NoneType"
        return None
    if isinstance(node, ast.Name):
        return last_tag.get(node.id)
    if isinstance(node, ast.Call):
        func = node.func
        if isinstance(func, ast.Name):
            name = func.id
        elif isinstance(func, ast.Attribute):
            name = func.attr
        else:
            return None
        return _RETURN_TYPE_KB.get(name)
    if isinstance(node, ast.BinOp):
        # Needed for the dep-density fixtures' own chain_var_i =
        # chain_var_{i-1} + 1 pattern - without this, every arithmetic
        # expression tags as unknown and the stability walk below
        # (build_pruned_dependency_edges/label_stable_dependencies) can
        # never earn confidence about a name defined this way. Only two
        # closed, low-risk cases: numeric-preserving arithmetic, and
        # str/list concatenation via `+`.
        left = _infer_tag(node.left, last_tag)
        right = _infer_tag(node.right, last_tag)
        if left in _NUMERIC_TAGS and right in _NUMERIC_TAGS:
            return "float" if "float" in (left, right) else "int"
        if left == right in ("str", "list", "tuple"):
            return left
        return None
    return None


def _find_line(cell_code: str, name: str) -> int:
    """Same text-search approach idle_cell_analysis.py/isolated_cell_analysis.py
    already use for locating a label's line - node.lineno isn't relied on
    elsewhere in this codebase, matching that precedent rather than
    introducing a new, untested way of getting a line number."""
    for line, line_text in enumerate(cell_code.split("\n")):
        if line_text.find(name) != -1:
            return line + 1
    return 1


def _compute_type_tags(notebook_IR) -> dict[str, list[tuple[int, str | None]]]:
    """
    Order-independent type-tag inference across the whole notebook.

    The previous version of this walk processed cells in a single pass
    in cell-ID order, so `_infer_tag`'s `ast.Name` lookups (`a = b` needs
    b's tag) could only ever resolve a name whose *own* defining cell
    happened to come earlier in the notebook's text. That's the exact
    same wrong assumption `context_builder.build_dependency_edges` used
    to make (see that module's comment, and experiments.md findings
    10-11) - cell-ID order isn't execution order, so `c = [1]` in cell 5
    and `a = c` in cell 2 should resolve just as well as the reverse.

    Fixed by iterating the whole-notebook inference to a fixpoint instead
    of a single pass: every plain `name = <expr>` Assign in the notebook
    is re-evaluated repeatedly (each pass sees the previous pass's
    resolved tags) until nothing changes, so a chain resolves correctly
    regardless of which cell text comes first. Bounded at
    len(triples) + 1 passes - enough for any acyclic chain to fully
    propagate; a genuine reference cycle (`a = b; b = a`) just never
    resolves past None, which is the correct, conservative outcome for
    something truly circular.

    Returns name -> list of (cell_id, tag) for every plain-Assign
    definition site of that name, in its fully fixpoint-resolved tag -
    the raw per-definer data both TYPE_CHANGE detection (do any of a
    name's definers disagree) and stability checks (do *all* of them
    agree) are built from.
    """
    triples: list[tuple[int, str, object]] = []
    for cell_id in sorted(notebook_IR):
        ir = notebook_IR[cell_id]
        for node in ast.walk(ir.AST):
            if not isinstance(node, ast.Assign):
                continue
            if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
                continue  # tuple-unpacking/chained targets: not handled by this pass
            triples.append((cell_id, node.targets[0].id, node.value))

    per_triple_tag: list[str | None] = [None] * len(triples)
    last_tag: dict[str, str | None] = {}

    changed = True
    passes = 0
    while changed and passes <= len(triples):
        changed = False
        passes += 1
        for i, (_, name, rhs_node) in enumerate(triples):
            new_tag = _infer_tag(rhs_node, last_tag)
            if new_tag != per_triple_tag[i]:
                per_triple_tag[i] = new_tag
                changed = True
            last_tag[name] = new_tag

    assignments: dict[str, list[tuple[int, str | None]]] = {}
    for (cell_id, name, _), tag in zip(triples, per_triple_tag):
        assignments.setdefault(name, []).append((cell_id, tag))
    return assignments


def detect_type_changes(notebook_IR) -> Result:
    """
    Flags a finding for every plain-Assign definition site of a name
    whose tag disagrees with at least one *other* definition site of the
    same name, anywhere in the notebook - order-independent (see
    _compute_type_tags), so this no longer means "changed since the
    previous cell in ID order," it means "this name doesn't have one
    consistent type across all the places that define it." Both sides
    must be classified (non-None) to count as a disagreement, so a gap
    through unclassifiable code never gets silently treated as "no
    change."
    """
    result = Result()
    for name, sites in _compute_type_tags(notebook_IR).items():
        distinct_tags = {tag for _, tag in sites if tag is not None}
        if len(distinct_tags) < 2:
            continue
        for cell_id, tag in sites:
            if tag is None:
                continue
            other = next((c, t) for c, t in sites if t is not None and t != tag)
            line = _find_line(notebook_IR[cell_id].cell_code, name)
            result.add_path_result(PathResult(
                path=[other[0], cell_id],
                error_infos=[ErrorInfo(
                    cell_id=cell_id,
                    line=line,
                    label=name,
                    error_type="TYPE_CHANGE",
                    error_message=(
                        f"'{name}' is {tag} here, but {other[1]} in cell {other[0]} - "
                        f"inconsistent across the notebook's possible execution orders."
                    ),
                )],
            ))
    return result


def _compute_type_stability(notebook_IR) -> tuple[dict[str, str | None], set[str]]:
    """
    Per-name final tag and whether it's "risky" - it has no single
    consistent tag across every plain-Assign definition site (see
    _compute_type_tags), or it's bound some other way this pass can't
    classify at all (loop targets, tuple-unpacking, augmented assign,
    def/import...) - `defined_vars` is the ground truth for *that*
    (matching build_fixpoint_dependency_edges's own def_variables
    tracking). Once a name is risky it's risky everywhere: since
    build_pruned_dependency_edges/label_stable_dependencies source their
    edges from build_fixpoint_dependency_edges (order-independent - a
    name a cell reads may come from a definer earlier *or* later in ID
    order), a name that's inconsistent anywhere is a genuine risk for
    *any* reader of it, not just readers positioned after the
    inconsistency in cell-ID order.
    """
    assignments = _compute_type_tags(notebook_IR)
    all_defined_names: set[str] = set()
    for ir in notebook_IR.values():
        all_defined_names.update(ir.UDA.defined_vars)

    last_tag: dict[str, str | None] = {}
    risky: set[str] = set(all_defined_names - assignments.keys())

    for name, sites in assignments.items():
        tags = {tag for _, tag in sites}
        if None in tags or len(tags) != 1:
            risky.add(name)
        else:
            last_tag[name] = next(iter(tags))

    return last_tag, risky


def _shared_names(notebook_IR, target_cell: int, definer_cell: int) -> set[str]:
    """Names build_fixpoint_dependency_edges's (target, definer) edge is
    actually *about* - re-derived from the same UDA sets
    build_dependency_edges/build_fixpoint_dependency_edges already use,
    rather than needing dependency_analysis.py to expose per-name detail
    itself."""
    return set(notebook_IR[definer_cell].UDA.defined_vars) & notebook_IR[target_cell].UDA.unbound_final


def _is_backward_edge(target_cell: int, definer_cell: int) -> bool:
    """
    True when the definer comes *after* the reader in cell-ID order - an
    edge build_dependency_edges's naive single pass could never find, and
    that only exists here because build_fixpoint_dependency_edges doesn't
    assume ID order is execution order (see that module's docstring).
    That's exactly `order_dependent`'s bug shape - a real, unfixed gap on
    its own axis (definition timing), completely orthogonal to whether
    the value's *type* ever changes. A name can be perfectly
    type-stable (e.g. always a plain list, never reassigned) and still
    be the read-before-defined edge that makes the bug: type stability
    says nothing about whether the definition has actually happened yet.
    Treating a backward edge as automatically risky - regardless of what
    _compute_type_stability concludes - keeps build_pruned_dependency_edges
    from deleting it and label_stable_dependencies from mislabeling it
    "low-risk" right when it matters most.
    """
    return definer_cell > target_cell


def build_pruned_dependency_edges(notebook_IR) -> dict[int, set[int]]:
    """
    Same shape as context_builder.build_dependency_edges (cell_id -> set
    of cell_ids it depends on), but sourced from
    build_fixpoint_dependency_edges (order-independent - see that
    module's docstring) rather than build_dependency_edges's single
    ID-ordered pass, with confirmed-stable edges dropped: a definer cell
    is excluded from a reading cell's dependency set once this pass has
    positive evidence every name connecting them never changed type/shape
    anywhere in the notebook - by construction, not the kind of value
    this module's type-change bugs live in. See
    _compute_type_stability's docstring for the conservative "can't tell
    -> keep it" default that keeps this from ever hiding a real,
    unclassified dependency, and _is_backward_edge's docstring for why a
    read-before-defined edge is never eligible for pruning regardless of
    type stability.

    Standalone - not wired into context_builder.py's own
    build_dependency_edges. Built for the "does using type/shape info to
    *remove* boring dependency edges help, instead of adding type-change
    findings as extra context (see detect_type_changes above, and
    experiments.md finding 8)" experiment - callers pass this dict via
    DetectBugsEvent's dependency_edges override param.
    """
    fixpoint_edges = build_fixpoint_dependency_edges(notebook_IR)
    _, risky = _compute_type_stability(notebook_IR)

    edges: dict[int, set[int]] = {}
    for target_cell, definers in fixpoint_edges.items():
        edges[target_cell] = {
            definer_cell
            for definer_cell in definers
            if _is_backward_edge(target_cell, definer_cell)
            or not (names := _shared_names(notebook_IR, target_cell, definer_cell)) or (names & risky)
        }
    return edges


def label_stable_dependencies(notebook_IR, terse: bool = False) -> Result:
    """
    The alternative to build_pruned_dependency_edges's hard removal:
    instead of hiding confirmed-stable edges from the (fixpoint-sourced,
    order-independent) dependency graph, keep the full graph and
    *annotate* each stable edge as low-risk, leaving the actual weighting
    decision to the model rather than deciding it unilaterally
    beforehand. Same Result/ErrorInfo shape as detect_type_changes, fed
    in via DetectBugsEvent's extra_findings param (so it renders
    alongside the untouched, full dependency graph, not in place of it).

    terse controls only the wording of error_message, not which edges
    get labeled or any other behavior - an isolated, single-variable
    change for the "how sensitive is the model's output to a small,
    same-information wording change" experiment (see experiments.md's
    scratch notes): the default (verbose) sentence is ~25 words per
    label; terse=True carries the identical (name, tag, definer_cell)
    facts in ~6.
    """
    fixpoint_edges = build_fixpoint_dependency_edges(notebook_IR)
    last_tag, risky = _compute_type_stability(notebook_IR)

    result = Result()
    for target_cell, definers in fixpoint_edges.items():
        ir = notebook_IR[target_cell]
        for definer_cell in sorted(definers):
            if _is_backward_edge(target_cell, definer_cell):
                continue  # ordering risk - never label "low-risk" (see _is_backward_edge)
            names = _shared_names(notebook_IR, target_cell, definer_cell)
            for name in sorted(names - risky):
                tag = last_tag.get(name)
                line = _find_line(ir.cell_code, name)
                message = (
                    f"'{name}': stable {tag} since cell {definer_cell}."
                    if terse else
                    f"'{name}' has been a consistent {tag} since cell "
                    f"{definer_cell} with no type/shape change observed "
                    f"- a low-risk dependency, unlikely on its own to be "
                    f"a type/shape-change bug."
                )
                result.add_path_result(PathResult(
                    path=[definer_cell, target_cell],
                    error_infos=[ErrorInfo(
                        cell_id=target_cell,
                        line=line,
                        label=name,
                        error_type="STABLE_DEPENDENCY",
                        error_message=message,
                    )],
                ))
    return result
