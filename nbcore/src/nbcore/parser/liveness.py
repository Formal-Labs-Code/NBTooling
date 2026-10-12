"""
Flow-sensitive cell inputs: the names live at a cell's CFG entry, i.e. read
on some path before anything in the cell assigns them. Standard backward
liveness over the CFG cfg_builder already produces:

    live_in(n)  = uses(n) | (live_out(n) - defs(n))
    live_out(n) = union of live_in(s) for each successor s

Order, branches, loops, try edges and inlined same-cell calls all come from
the CFG itself; this module only says what each node reads and binds.
Node fields can't be used for that as-is (call nodes carry a pseudo lhs like
`X.fillna`, cond nodes carry no names, `with`/`try` nodes hold their whole
statement including the body), so uses/defs are taken from each node's own
AST fragment instead.

Known approximations, inherited from the CFG's shape:
  * Import/ImportFrom produce no CFG node (see cfg_builder.visit_Import), so
    imported names are not handled here - DefUseChains treats them as bound
    cell-wide, as before.
  * `except E as e` has no node of its own: `E` is read and `e` bound at the
    TryNode, which every handler is reachable from.
  * A `for` header binds its target on every path through it, including
    the loop-exit edge of an empty iterable.
  * def/class bodies are opaque single nodes; their free variables are
    handled by DefUseChains' scope rule, not here (an inlined call's body
    *is* in the CFG and is covered here, read at the call site).
"""

from . import ast_nodes as n
from .cfg_nodes import CondNode, RestoreNode, TryNode

_COMPREHENSION_TYPES = (n.ListComp, n.SetComp, n.GeneratorExp, n.DictComp)


def _reads(*exprs):
    """Load-context names in the given expressions, minus names bound by a
    comprehension inside them (mirrors cfg_builder._collect_names)."""
    reads, comp_bound = set(), set()
    for expr in exprs:
        if expr is None:
            continue
        for node in n.walk(expr):
            if isinstance(node, _COMPREHENSION_TYPES):
                for comp in node.generators:
                    comp_bound.update(x.id for x in n.walk(comp.target) if isinstance(x, n.Name))
            elif isinstance(node, n.Name) and isinstance(node.ctx, n.Load):
                reads.add(node.id)
    return reads - comp_bound


def _stores(*targets):
    return {
        x.id for t in targets if t is not None
        for x in n.walk(t) if isinstance(x, n.Name) and isinstance(x.ctx, n.Store)
    }


def _function_header_reads(stmt):
    """What a def/class statement evaluates where it appears: decorators,
    defaults, annotations / bases, keywords."""
    if isinstance(stmt, n.ClassDef):
        return _reads(*stmt.bases, *stmt.keywords, *stmt.decorator_list)
    args = stmt.args
    params = [*args.args, *args.kwonlyargs, *(a for a in (args.vararg, args.kwarg) if a)]
    return _reads(
        *stmt.decorator_list, *args.defaults, *(d for d in args.kw_defaults if d is not None),
        *(a.annotation for a in params), stmt.returns,
    )


def uses_defs(node):
    """(names read, names bound) by one CFG node."""
    if isinstance(node, RestoreNode):
        return set(node.right_hand_side), {node.left_hand_side}

    a = node.ast_node
    if a is None:
        return set(), set()

    if isinstance(node, TryNode):
        reads = _reads(*(h.type for h in a.handlers))
        return reads, {h.name for h in a.handlers if h.name}

    if isinstance(node, CondNode):
        if isinstance(a, n.For):
            return _reads(a.iter), _stores(a.target)
        return _reads(a.test), set()

    if isinstance(a, n.Assign):
        # Name targets are Store; Subscript/Attribute targets read their base.
        return _reads(a.value, *a.targets), _stores(*a.targets)
    if isinstance(a, n.AugAssign):
        target_reads = {a.target.id} if isinstance(a.target, n.Name) else set()
        return _reads(a.value, a.target) | target_reads, _stores(a.target)
    if isinstance(a, n.AnnAssign):
        binds = _stores(a.target) if a.value is not None else set()
        return _reads(a.value, a.annotation, a.target), binds
    if isinstance(a, n.With):
        return _reads(*(i.context_expr for i in a.items)), _stores(*(i.optional_vars for i in a.items))
    if isinstance(a, (n.FunctionDef, n.ClassDef)):
        return _function_header_reads(a), {a.name}
    if isinstance(a, n.Return):
        return _reads(a.value), set()
    if isinstance(a, n.Raise):
        return _reads(a.exc, a.cause), set()
    if isinstance(a, (n.Global, n.Pass, n.Break, n.Continue)):
        return set(), set()
    # Bare expressions: a Call (BBorBInode) or any other expression node.
    return _reads(a), set()


def live_at_entry(cfg):
    """Names read on some path through the cell before being bound."""
    nodes = cfg.nodes
    ud = {node: uses_defs(node) for node in nodes}
    live_in = {node: set() for node in nodes}
    changed = True
    while changed:
        changed = False
        for node in reversed(nodes):
            out = set().union(*(live_in[s] for s in node.outgoing)) if node.outgoing else set()
            uses, defs = ud[node]
            new = uses | (out - defs)
            if new != live_in[node]:
                live_in[node] = new
                changed = True
    return live_in[nodes[0]] if nodes else set()
