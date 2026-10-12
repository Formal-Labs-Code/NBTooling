"""
Replacement for beniget.DefUseChains + IR/visitors/assigns_visitor.py's
AssignsVisitor, ported onto our own AST.

`DefUseChains` computes a cell's inputs flow-sensitively: the names live at
the entry of the cell's CFG, i.e. read on some path before the cell binds
them (see liveness.py) - so `X = X.fillna(0)` reads an earlier `X`, and a
name bound on only one branch of an `if` is still an input.

Three things the CFG doesn't carry are handled here, cell-wide and
order-insensitively instead:
  * imports (no CFG node) bind their names everywhere in the cell;
  * names a function assigns under `global` count as bound by the cell;
  * def/class bodies are opaque CFG nodes, so their free variables are
    inputs unless the cell binds them somewhere. A body is its own scope:
    its parameters and locals never bind at cell level, and decorators,
    defaults, annotations and bases are read where the def/class appears.
Comprehensions are flat: their loop variables don't count as reads.

What actually gets consumed downstream is `unbound_names` -- identifiers
used in this cell that resolve to *neither* a binding in this cell nor a
Python builtin, i.e. that must come from elsewhere (a previous cell, an
import).
"""

import builtins as _builtins_module

from . import ast_nodes as n
from .liveness import live_at_entry

# beniget resolves names against Python's builtins scope, so e.g. `print` in
# `print(x)` is never "unbound" even though it's never assigned in the cell.
# Mirror that here rather than relying solely on the funcs-subtraction that
# downstream analyses apply to `unbound_final` -- IsolatedCellAnalysis and
# IdleCellAnalysis read `unbound_names` directly, with no such subtraction.
_BUILTIN_NAMES = frozenset(dir(_builtins_module))


class DefUseChains:
    def __init__(self):
        self.unbound_names = set()

    def visit(self, module, cfg=None):
        if cfg is None:
            from .cfg_builder import get_cfg
            cfg = get_cfg(module, None)
        bound, body_free = set(), set()
        self._imported, self._global_bound = set(), set()
        self._collect([module], bound, set(), set(), bound, nested_into=body_free)
        live = {name for name in live_at_entry(cfg) if not name.startswith("~")}
        self.unbound_names = (
            (live - self._imported - self._global_bound) | (body_free - bound)
        ) - _BUILTIN_NAMES
        return self

    def _collect(self, roots, bound, used, declared_global, cell_bound, nested_into=None):
        """Bindings/reads of one scope into `bound`/`used`, without
        descending into nested function or class bodies (see _nested).
        At cell level, nested bodies' free variables go to `nested_into`."""
        stack = list(roots)
        while stack:
            node = stack.pop()
            if isinstance(node, n.FunctionDef):
                bound.add(node.name)
                args = node.args
                stack.extend(node.decorator_list)
                stack.extend(args.defaults)
                stack.extend(d for d in args.kw_defaults if d is not None)
                params = [*args.args, *args.kwonlyargs, *(a for a in (args.vararg, args.kwarg) if a)]
                stack.extend(a.annotation for a in params if a.annotation is not None)
                if node.returns is not None:
                    stack.append(node.returns)
                self._nested(node.body, {a.arg for a in params}, used if nested_into is None else nested_into, cell_bound)
                continue
            if isinstance(node, n.ClassDef):
                bound.add(node.name)
                stack.extend(node.bases)
                stack.extend(node.keywords)
                stack.extend(node.decorator_list)
                self._nested(node.body, set(), used if nested_into is None else nested_into, cell_bound)
                continue
            if isinstance(node, n.Name):
                if isinstance(node.ctx, n.Store):
                    bound.add(node.id)
                elif isinstance(node.ctx, n.Load):
                    used.add(node.id)
            elif isinstance(node, n.Global):
                declared_global.update(node.names)
            elif isinstance(node, (n.Import, n.ImportFrom)):
                for alias in node.names:
                    if alias.name != "*":
                        bound.add(alias.asname or alias.name.split(".")[0])
                        if bound is cell_bound:
                            self._imported.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, n.ExceptHandler):
                if node.name:
                    bound.add(node.name)
            stack.extend(n.iter_child_nodes(node))

    def _nested(self, body, params, enclosing_used, cell_bound):
        """A function/class body as its own scope: its free variables become
        reads of the enclosing scope, and `global` names it assigns become
        cell-level bindings."""
        bound, used, declared_global = set(params), set(), set()
        self._collect(body, bound, used, declared_global, cell_bound)
        cell_bound.update(bound & declared_global)
        self._global_bound.update(bound & declared_global)
        enclosing_used.update(used - (bound - declared_global))


def _call_func_name(value):
    if isinstance(value, n.Call) and isinstance(value.func, n.Name):
        return value.func.id
    return None


class AssignsVisitor(n.NodeVisitor):
    """Ported 1:1 from the original's behavior: it's a NodeVisitor whose
    visit_Import/visit_ImportFrom/visit_Expr/visit_Assign methods don't call
    generic_visit, so they don't descend into their own children, but every
    *other* node type (If/For/While/Try/FunctionDef/...) falls through to
    generic_visit and keeps recursing -- so nested Assigns inside a branch
    or loop are still picked up, only the four handled statement types stop
    the descent at themselves."""

    def __init__(self, def_use_chains):
        self.def_use_chains = def_use_chains
        self.funcs = set()
        self.imports = set()
        self.defined_vars = {}

    def visit_Import(self, node):
        for alias in node.names:
            self.imports.add(alias.asname or alias.name)

    def visit_ImportFrom(self, node):
        for alias in node.names:
            self.imports.add(alias.asname or alias.name)

    def visit_Expr(self, node):
        fn = _call_func_name(node.value)
        if fn:
            self.funcs.add(fn)

    def visit_Assign(self, node):
        value = node.value
        if isinstance(value, n.BinOp):
            for fn in (_call_func_name(value.left), _call_func_name(value.right)):
                if fn:
                    self.funcs.add(fn)
        else:
            fn = _call_func_name(value)
            if fn:
                self.funcs.add(fn)
        for target in node.targets:
            self._record_target(target, node.lineno)

    def _record_target(self, target, lineno):
        if isinstance(target, n.Name) and isinstance(target.ctx, n.Store):
            self.defined_vars[target.id] = lineno
        elif isinstance(target, (n.List, n.Tuple)):
            for element in target.elts:
                for sub in n.walk(element):
                    if isinstance(sub, n.Name) and isinstance(sub.ctx, n.Store):
                        self.defined_vars[sub.id] = lineno
        elif isinstance(target, (n.Subscript, n.Attribute)):
            base = target.value
            if isinstance(base, n.Name):
                self.defined_vars[base.id] = lineno
            else:
                for sub in n.walk(base):
                    if isinstance(sub, n.Name):
                        self.defined_vars[sub.id] = lineno

    def combine(self):
        self.unbound_final = self.def_use_chains.unbound_names - (self.funcs | self.imports)
