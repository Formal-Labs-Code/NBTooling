"""
Builds a CFG (see cfg_nodes.py) directly from our own AST (ast_nodes.py),
replacing simple_cfg's ast-module-based StmtVisitor/ExprVisitor.

Phase-1 scope (see parser/README.md for the tracked backlog):
  * Same-cell function-call inlining is implemented for a *reduced* case
    only: a call to a locally-defined function, with plain positional
    parameters/arguments only (no defaults/*args/**kwargs/keywords), whose
    body contains no calls to other locally-defined functions (which rules
    out recursion and nested inlining), and whose result -- if used --
    comes from a single trailing `return <expr>`. Anything outside that
    (see _check_inline_eligible) raises NotImplementedError naming exactly
    which condition failed, rejecting the whole cell rather than silently
    falling back to black-box or mis-inlining. Broadening this (defaults,
    kwargs, nested/recursive calls, multiple returns) is a tracked TODO,
    not an oversight -- see parser/README.md.
  * Every other call is treated as black-box (BBorBInode).
  * FunctionDef/ClassDef bodies are opaque at their *definition* site: a
    single node is emitted for the
    `def`/`class` statement itself, but its body is not spliced into the
    surrounding control flow (consistent with never inlining calls to it).
  * try/except models "entering the try body can jump to any handler" by
    connecting the TryNode itself to each handler, rather than simple_cfg's
    finer "any statement in the body can raise" edges -- an under-
    approximation, not an over-approximation.
  * the MUTATORS rewrite (`lst.append(x)` -> treated like `lst += ...`) is
    not implemented; mutating calls are visible as calls but don't yet
    propagate a taint/impact edge onto their receiver.
"""

import copy

from . import ast_nodes as n
from .cfg_nodes import (
    AssignmentCallNode,
    AssignmentNode,
    BBorBInode,
    BreakNode,
    CondNode,
    EntryOrExitNode,
    Node,
    RaiseNode,
    RestoreNode,
    ReturnNode,
    TryNode,
)
from .unparse import unparse


class CFG:
    def __init__(self, nodes, blackbox_assignments, filename, inline_names=None):
        self.nodes = nodes
        self.blackbox_assignments = blackbox_assignments
        self.filename = filename
        # mangled name -> (original name, inlined function's name), for every
        # name _inline_call renamed; lets analyses report user-facing names.
        self.inline_names = inline_names if inline_names is not None else {}

    def __repr__(self):
        return "".join(f"Node: {i} {n!r}\n\n" for i, n in enumerate(self.nodes))

    def __str__(self):
        return "".join(f"Node: {i} {n}\n\n" for i, n in enumerate(self.nodes))


class BlockResult:
    __slots__ = ("first", "lasts", "breaks", "continues")

    def __init__(self, first, lasts, breaks=None, continues=None):
        self.first = first
        self.lasts = lasts
        self.breaks = breaks or []
        self.continues = continues or []


_AUG_SYMBOLS = {
    n.Add: "+", n.Sub: "-", n.Mult: "*", n.Div: "/", n.FloorDiv: "//",
    n.Mod: "%", n.Pow: "**", n.BitAnd: "&", n.BitOr: "|", n.BitXor: "^",
    n.RShift: ">>", n.LShift: "<<", n.MatMult: "@",
}


_COMPREHENSION_TYPES = (n.ListComp, n.SetComp, n.GeneratorExp, n.DictComp)


def _collect_names(expr_node):
    if expr_node is None:
        return []
    # Names bound by a comprehension's own target (e.g. `x` in
    # `[x*x for x in z]`) must not be reported as "used" - not just the
    # Store-context target occurrence itself, but every Load-context
    # reference to it inside the comprehension body too (`x*x` reads `x`
    # twice). A ctx==Load filter alone doesn't catch this: those reads are
    # genuinely Load-context, just not *free* variables. So this excludes
    # by name identity instead, cell-wide, consistent with def_use.py's
    # own deliberate scope-insensitivity (see its module docstring) rather
    # than trying to scope the exclusion to just inside that comprehension.
    bound_by_comprehensions = set()
    for node in n.walk(expr_node):
        if isinstance(node, _COMPREHENSION_TYPES):
            for comp in node.generators:
                bound_by_comprehensions.update(
                    x.id for x in n.walk(comp.target) if isinstance(x, n.Name)
                )
    return [
        x.id for x in n.walk(expr_node)
        if isinstance(x, n.Name) and x.id not in bound_by_comprehensions
    ]


def _extract_lhs(target):
    if isinstance(target, n.Name):
        return target.id
    if isinstance(target, (n.Attribute, n.Subscript)):
        return _extract_lhs(target.value)
    return unparse(target)


class CFGBuilder:
    def __init__(self, filename):
        self.nodes = []
        self.filename = filename
        self.blackbox_assignments = set()
        self.local_functions = {}  # name -> FunctionDef, populated as defs are visited (source order)
        self._inline_counter = 0
        self.inline_names = {}

    def build(self, module):
        entry = self._append(EntryOrExitNode("Entry module"))
        result = self.visit_stmts(module.body)
        exit_node = self._append(EntryOrExitNode("Exit module"))
        if result.first is not None:
            entry.connect(result.first)
            exit_node.connect_predecessors(result.lasts)
        else:
            entry.connect(exit_node)
        return CFG(self.nodes, self.blackbox_assignments, self.filename, self.inline_names)

    def _append(self, node):
        self.nodes.append(node)
        return node

    @staticmethod
    def _splice(predecessors, body):
        """Connect `predecessors` into `body`, or -- if `body` is an
        all-ignored suite (e.g. `if x:\n    import os\n`) with no real
        node -- treat `predecessors` as flowing straight through untouched."""
        if body.first is None:
            return predecessors, body.breaks, body.continues
        for p in predecessors:
            p.connect(body.first)
        return body.lasts, body.breaks, body.continues

    @staticmethod
    def _chain(items):
        """Sequence a list of items (each a Node, a BlockResult, or None to
        skip) exactly like visit_stmts chains statements. Used to assemble
        an inlined call's argument-binding nodes + spliced body + return
        hand-off into one BlockResult."""
        first = None
        lasts = []
        have_prev = False
        breaks, continues = [], []
        for item in items:
            if item is None:
                continue
            if isinstance(item, BlockResult):
                item_first, item_lasts = item.first, item.lasts
                breaks.extend(item.breaks)
                continues.extend(item.continues)
            else:
                item_first, item_lasts = item, [item]
            if item_first is None:
                continue
            if first is None:
                first = item_first
            if have_prev:
                for node in lasts:
                    node.connect(item_first)
            lasts = item_lasts
            have_prev = True
        return first, (lasts if have_prev else []), breaks, continues

    def visit(self, stmt):
        method = getattr(self, "visit_" + type(stmt).__name__, None)
        if method is None:
            raise NotImplementedError(
                f"CFG builder has no handler for statement type '{type(stmt).__name__}' "
                "(tracked backlog item, see cfg_builder.py docstring)"
            )
        return method(stmt)

    def visit_stmts(self, stmts):
        # A statement visitor may return None for statements that
        # contribute no CFG node at all (currently just Import/ImportFrom,
        # matching simple_cfg's IgnoredNode) -- these are transparent:
        # skipped without breaking the predecessor -> successor chain.
        first = None
        prev_lasts = []
        have_prev = False
        breaks, continues = [], []
        for stmt in stmts:
            result = self.visit(stmt)
            if result is None:
                continue
            if first is None:
                first = result.first
            if have_prev:
                for node in prev_lasts:
                    node.connect(result.first)
            prev_lasts = result.lasts
            have_prev = True
            breaks.extend(result.breaks)
            continues.extend(result.continues)
        return BlockResult(first, prev_lasts if have_prev else [], breaks, continues)

    # -- simple/linear statements -----------------------------------------

    def visit_Assign(self, stmt):
        rhs_names = _collect_names(stmt.value)
        targets = stmt.targets
        flat_targets = []
        for target in targets:
            if isinstance(target, (n.Tuple, n.List)):
                flat_targets.extend(target.elts)
            else:
                flat_targets.append(target)

        if isinstance(stmt.value, n.Call) and self._is_local_call(stmt.value):
            if len(flat_targets) != 1:
                raise NotImplementedError(
                    f"inlining '{stmt.value.func.id}': tuple/multi-target unpacking of an "
                    "inlined call's result is not supported yet -- TODO, see parser/README.md"
                )
            return self._inline_call(_extract_lhs(flat_targets[0]), stmt, stmt.value)

        if isinstance(stmt.value, n.Call):
            # Every target gets its own black-box-call + assignment pair
            # (re-visiting the same Call ast_node once per target), matching
            # simple_cfg's assignment_call_node behavior for e.g.
            # `a, b = f(...)` -- duplicative but keeps each unpacked name
            # individually visible as "assigned from a call" to analyses
            # like find_necessary_cells that scan for BBorBInode.
            results = [self._make_assignment_call(_extract_lhs(t), stmt, stmt.value) for t in flat_targets]
            for a, b in zip(results, results[1:]):
                for last in a.lasts:
                    last.connect(b.first)
            return BlockResult(results[0].first, results[-1].lasts, [], [])

        made = []
        for target in flat_targets:
            lhs = _extract_lhs(target)
            label = f"{unparse(target)} = {unparse(stmt.value)}"
            node = AssignmentNode(label, lhs, stmt, rhs_names, path=self.filename)
            self._append(node)
            made.append(node)
        for a, b in zip(made, made[1:]):
            a.connect(b)
        return BlockResult(made[0], [made[-1]], [], [])

    def _make_call_node(self, call_node):
        func_name = unparse(call_node.func)
        rhs_names = _collect_names(call_node)
        label = unparse(call_node)
        node = BBorBInode(
            label, func_name, call_node, rhs_names,
            line_number=call_node.lineno, path=self.filename, func_name=func_name,
        )
        node.args = list(call_node.args)
        self._append(node)
        self.blackbox_assignments.add(node)
        return node

    def _make_assignment_call(self, lhs, assign_stmt, call_node):
        call = self._make_call_node(call_node)
        label = f"{lhs} = {call.label}"
        assign_node = AssignmentCallNode(
            label, lhs, assign_stmt, [call.left_hand_side],
            line_number=assign_stmt.lineno, path=self.filename, call_node=call,
        )
        self._append(assign_node)
        call.connect(assign_node)
        return BlockResult(call, [assign_node], [], [])

    def visit_AugAssign(self, stmt):
        lhs = _extract_lhs(stmt.target)
        rhs_names = _collect_names(stmt.value) + [lhs]
        symbol = _AUG_SYMBOLS.get(type(stmt.op), "?")
        label = f"{unparse(stmt.target)} {symbol}= {unparse(stmt.value)}"
        node = AssignmentNode(label, lhs, stmt, rhs_names, path=self.filename)
        self._append(node)
        return BlockResult(node, [node], [], [])

    def visit_AnnAssign(self, stmt):
        lhs = _extract_lhs(stmt.target)
        rhs_names = _collect_names(stmt.value)
        label = f"{unparse(stmt.target)}: {unparse(stmt.annotation)}"
        if stmt.value is not None:
            label += f" = {unparse(stmt.value)}"
        node = AssignmentNode(label, lhs, stmt, rhs_names, path=self.filename)
        self._append(node)
        return BlockResult(node, [node], [], [])

    def visit_Expr(self, stmt):
        if isinstance(stmt.value, n.Call) and self._is_local_call(stmt.value):
            return self._inline_call(None, stmt, stmt.value)
        if isinstance(stmt.value, n.Call):
            node = self._make_call_node(stmt.value)
        else:
            # simple_cfg's visit_Expr delegates straight to visit(node.value)
            # rather than wrapping in the Expr statement -- so e.g. a bare
            # `z` reference gets a node whose ast_node is the Name `z`
            # itself. stale_cell_analysis.F_transformer specifically checks
            # `isinstance(cfg_node.ast_node, ast.Name)` to catch exactly
            # this "cell just re-reads a stale variable" pattern.
            node = Node(unparse(stmt.value), stmt.value, path=self.filename)
            self._append(node)
        return BlockResult(node, [node], [], [])

    def _leaf(self, label, stmt):
        node = Node(label, stmt, path=self.filename)
        self._append(node)
        return BlockResult(node, [node], [], [])

    def visit_Import(self, stmt):
        # Matches simple_cfg's visit_Import: imports contribute no CFG node
        # at all (IgnoredNode there, None here -- see visit_stmts). This
        # matters beyond tidiness: the abstract-interpretation fixpoint in
        # Runner.intra_fixpoint_runner stops propagating past a node whose
        # transformed state == its (fresh, empty) previous state, so a real
        # no-op node placed first in a cell (e.g. a leading import block)
        # would silently strand every later statement as unreachable.
        return None

    def visit_ImportFrom(self, stmt):
        return None

    def visit_Global(self, stmt):
        return self._leaf("global " + ", ".join(stmt.names), stmt)

    def visit_Pass(self, stmt):
        return self._leaf("pass", stmt)

    def visit_FunctionDef(self, stmt):
        params = ", ".join(a.arg for a in stmt.args.args)
        result = self._leaf(f"def {stmt.name}({params}):", stmt)
        # Registered *after* building this leaf, in source order -- a call
        # appearing textually before its own def (a NameError at runtime)
        # is correctly not treated as resolving to it.
        self.local_functions[stmt.name] = stmt
        return result

    def visit_ClassDef(self, stmt):
        bases = ", ".join(unparse(b) for b in stmt.bases)
        return self._leaf(f"class {stmt.name}({bases}):" if bases else f"class {stmt.name}:", stmt)

    def visit_Return(self, stmt):
        rhs = _collect_names(stmt.value)
        label = "return " + unparse(stmt.value) if stmt.value is not None else "return"
        node = ReturnNode(label, None, stmt, rhs, path=self.filename)
        self._append(node)
        return BlockResult(node, [node], [], [])

    def visit_Raise(self, stmt):
        label = "raise"
        if stmt.exc is not None:
            label += " " + unparse(stmt.exc)
        if stmt.cause is not None:
            label += " from " + unparse(stmt.cause)
        node = RaiseNode(stmt, label=label, path=self.filename)
        self._append(node)
        return BlockResult(node, [node], [], [])

    def visit_Break(self, stmt):
        node = BreakNode(stmt, path=self.filename)
        self._append(node)
        return BlockResult(node, [], [node], [])

    def visit_Continue(self, stmt):
        node = Node("Continue", stmt, path=self.filename)
        self._append(node)
        return BlockResult(node, [], [], [node])

    # -- with (no branching; target/body spliced in sequence) -------------

    def visit_With(self, stmt):
        heads = []
        for item in stmt.items:
            rhs = _collect_names(item.context_expr)
            if item.optional_vars is not None:
                lhs = _extract_lhs(item.optional_vars)
                label = f"{unparse(item.optional_vars)} = {unparse(item.context_expr)}"
            else:
                lhs = None
                label = unparse(item.context_expr)
            node = AssignmentNode(label, lhs, stmt, rhs, path=self.filename)
            self._append(node)
            heads.append(node)
        for a, b in zip(heads, heads[1:]):
            a.connect(b)
        body = self.visit_stmts(stmt.body)
        lasts, breaks, continues = self._splice([heads[-1]], body)
        return BlockResult(heads[0], lasts, breaks, continues)

    # -- branching control flow -------------------------------------------

    def visit_If(self, stmt):
        cond = self._append(CondNode(stmt.test, stmt, label="cond " + unparse(stmt.test) + ":", path=self.filename))
        body = self.visit_stmts(stmt.body)
        lasts, breaks, continues = self._splice([cond], body)
        cond.positive_nodes.append(body.first if body.first is not None else cond)
        lasts, breaks, continues = list(lasts), list(breaks), list(continues)
        if stmt.orelse:
            orelse = self.visit_stmts(stmt.orelse)
            orelse_lasts, orelse_breaks, orelse_continues = self._splice([cond], orelse)
            lasts.extend(orelse_lasts)
            breaks.extend(orelse_breaks)
            continues.extend(orelse_continues)
        else:
            lasts.append(cond)
        return BlockResult(cond, lasts, breaks, continues)

    def _loop(self, header, body_stmts, orelse_stmts):
        body = self.visit_stmts(body_stmts)
        body_lasts, body_breaks, body_continues = self._splice([header], body)
        header.positive_nodes.append(body.first if body.first is not None else header)
        for last in body_lasts:
            last.connect(header)  # back-edge
        for cont in body_continues:
            cont.connect(header)
        lasts = list(body_breaks)
        if orelse_stmts:
            orelse = self.visit_stmts(orelse_stmts)
            orelse_lasts, _, _ = self._splice([header], orelse)
            lasts.extend(orelse_lasts)
        else:
            lasts.append(header)
        return BlockResult(header, lasts, [], [])

    def visit_While(self, stmt):
        header = self._append(CondNode(stmt.test, stmt, label="cond " + unparse(stmt.test) + ":", path=self.filename))
        return self._loop(header, stmt.body, stmt.orelse)

    def visit_For(self, stmt):
        label = f"for {unparse(stmt.target)} in {unparse(stmt.iter)}:"
        header = self._append(CondNode(stmt.iter, stmt, label=label, path=self.filename))
        return self._loop(header, stmt.body, stmt.orelse)

    def visit_Try(self, stmt):
        try_node = self._append(TryNode(stmt, path=self.filename))
        body = self.visit_stmts(stmt.body)
        body_lasts, breaks, continues = self._splice([try_node], body)
        breaks, continues = list(breaks), list(continues)

        lasts = []
        for handler in stmt.handlers:
            h_result = self.visit_stmts(handler.body)
            h_lasts, h_breaks, h_continues = self._splice([try_node], h_result)
            lasts.extend(h_lasts)
            breaks.extend(h_breaks)
            continues.extend(h_continues)

        if stmt.orelse:
            orelse = self.visit_stmts(stmt.orelse)
            orelse_lasts, orelse_breaks, orelse_continues = self._splice(body_lasts, orelse)
            lasts.extend(orelse_lasts)
            breaks.extend(orelse_breaks)
            continues.extend(orelse_continues)
        else:
            lasts.extend(body_lasts)

        if stmt.finalbody:
            final = self.visit_stmts(stmt.finalbody)
            final_lasts, final_breaks, final_continues = self._splice(lasts, final)
            lasts = final_lasts
            breaks.extend(final_breaks)
            continues.extend(final_continues)

        return BlockResult(try_node, lasts, breaks, continues)

    # -- reduced-scope same-cell function-call inlining --------------------
    #
    # See the module docstring for exactly what's supported. The mechanism:
    #   1. deep-copy the callee's body so each call site gets its own copy
    #      (a function called twice must not share CFG nodes/state keys);
    #   2. rename every name the function binds internally (its parameters
    #      and any Assign/For/... targets in its body) to a call-site-unique
    #      name, since the CFG is a single flat namespace -- unrenamed
    #      params/locals would collide with the caller's own variables or
    #      with another call to the same function;
    #   3. bind each (renamed) parameter to its argument expression via a
    #      normal AssignmentNode wrapping a synthetic `Assign` node, so the
    #      existing analyses (which pattern-match on `.ast_node.value`) see
    #      an ordinary assignment;
    #   4. splice the (renamed) body in with visit_stmts;
    #   5. if the result is used, the trailing `return <expr>` becomes one
    #      more synthetic assignment (`~..._return__ = <expr>`), followed by
    #      a RestoreNode copying that into the real target -- RestoreNode is
    #      exactly the primitive simple_cfg already used for this "transfer
    #      state between two existing keys" step (see dataleak_analysis's
    #      F_transformer, which special-cases RestoreNode generically at the
    #      very top, unchanged by any of this).

    def _is_local_call(self, call_node):
        return isinstance(call_node.func, n.Name) and call_node.func.id in self.local_functions

    def _collect_locally_bound(self, stmts, params):
        bound = set(params)
        for stmt in stmts:
            for node in n.walk(stmt):
                if isinstance(node, n.Name) and isinstance(node.ctx, n.Store):
                    bound.add(node.id)
                elif isinstance(node, (n.FunctionDef, n.ClassDef)):
                    bound.add(node.name)
                elif isinstance(node, (n.Import, n.ImportFrom)):
                    for alias in node.names:
                        if alias.name != "*":
                            bound.add(alias.asname or alias.name.split(".")[0])
                elif isinstance(node, n.ExceptHandler) and node.name:
                    bound.add(node.name)
        return bound

    def _check_inline_eligible(self, definition, call_node, *, needs_return):
        name = definition.name
        args = definition.args
        if args.vararg or args.kwarg or args.kwonlyargs or args.defaults:
            raise NotImplementedError(
                f"inlining '{name}': only plain positional parameters are supported "
                "(no *args/**kwargs/defaults/keyword-only params yet) -- TODO, "
                "see parser/README.md"
            )
        if call_node.keywords or any(isinstance(a, n.Starred) for a in call_node.args):
            raise NotImplementedError(
                f"inlining '{name}': only plain positional call arguments are supported "
                "(no keyword args or */** unpacking at the call site yet) -- TODO"
            )
        if len(call_node.args) != len(args.args):
            raise NotImplementedError(
                f"inlining '{name}': call passes {len(call_node.args)} argument(s) but "
                f"the function takes {len(args.args)} -- TODO (no defaults support yet)"
            )
        for stmt in definition.body:
            for node in n.walk(stmt):
                if node is not call_node and isinstance(node, n.Call) \
                        and isinstance(node.func, n.Name) and node.func.id in self.local_functions:
                    raise NotImplementedError(
                        f"inlining '{name}': its body calls another locally-defined "
                        f"function ('{node.func.id}') -- nested inlining and recursion "
                        "are not supported yet -- TODO"
                    )
        body = definition.body
        trailing_is_return = isinstance(body[-1], n.Return)
        trailing_return_ok = trailing_is_return and body[-1].value is not None
        other_returns = any(
            isinstance(node, n.Return) and node is not body[-1]
            for stmt in body for node in n.walk(stmt)
        )
        if other_returns or (needs_return and not trailing_return_ok):
            raise NotImplementedError(
                f"inlining '{name}': only a single 'return <expr>' as the function's "
                "very last statement is supported yet -- TODO"
            )

    def _inline_call(self, lhs, containing_stmt, call_node):
        definition = self.local_functions[call_node.func.id]
        self._check_inline_eligible(definition, call_node, needs_return=lhs is not None)

        call_index = self._inline_counter
        self._inline_counter += 1
        prefix = f"~inline_{call_index}_{definition.name}_"

        params = [a.arg for a in definition.args.args]
        body = copy.deepcopy(definition.body)
        return_stmt = body.pop() if body and isinstance(body[-1], n.Return) else None

        rename_scope = body + ([return_stmt] if return_stmt else [])
        bound = self._collect_locally_bound(rename_scope, params)
        rename_map = {name: f"{prefix}{name}" for name in bound}
        for name, mangled in rename_map.items():
            self.inline_names[mangled] = (name, definition.name)
        for stmt in rename_scope:
            for node in n.walk(stmt):
                if isinstance(node, n.Name) and node.id in rename_map:
                    node.id = rename_map[node.id]

        items = []
        for param, arg_expr in zip(params, call_node.args):
            renamed_param = rename_map[param]
            synthetic = n.Assign(
                targets=[n.Name(id=renamed_param, ctx=n.Store())],
                value=arg_expr, lineno=call_node.lineno,
            )
            node = AssignmentNode(
                f"{renamed_param} = {unparse(arg_expr)}", renamed_param, synthetic,
                _collect_names(arg_expr), path=self.filename,
            )
            self._append(node)
            items.append(node)

        items.append(self.visit_stmts(body))

        if return_stmt is not None:
            return_temp = f"{prefix}__return__"
            self.inline_names[return_temp] = (f"{definition.name}()", definition.name)
            synthetic = n.Assign(
                targets=[n.Name(id=return_temp, ctx=n.Store())],
                value=return_stmt.value, lineno=return_stmt.lineno,
            )
            ret_node = AssignmentNode(
                f"{return_temp} = {unparse(return_stmt.value)}", return_temp, synthetic,
                _collect_names(return_stmt.value), path=self.filename,
            )
            self._append(ret_node)
            items.append(ret_node)

            if lhs is not None:
                restore_node = RestoreNode(
                    f"{lhs} = {return_temp}", lhs, [return_temp],
                    line_number=containing_stmt.lineno, path=self.filename,
                )
                self._append(restore_node)
                items.append(restore_node)

        first, lasts, breaks, continues = self._chain(items)
        if first is None:
            raise NotImplementedError(f"inlining '{definition.name}': empty function body not supported")
        return BlockResult(first, lasts, breaks, continues)


def get_cfg(module, filename):
    return CFGBuilder(filename).build(module)
