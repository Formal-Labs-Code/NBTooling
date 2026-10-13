"""
Plain-Python checks repair.py runs on a patch, for mistakes the data-leak
analysis doesn't look at but that make a "fixed" notebook wrong anyway: reading
a name before any earlier cell defines it (the patched notebook would raise
NameError), and computing a preprocessing step's result without ever using it.

Both are approximations over the stdlib ast - names bound in any branch count
as defined, a name counts as used if it is read anywhere later in the code -
and repair.py only acts on what a patch *adds* relative to the original, so
quirks the original code already has are never blamed on the patch.

`cells` is always {cell index: source} in execution order; IPython magic and
shell lines are blanked before parsing.
"""
from __future__ import annotations

import ast
import builtins
import re

_MAGIC = re.compile(r"^[!%]")
# Names a notebook or script can read without defining them.
_PREDEFINED = set(dir(builtins)) | {"__file__", "get_ipython", "display", "In", "Out"}


def parse_cell(code: str) -> ast.Module:
    return ast.parse("\n".join("" if _MAGIC.match(line) else line for line in code.splitlines()))


def undefined_reads(cells: dict[int, str]) -> dict[str, int]:
    """Name -> first cell that reads it at module level before any earlier
    statement binds it. Function and class bodies are not checked (they run
    whenever they're called)."""
    defined, undefined = set(_PREDEFINED), {}
    for cell_id, code in cells.items():
        _scan(parse_cell(code).body, defined, undefined, cell_id)
    return undefined


def unused_results(cells: dict[int, str], methods: set[str]) -> set[str]:
    """Names assigned from a `<receiver>.<method>(...)` call, method in
    `methods`, that are never read after that assignment."""
    trees = [parse_cell(code) for code in cells.values()]
    assigned, loads = [], []
    for position, tree in enumerate(trees):
        for node in ast.walk(tree):
            if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                loads.append(((position, node.lineno, node.col_offset), node.id))
            if (isinstance(node, ast.Assign) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Attribute) and node.value.func.attr in methods):
                end = (position, node.end_lineno, node.end_col_offset)
                assigned.extend((end, name) for target in node.targets for name in _stored(target))
    return {name for end, name in assigned if not any(at > end and read == name for at, read in loads)}


def _scan(statements, defined, undefined, cell_id):
    """Walks statements in order, recording reads of names not yet bound,
    and binding names as it goes."""
    for statement in statements:
        if isinstance(statement, (ast.If, ast.While)):
            _read(statement.test, defined, undefined, cell_id)
            _scan(statement.body, defined, undefined, cell_id)
            _scan(statement.orelse, defined, undefined, cell_id)
        elif isinstance(statement, (ast.For, ast.AsyncFor)):
            _read(statement.iter, defined, undefined, cell_id)
            defined.update(_stored(statement.target))
            _scan(statement.body, defined, undefined, cell_id)
            _scan(statement.orelse, defined, undefined, cell_id)
        elif isinstance(statement, (ast.With, ast.AsyncWith)):
            for item in statement.items:
                _read(item.context_expr, defined, undefined, cell_id)
                if item.optional_vars is not None:
                    defined.update(_stored(item.optional_vars))
            _scan(statement.body, defined, undefined, cell_id)
        elif isinstance(statement, ast.Try):
            _scan(statement.body, defined, undefined, cell_id)
            for handler in statement.handlers:
                if handler.type is not None:
                    _read(handler.type, defined, undefined, cell_id)
                if handler.name:
                    defined.add(handler.name)
                _scan(handler.body, defined, undefined, cell_id)
            _scan(statement.orelse, defined, undefined, cell_id)
            _scan(statement.finalbody, defined, undefined, cell_id)
        else:
            names = _Names()
            names.visit(statement)
            for name in sorted(names.loads - defined):
                undefined.setdefault(name, cell_id)
            defined.update(names.binds)


def _read(expression, defined, undefined, cell_id):
    names = _Names()
    names.visit(expression)
    for name in sorted(names.loads - defined):
        undefined.setdefault(name, cell_id)
    defined.update(names.binds)  # walrus


def _stored(target) -> set[str]:
    return {node.id for node in ast.walk(target) if isinstance(node, ast.Name)}


class _Names(ast.NodeVisitor):
    """Names one simple statement reads and binds, not looking inside nested
    function/class/lambda bodies, minus comprehension variables."""

    def __init__(self):
        self.loads, self.binds, self._local = set(), set(), set()

    def visit(self, node):
        super().visit(node)
        self.loads -= self._local

    def visit_Name(self, node):
        (self.loads if isinstance(node.ctx, ast.Load) else self.binds).add(node.id)

    def visit_FunctionDef(self, node):
        for decorator in node.decorator_list:
            self.generic_visit_node(decorator)
        self.binds.add(node.name)

    visit_AsyncFunctionDef = visit_FunctionDef

    def visit_ClassDef(self, node):
        for expression in [*node.decorator_list, *node.bases]:
            self.generic_visit_node(expression)
        self.binds.add(node.name)

    def visit_Lambda(self, node):
        pass

    def visit_Import(self, node):
        self.binds.update(alias.asname or alias.name.split(".")[0] for alias in node.names)

    def visit_ImportFrom(self, node):
        self.binds.update(alias.asname or alias.name for alias in node.names if alias.name != "*")

    def visit_comprehension(self, node):
        self._local.update(_stored(node.target))
        self.generic_visit(node)

    def visit_Global(self, node):
        self.binds.update(node.names)

    def generic_visit_node(self, node):
        super().visit(node)
