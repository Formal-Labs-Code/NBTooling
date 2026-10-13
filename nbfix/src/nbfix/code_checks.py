"""
Plain-Python checks repair.py runs on a patch, for mistakes the data-leak
analysis doesn't look at but that make a "fixed" notebook wrong anyway: reading
a name before any earlier cell defines it (the patched notebook would raise
NameError), computing a preprocessing step's result without ever using it, and
fitting something on test data, and training or scoring on features and labels
from different splits.

All are approximations over the stdlib ast - names bound in any branch count
as defined, a name counts as used if it is read anywhere later in the code,
test data is tracked by name - and repair.py only acts on what a patch *adds*
relative to the original, so quirks the original code already has are never
blamed on the patch.

`cells` is always {cell index: source} in execution order; IPython magic and
shell lines are blanked before parsing.
"""
from __future__ import annotations

import ast
import builtins
import re

_MAGIC = re.compile(r"^[!%]")
_FIT_METHODS = {"fit", "fit_transform", "fit_resample"}
# Names a notebook or script can read without defining them.
_PREDEFINED = set(dir(builtins)) | {"__file__", "get_ipython", "display", "In", "Out"}


def parse_cell(code: str) -> ast.Module:
    return ast.parse("\n".join("" if _MAGIC.match(line) else line for line in code.splitlines()))


def first_binding(cells: dict[int, str], name: str, after: int = -1) -> int | None:
    """The first cell after `after` that binds name at module level, if any."""
    for cell_id, code in cells.items():
        if cell_id <= after:
            continue
        names = _Names()
        for statement in parse_cell(code).body:
            names.visit(statement)
        if name in names.binds:
            return cell_id
    return None


def undefined_reads(cells: dict[int, str]) -> dict[str, int]:
    """Name -> first cell that reads it before it can be defined: at module
    level, before any earlier statement binds it; inside a function (which
    runs whenever it's called), when it's neither a parameter nor bound in
    that function nor bound at module level anywhere."""
    defined, undefined = set(_PREDEFINED), {}
    trees = {cell_id: parse_cell(code) for cell_id, code in cells.items()}
    for cell_id, tree in trees.items():
        _scan(tree.body, defined, undefined, cell_id)
    for cell_id, tree in trees.items():
        for function in (node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
            for name in sorted(_function_reads(function) - defined):
                undefined.setdefault(name, cell_id)
    return undefined


def _function_reads(function) -> set[str]:
    """Names a function reads that it doesn't bind itself (parameters, any
    assignment in its body, nested definitions, imports) - flow-insensitively,
    so only never-bound names, not reads before assignment. Nested functions
    are checked on their own, with the enclosing function's names visible."""
    args = function.args
    bound = {arg.arg for arg in [*args.posonlyargs, *args.args, *args.kwonlyargs]}
    bound |= {arg.arg for arg in (args.vararg, args.kwarg) if arg}
    reads = set()
    for node in ast.walk(function):
        if isinstance(node, ast.Name):
            (reads if isinstance(node.ctx, ast.Load) else bound).add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)) and node is not function:
            bound.add(node.name)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            bound |= {alias.asname or alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.arg):
            bound.add(node.arg)  # lambda and nested-function parameters
        elif isinstance(node, ast.ExceptHandler) and node.name:
            bound.add(node.name)
    return reads - bound


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


def fits_on_test_data(cells: dict[int, str]) -> set[tuple[str, str]]:
    """(call, test variable) for each fit/fit_transform/fit_resample whose
    data comes from the test half of a train_test_split - directly
    (`scaler.fit(X_test)`), or through a local function that fits on that
    argument (`preprocess(X_test)` where preprocess calls fit_transform on
    it). Test data is tracked by name, flow-insensitively: the odd-position
    targets of a train_test_split, and anything assigned from them."""
    trees = [parse_cell(code) for code in cells.values()]
    nodes = [node for tree in trees for node in ast.walk(tree)]
    assigns = [node for node in nodes if isinstance(node, ast.Assign)]

    test = set()
    for assign in assigns:
        if _call_name(assign.value) == "train_test_split" and isinstance(assign.targets[0], ast.Tuple):
            test |= {elt.id for elt in assign.targets[0].elts[1::2] if isinstance(elt, ast.Name)}
    test = _derived(test, assigns)

    # Local functions that fit on one of their parameters: name -> {position}.
    fitting_params = {}
    for function in (node for node in nodes if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))):
        body = [node for node in ast.walk(function) if node is not function]
        params = [arg.arg for arg in function.args.args]
        for position, param in enumerate(params):
            reached = _derived({param}, [node for node in body if isinstance(node, ast.Assign)])
            if any(_fit_argument(node) in reached for node in body):
                fitting_params.setdefault(function.name, set()).add(position)

    found = set()
    for node in nodes:
        argument = _fit_argument(node)
        if argument in test:
            found.add((ast.unparse(node.func), argument))
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            for position in fitting_params.get(node.func.id, ()):
                if position < len(node.args) and isinstance(node.args[position], ast.Name) \
                        and node.args[position].id in test:
                    found.add((node.func.id, node.args[position].id))
    return found


def misaligned_splits(cells: dict[int, str]) -> set[tuple[str, str, str]]:
    """(call, features, labels) for each fit/score/fit_resample call whose
    features and labels come from different train_test_split calls - e.g.
    features from one split, labels re-split separately - so rows and labels
    no longer line up. Names are traced to their split flow-insensitively."""
    nodes = [node for code in cells.values() for node in ast.walk(parse_cell(code))]
    assigns = [node for node in nodes if isinstance(node, ast.Assign)]

    origins: dict[str, set[int]] = {}
    for split, assign in enumerate(a for a in assigns if _call_name(a.value) == "train_test_split"):
        for name in (name for target in assign.targets for name in _stored(target)):
            origins.setdefault(name, set()).add(split)
    changed = True
    while changed:
        changed = False
        for assign in assigns:
            if _call_name(assign.value) == "train_test_split":
                continue
            reads = {node.id for node in ast.walk(assign.value) if isinstance(node, ast.Name)}
            inherited = set().union(*(origins.get(name, set()) for name in reads))
            for name in (name for target in assign.targets for name in _stored(target)):
                if not inherited <= origins.get(name, set()):
                    origins.setdefault(name, set()).update(inherited)
                    changed = True

    found = set()
    for node in nodes:
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in {"fit", "score", "fit_resample"} and len(node.args) >= 2
                and all(isinstance(arg, ast.Name) for arg in node.args[:2])):
            features, labels = node.args[0].id, node.args[1].id
            if origins.get(features) and origins.get(labels) and not origins[features] & origins[labels]:
                found.add((ast.unparse(node.func), features, labels))
    return found


def _call_name(node) -> str | None:
    if not isinstance(node, ast.Call):
        return None
    return node.func.attr if isinstance(node.func, ast.Attribute) else getattr(node.func, "id", None)


def _fit_argument(node) -> str | None:
    """The first argument of a `<receiver>.fit*(name, ...)` call, if a name."""
    if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in _FIT_METHODS
            and node.args and isinstance(node.args[0], ast.Name)):
        return node.args[0].id
    return None


def _derived(names: set[str], assigns) -> set[str]:
    """names, plus every name assigned from an expression reading one of them."""
    names = set(names)
    changed = True
    while changed:
        changed = False
        for assign in assigns:
            reads = {node.id for node in ast.walk(assign.value) if isinstance(node, ast.Name)}
            if reads & names:
                targets = {name for target in assign.targets for name in _stored(target)} - names
                if targets:
                    names |= targets
                    changed = True
    return names


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
