"""
Our own AST node hierarchy for notebook-cell code (Phase 1: P0 constructs).

Deliberately shaped like the stdlib `ast` module (a `_fields` tuple per node,
`lineno`/`col_offset`, a `NodeVisitor` base class with `visit_<ClassName>`
dispatch) so the traversal *pattern* is familiar, but every class here is our
own -- nothing is imported from `ast` or `gast`, and nothing here is an
instance of those modules' types.

Not yet implemented (tracked, not silently dropped -- see parser/README.md):
lambda, ternary `if/else` expressions, the walrus operator, decorators,
async/await, f-string interpolation (an f-string currently round-trips as
an opaque Constant, see notebook_python.lark), `del`/`assert`/`nonlocal`,
set literals, match statements, and same-cell function-call inlining
(calls are always treated as black-box in phase 1). Comprehensions
(list/set/dict, generator expressions) *are* implemented (`comprehension`,
`ListComp`/`SetComp`/`DictComp`/`GeneratorExp` below) but only for a
single `for` clause -- the grammar has no chained `for` support yet
(`[x for x in a for y in b]`), see parser/README.md.
"""

from collections import deque


class AST:
    _fields = ()

    def __init__(self, lineno=None, col_offset=None):
        self.lineno = lineno
        self.col_offset = col_offset

    def __repr__(self):
        fields = ", ".join(f"{f}={getattr(self, f, None)!r}" for f in self._fields)
        return f"{self.__class__.__name__}({fields})"


def iter_fields(node):
    for field in node._fields:
        if hasattr(node, field):
            yield field, getattr(node, field)


def iter_child_nodes(node):
    for _, value in iter_fields(node):
        if isinstance(value, list):
            for item in value:
                if isinstance(item, AST):
                    yield item
        elif isinstance(value, AST):
            yield value


def walk(node):
    todo = deque([node])
    while todo:
        n = todo.popleft()
        todo.extend(iter_child_nodes(n))
        yield n


def dump(node):
    """Structural repr used where analyses compare two nodes for equality
    via their dump() (e.g. detecting the "same statement" across cells);
    doesn't need to match stdlib ast.dump()'s exact text, only to be a
    stable, content-based string for a given node."""
    if isinstance(node, AST):
        fields = ", ".join(f"{f}={dump(getattr(node, f, None))}" for f in node._fields)
        return f"{node.__class__.__name__}({fields})"
    if isinstance(node, list):
        return "[" + ", ".join(dump(x) for x in node) + "]"
    return repr(node)


class NodeVisitor:
    def visit(self, node):
        return getattr(self, "visit_" + node.__class__.__name__, self.generic_visit)(node)

    def generic_visit(self, node):
        for _, value in iter_fields(node):
            if isinstance(value, list):
                for item in value:
                    if isinstance(item, AST):
                        self.visit(item)
            elif isinstance(value, AST):
                self.visit(value)


# -- expression contexts (mirrors ast.Load/Store/Del) -----------------------

class expr_context(AST):
    pass


class Load(expr_context):
    pass


class Store(expr_context):
    pass


class Del(expr_context):
    pass


# -- operators ----------------------------------------------------------

class operator(AST):
    pass


class boolop(AST):
    pass


class unaryop(AST):
    pass


class cmpop(AST):
    pass


def _op(name, base):
    return type(name, (base,), {})


Add, Sub, Mult, Div, FloorDiv, Mod, Pow, MatMult, LShift, RShift, BitOr, BitXor, BitAnd = (
    _op(n, operator)
    for n in (
        "Add", "Sub", "Mult", "Div", "FloorDiv", "Mod", "Pow", "MatMult",
        "LShift", "RShift", "BitOr", "BitXor", "BitAnd",
    )
)
And, Or = (_op(n, boolop) for n in ("And", "Or"))
UAdd, USub, Not, Invert = (_op(n, unaryop) for n in ("UAdd", "USub", "Not", "Invert"))
Eq, NotEq, Lt, LtE, Gt, GtE, Is, IsNot, In, NotIn = (
    _op(n, cmpop) for n in ("Eq", "NotEq", "Lt", "LtE", "Gt", "GtE", "Is", "IsNot", "In", "NotIn")
)


# -- expressions ----------------------------------------------------------

class expr(AST):
    pass


class Name(expr):
    _fields = ("id", "ctx")

    def __init__(self, id, ctx=None, **kw):
        super().__init__(**kw)
        self.id = id
        self.ctx = ctx or Load()


class Attribute(expr):
    _fields = ("value", "attr", "ctx")

    def __init__(self, value, attr, ctx=None, **kw):
        super().__init__(**kw)
        self.value = value
        self.attr = attr
        self.ctx = ctx or Load()


class Slice(AST):
    _fields = ("lower", "upper", "step")

    def __init__(self, lower=None, upper=None, step=None, **kw):
        super().__init__(**kw)
        self.lower = lower
        self.upper = upper
        self.step = step


class Subscript(expr):
    _fields = ("value", "slice", "ctx")

    def __init__(self, value, slice, ctx=None, **kw):
        super().__init__(**kw)
        self.value = value
        self.slice = slice
        self.ctx = ctx or Load()


class Starred(expr):
    _fields = ("value", "ctx")

    def __init__(self, value, ctx=None, **kw):
        super().__init__(**kw)
        self.value = value
        self.ctx = ctx or Load()


class keyword(AST):
    _fields = ("arg", "value")

    def __init__(self, arg, value, **kw):
        super().__init__(**kw)
        self.arg = arg  # None => **kwargs
        self.value = value


class Call(expr):
    _fields = ("func", "args", "keywords")

    def __init__(self, func, args=None, keywords=None, **kw):
        super().__init__(**kw)
        self.func = func
        self.args = args or []
        self.keywords = keywords or []


class BinOp(expr):
    _fields = ("left", "op", "right")

    def __init__(self, left, op, right, **kw):
        super().__init__(**kw)
        self.left = left
        self.op = op
        self.right = right


class BoolOp(expr):
    _fields = ("op", "values")

    def __init__(self, op, values, **kw):
        super().__init__(**kw)
        self.op = op
        self.values = values


class UnaryOp(expr):
    _fields = ("op", "operand")

    def __init__(self, op, operand, **kw):
        super().__init__(**kw)
        self.op = op
        self.operand = operand


class Compare(expr):
    _fields = ("left", "ops", "comparators")

    def __init__(self, left, ops, comparators, **kw):
        super().__init__(**kw)
        self.left = left
        self.ops = ops
        self.comparators = comparators


class Tuple(expr):
    _fields = ("elts", "ctx")

    def __init__(self, elts, ctx=None, **kw):
        super().__init__(**kw)
        self.elts = elts
        self.ctx = ctx or Load()


class List(expr):
    _fields = ("elts", "ctx")

    def __init__(self, elts, ctx=None, **kw):
        super().__init__(**kw)
        self.elts = elts
        self.ctx = ctx or Load()


class Dict(expr):
    _fields = ("keys", "values")

    def __init__(self, keys, values, **kw):
        super().__init__(**kw)
        self.keys = keys
        self.values = values


class comprehension(AST):
    """One `for ... in ... [if ...]*` clause of a comprehension. Mirrors
    stdlib ast.comprehension's target/iter/ifs shape. `generators` on the
    comprehension-expression classes below is a list of these for forward
    compatibility with stdlib ast's shape, even though the grammar
    currently only ever produces exactly one (no chained `for` clauses)."""

    _fields = ("target", "iter", "ifs")

    def __init__(self, target, iter, ifs=None, **kw):
        super().__init__(**kw)
        self.target = target
        self.iter = iter
        self.ifs = ifs or []


class ListComp(expr):
    _fields = ("elt", "generators")

    def __init__(self, elt, generators, **kw):
        super().__init__(**kw)
        self.elt = elt
        self.generators = generators


class SetComp(expr):
    _fields = ("elt", "generators")

    def __init__(self, elt, generators, **kw):
        super().__init__(**kw)
        self.elt = elt
        self.generators = generators


class GeneratorExp(expr):
    _fields = ("elt", "generators")

    def __init__(self, elt, generators, **kw):
        super().__init__(**kw)
        self.elt = elt
        self.generators = generators


class DictComp(expr):
    _fields = ("key", "value", "generators")

    def __init__(self, key, value, generators, **kw):
        super().__init__(**kw)
        self.key = key
        self.value = value
        self.generators = generators


class Constant(expr):
    _fields = ("value",)

    def __init__(self, value, **kw):
        super().__init__(**kw)
        self.value = value


# -- statements ----------------------------------------------------------

class stmt(AST):
    pass


class Module(AST):
    _fields = ("body",)

    def __init__(self, body, **kw):
        super().__init__(**kw)
        self.body = body


class Expr(stmt):
    _fields = ("value",)

    def __init__(self, value, **kw):
        super().__init__(**kw)
        self.value = value


class Assign(stmt):
    _fields = ("targets", "value")

    def __init__(self, targets, value, **kw):
        super().__init__(**kw)
        self.targets = targets
        self.value = value


class AugAssign(stmt):
    _fields = ("target", "op", "value")

    def __init__(self, target, op, value, **kw):
        super().__init__(**kw)
        self.target = target
        self.op = op
        self.value = value


class AnnAssign(stmt):
    _fields = ("target", "annotation", "value", "simple")

    def __init__(self, target, annotation, value=None, simple=1, **kw):
        super().__init__(**kw)
        self.target = target
        self.annotation = annotation
        self.value = value
        self.simple = simple


class Global(stmt):
    _fields = ("names",)

    def __init__(self, names, **kw):
        super().__init__(**kw)
        self.names = names


class alias(AST):
    _fields = ("name", "asname")

    def __init__(self, name, asname=None, **kw):
        super().__init__(**kw)
        self.name = name
        self.asname = asname


class Import(stmt):
    _fields = ("names",)

    def __init__(self, names, **kw):
        super().__init__(**kw)
        self.names = names


class ImportFrom(stmt):
    _fields = ("module", "names", "level")

    def __init__(self, module, names, level=0, **kw):
        super().__init__(**kw)
        self.module = module
        self.names = names
        self.level = level


class If(stmt):
    _fields = ("test", "body", "orelse")

    def __init__(self, test, body, orelse=None, **kw):
        super().__init__(**kw)
        self.test = test
        self.body = body
        self.orelse = orelse or []


class While(stmt):
    _fields = ("test", "body", "orelse")

    def __init__(self, test, body, orelse=None, **kw):
        super().__init__(**kw)
        self.test = test
        self.body = body
        self.orelse = orelse or []


class For(stmt):
    _fields = ("target", "iter", "body", "orelse")

    def __init__(self, target, iter, body, orelse=None, **kw):
        super().__init__(**kw)
        self.target = target
        self.iter = iter
        self.body = body
        self.orelse = orelse or []


class ExceptHandler(AST):
    _fields = ("type", "name", "body")

    def __init__(self, type=None, name=None, body=None, **kw):
        super().__init__(**kw)
        self.type = type
        self.name = name
        self.body = body or []


class Try(stmt):
    _fields = ("body", "handlers", "orelse", "finalbody")

    def __init__(self, body, handlers=None, orelse=None, finalbody=None, **kw):
        super().__init__(**kw)
        self.body = body
        self.handlers = handlers or []
        self.orelse = orelse or []
        self.finalbody = finalbody or []


class withitem(AST):
    _fields = ("context_expr", "optional_vars")

    def __init__(self, context_expr, optional_vars=None, **kw):
        super().__init__(**kw)
        self.context_expr = context_expr
        self.optional_vars = optional_vars


class With(stmt):
    _fields = ("items", "body")

    def __init__(self, items, body, **kw):
        super().__init__(**kw)
        self.items = items
        self.body = body


class Return(stmt):
    _fields = ("value",)

    def __init__(self, value=None, **kw):
        super().__init__(**kw)
        self.value = value


class Raise(stmt):
    _fields = ("exc", "cause")

    def __init__(self, exc=None, cause=None, **kw):
        super().__init__(**kw)
        self.exc = exc
        self.cause = cause


class Break(stmt):
    _fields = ()


class Continue(stmt):
    _fields = ()


class Pass(stmt):
    _fields = ()


class Unsupported(stmt):
    """Placeholder for a statement using a construct the IR doesn't support
    yet (see ast_transformer's __default__). Reads and binds nothing; the CFG
    builder turns it into a no-op node plus a warning, so one unsupported
    statement doesn't reject the whole cell."""
    _fields = ()

    def __init__(self, construct, **kw):
        super().__init__(**kw)
        self.construct = construct


class arg(AST):
    _fields = ("arg", "annotation")

    def __init__(self, arg, annotation=None, **kw):
        super().__init__(**kw)
        self.arg = arg
        self.annotation = annotation


class arguments(AST):
    _fields = ("args", "vararg", "kwonlyargs", "kw_defaults", "kwarg", "defaults")

    def __init__(self, args=None, vararg=None, kwonlyargs=None, kw_defaults=None,
                 kwarg=None, defaults=None, **kw):
        super().__init__(**kw)
        self.args = args or []
        self.vararg = vararg
        self.kwonlyargs = kwonlyargs or []
        self.kw_defaults = kw_defaults or []
        self.kwarg = kwarg
        self.defaults = defaults or []


class FunctionDef(stmt):
    _fields = ("name", "args", "body", "decorator_list", "returns")

    def __init__(self, name, args, body, decorator_list=None, returns=None, **kw):
        super().__init__(**kw)
        self.name = name
        self.args = args
        self.body = body
        self.decorator_list = decorator_list or []
        self.returns = returns


class ClassDef(stmt):
    _fields = ("name", "bases", "keywords", "body", "decorator_list")

    def __init__(self, name, bases=None, keywords=None, body=None, decorator_list=None, **kw):
        super().__init__(**kw)
        self.name = name
        self.bases = bases or []
        self.keywords = keywords or []
        self.body = body or []
        self.decorator_list = decorator_list or []
