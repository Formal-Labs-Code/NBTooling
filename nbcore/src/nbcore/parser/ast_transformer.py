"""
Lark parse tree -> ast_nodes.* transformer.

Only the grammar rules needed for the P0 (and a few easy P1) constructs have
an explicit method here -- see notebook_python.lark's priority tags. Anything
else (lambda, ternary `if/else`, walrus, decorated call args, async,
del/assert/nonlocal, f-string interpolation, dict/set unpacking, bare set
literals) hits `__default__` below and raises `NotImplementedError` naming
the construct, rather than silently producing a wrong tree. The innermost
statement containing it then becomes an `n.Unsupported` placeholder (see
ASTTransformer._transform_tree), so the rest of the cell is still analyzed.
That list is the tracked backlog for the next increment -- see
parser/README.md.

Comprehensions (list/set/dict, generator expressions) *are* implemented,
via `comp_for`/`comp_if`/`set_comp`/`dict_comp`/`genexpr_arg` below, but
only for a single `for` clause (the grammar has no chained-`for` support).
"""

import codecs
import re

from lark import Transformer, v_args
from lark.exceptions import VisitError

from . import ast_nodes as n

_STRING_PREFIX_RE = re.compile(r"^[a-zA-Z]{0,2}")


def _flatten(items):
    out = []
    for item in items:
        if isinstance(item, list):
            out.extend(item)
        elif item is not None:
            out.append(item)
    return out


def _set_ctx(node, ctx_cls):
    """Recursively rewrite an expression used as an assignment target/del
    target so Name/Attribute/Subscript/Starred/Tuple/List carry `ctx_cls`
    instead of the Load() they got when built as a plain expression."""
    if isinstance(node, (n.Name, n.Attribute, n.Subscript)):
        node.ctx = ctx_cls()
    elif isinstance(node, n.Starred):
        node.ctx = ctx_cls()
        _set_ctx(node.value, ctx_cls)
    elif isinstance(node, (n.Tuple, n.List)):
        node.ctx = ctx_cls()
        for elt in node.elts:
            _set_ctx(elt, ctx_cls)
    return node


def _parse_number(text):
    t = text.replace("_", "")
    if t[-1] in "jJ":
        return complex(t)
    low = t.lower()
    if low.startswith(("0x", "0o", "0b")):
        return int(t, 0)
    if "." in t or "e" in low:
        return float(t)
    return int(t)


def _parse_string_literal(text):
    """Decode a STRING token's text into its Python value.

    Known simplification: f-strings are *not* interpolated here -- the `{..}`
    parts round-trip as literal characters. Byte-string literals are kept as
    `str`, not `bytes`. Both are P1 gaps tracked for a later increment.
    """
    prefix = _STRING_PREFIX_RE.match(text).group(0).lower()
    body = text[len(prefix):]
    quote = body[:3] if body[:3] in ('"""', "'''") else body[0]
    inner = body[len(quote):-len(quote)]
    if "r" in prefix:
        return inner
    try:
        return codecs.decode(inner.encode("utf-8", "backslashreplace"), "unicode_escape")
    except UnicodeDecodeError:
        return inner


_AUG_OPS = {
    "+=": n.Add, "-=": n.Sub, "*=": n.Mult, "/=": n.Div, "//=": n.FloorDiv,
    "%=": n.Mod, "**=": n.Pow, "&=": n.BitAnd, "|=": n.BitOr, "^=": n.BitXor,
    ">>=": n.RShift, "<<=": n.LShift, "@=": n.MatMult,
}

_COMP_OPS = {
    "<": n.Lt, ">": n.Gt, "==": n.Eq, ">=": n.GtE, "<=": n.LtE,
    "<>": n.NotEq, "!=": n.NotEq, "in": n.In, "not in": n.NotIn,
    "is": n.Is, "is not": n.IsNot,
}

_ARITH_OPS = {
    "+": n.Add, "-": n.Sub, "*": n.Mult, "@": n.MatMult, "/": n.Div,
    "%": n.Mod, "//": n.FloorDiv, "<<": n.LShift, ">>": n.RShift,
}

_UNARY_OPS = {"+": n.UAdd, "-": n.USub, "~": n.Invert}


def _is_statement_rule(rule: str) -> bool:
    return (rule.endswith("_stmt") and rule != "simple_stmt") or rule in ("funcdef", "classdef")


@v_args(meta=True)
class ASTTransformer(Transformer):
    def _transform_tree(self, tree):
        """A statement whose subtree hits an unsupported construct becomes an
        n.Unsupported placeholder instead of failing the whole cell. Lark
        transforms bottom-up, so this catches at the innermost statement: an
        unsupported expression inside a def body only replaces that one body
        statement, not the def."""
        if not _is_statement_rule(str(tree.data)):
            return super()._transform_tree(tree)
        try:
            return super()._transform_tree(tree)
        except NotImplementedError as exc:  # raised by __default__, which lark calls unwrapped
            return n.Unsupported(construct=str(exc), lineno=tree.meta.line)
        except VisitError as exc:  # raised inside a rule method
            if not isinstance(exc.orig_exc, NotImplementedError):
                raise
            return n.Unsupported(construct=str(exc.orig_exc), lineno=tree.meta.line)

    # -- module / top level -------------------------------------------

    def start(self, meta, children):
        return n.Module(body=_flatten(children), lineno=1)

    def simple_stmt(self, meta, children):
        return _flatten(children)

    def suite(self, meta, children):
        return _flatten(children)

    # -- simple statements ----------------------------------------------

    def expr_stmt(self, meta, children):
        return n.Expr(value=children[0], lineno=meta.line)

    def assign_target(self, meta, children):
        return _set_ctx(children[0], n.Store)

    def assign_stmt(self, meta, children):
        *targets, value = children
        return n.Assign(targets=targets, value=value, lineno=meta.line)

    def aug_assign_op(self, meta, children):
        return "".join(str(c) for c in children)

    def aug_assign_stmt(self, meta, children):
        target, op_text, value = children
        _set_ctx(target, n.Store)
        return n.AugAssign(target=target, op=_AUG_OPS[op_text](), value=value, lineno=meta.line)

    def ann_assign_stmt(self, meta, children):
        target, annotation, value = children
        _set_ctx(target, n.Store)
        return n.AnnAssign(target=target, annotation=annotation, value=value, lineno=meta.line)

    def pass_stmt(self, meta, children):
        return n.Pass(lineno=meta.line)

    def break_stmt(self, meta, children):
        return n.Break(lineno=meta.line)

    def continue_stmt(self, meta, children):
        return n.Continue(lineno=meta.line)

    def return_stmt(self, meta, children):
        return n.Return(value=children[0], lineno=meta.line)

    def raise_stmt(self, meta, children):
        exc, cause = children
        return n.Raise(exc=exc, cause=cause, lineno=meta.line)

    def global_stmt(self, meta, children):
        return n.Global(names=[str(t) for t in children], lineno=meta.line)

    # -- imports ----------------------------------------------------------

    def dotted_name(self, meta, children):
        return ".".join(str(t) for t in children)

    def dotted_as_name(self, meta, children):
        name, asname = children
        return n.alias(name=name, asname=str(asname) if asname is not None else None)

    def dotted_as_names(self, meta, children):
        return list(children)

    def import_stmt(self, meta, children):
        return n.Import(names=children[0], lineno=meta.line)

    def import_from_source(self, meta, children):
        # Token is itself a `str` subclass, so the dots (raw Tokens) and the
        # already-transformed dotted_name (a plain `str`) must be told apart
        # by excluding Token, not just checking `isinstance(c, str)`.
        from lark import Token
        level = sum(1 for c in children if isinstance(c, Token))
        module = next((c for c in children if isinstance(c, str) and not isinstance(c, Token)), None)
        return (level, module)

    def import_as_name(self, meta, children):
        name, asname = children
        return n.alias(name=str(name), asname=str(asname) if asname is not None else None)

    def import_as_names(self, meta, children):
        return list(children)

    def import_from_targets(self, meta, children):
        if not children:
            return [n.alias(name="*", asname=None)]
        return children[0]

    def import_from_stmt(self, meta, children):
        (level, module), names = children
        return n.ImportFrom(module=module, names=names, level=level, lineno=meta.line)

    # -- compound statements ----------------------------------------------

    def if_stmt(self, meta, children):
        orelse = children[-1] or []
        pairs = list(zip(children[:-1][0::2], children[:-1][1::2]))
        node = n.If(test=pairs[-1][0], body=pairs[-1][1], orelse=orelse, lineno=meta.line)
        for test, body in reversed(pairs[:-1]):
            node = n.If(test=test, body=body, orelse=[node], lineno=meta.line)
        return node

    def while_stmt(self, meta, children):
        test, body, orelse = children
        return n.While(test=test, body=body, orelse=orelse or [], lineno=meta.line)

    def for_stmt(self, meta, children):
        target, iter_, body, orelse = children
        _set_ctx(target, n.Store)
        return n.For(target=target, iter=iter_, body=body, orelse=orelse or [], lineno=meta.line)

    def except_clause(self, meta, children):
        exc_type, name, body = children
        return n.ExceptHandler(
            type=exc_type, name=str(name) if name is not None else None,
            body=body, lineno=meta.line,
        )

    def try_stmt(self, meta, children):
        if len(children) == 2:
            body, finalbody = children
            return n.Try(body=body, handlers=[], orelse=[], finalbody=finalbody, lineno=meta.line)
        body, *rest, orelse, finalbody = children
        return n.Try(
            body=body, handlers=rest, orelse=orelse or [], finalbody=finalbody or [],
            lineno=meta.line,
        )

    def with_item(self, meta, children):
        context_expr, optional_vars = children
        if optional_vars is not None:
            _set_ctx(optional_vars, n.Store)
        return n.withitem(context_expr=context_expr, optional_vars=optional_vars)

    def with_stmt(self, meta, children):
        *items, body = children
        return n.With(items=items, body=body, lineno=meta.line)

    # -- function / class defs --------------------------------------------

    def decorators(self, meta, children):
        return list(children)

    def decorator(self, meta, children):
        dotted = children[0]
        # Decorator call-args aren't preserved in phase 1 -- decorator_list
        # isn't consumed by the CFG builder yet, only its presence matters.
        parts = dotted.split(".")
        node = n.Name(id=parts[0], ctx=n.Load(), lineno=meta.line)
        for attr in parts[1:]:
            node = n.Attribute(value=node, attr=attr, ctx=n.Load(), lineno=meta.line)
        return node

    def plain_param(self, meta, children):
        name, annotation, default = children
        return ("plain", n.arg(arg=str(name), annotation=annotation, lineno=meta.line), default)

    def star_param(self, meta, children):
        name, annotation = children
        return ("star", n.arg(arg=str(name), annotation=annotation, lineno=meta.line) if name is not None else None)

    def dstar_param(self, meta, children):
        name, annotation = children
        return ("dstar", n.arg(arg=str(name), annotation=annotation, lineno=meta.line))

    def parameters(self, meta, children):
        args, defaults, kwonlyargs, kw_defaults = [], [], [], []
        vararg, kwarg = None, None
        seen_star = False
        for item in children:
            kind = item[0]
            if kind == "plain":
                _, a, default = item
                if seen_star:
                    kwonlyargs.append(a)
                    kw_defaults.append(default)
                else:
                    args.append(a)
                    if default is not None:
                        defaults.append(default)
            elif kind == "star":
                vararg = item[1]
                seen_star = True
            elif kind == "dstar":
                kwarg = item[1]
        return n.arguments(
            args=args, vararg=vararg, kwonlyargs=kwonlyargs,
            kw_defaults=kw_defaults, kwarg=kwarg, defaults=defaults,
        )

    def funcdef(self, meta, children):
        decorators, name, parameters, returns, body = children
        args = parameters if parameters is not None else n.arguments()
        return n.FunctionDef(
            name=str(name), args=args, body=body,
            decorator_list=decorators or [], returns=returns, lineno=meta.line,
        )

    def classdef(self, meta, children):
        if len(children) == 3:
            decorators, name, body = children
            bases, keywords = [], []
        else:
            decorators, name, arglist_result, body = children
            bases, keywords = arglist_result if arglist_result is not None else ([], [])
        return n.ClassDef(
            name=str(name), bases=bases, keywords=keywords, body=body,
            decorator_list=decorators or [], lineno=meta.line,
        )

    # -- testlists / star expressions -------------------------------------

    def testlist_star_expr(self, meta, children):
        if len(children) == 1:
            return children[0]
        return n.Tuple(elts=children, ctx=n.Load(), lineno=meta.line)

    def star_expr(self, meta, children):
        return n.Starred(value=children[0], ctx=n.Load(), lineno=meta.line)

    def exprlist(self, meta, children):
        if len(children) == 1:
            return children[0]
        return n.Tuple(elts=children, ctx=n.Load(), lineno=meta.line)

    def ternary_test(self, meta, children):
        value = children[0]
        if any(c is not None for c in children[1:]):
            raise NotImplementedError(
                "ternary conditional expressions ('x if y else z') not yet supported"
            )
        return value

    # -- boolean / comparison ----------------------------------------------

    def or_test(self, meta, children):
        return n.BoolOp(op=n.Or(), values=children, lineno=meta.line)

    def and_test(self, meta, children):
        return n.BoolOp(op=n.And(), values=children, lineno=meta.line)

    def not_op(self, meta, children):
        return n.UnaryOp(op=n.Not(), operand=children[0], lineno=meta.line)

    def comp_op(self, meta, children):
        text = " ".join(str(t) for t in children)
        return _COMP_OPS[text]()

    def comparison(self, meta, children):
        left = children[0]
        ops = children[1::2]
        comparators = children[2::2]
        return n.Compare(left=left, ops=ops, comparators=comparators, lineno=meta.line)

    # -- bitwise / arithmetic ----------------------------------------------

    def or_bitwise(self, meta, children):
        node = children[0]
        for rhs in children[1:]:
            node = n.BinOp(left=node, op=n.BitOr(), right=rhs, lineno=meta.line)
        return node

    def xor_bitwise(self, meta, children):
        node = children[0]
        for rhs in children[1:]:
            node = n.BinOp(left=node, op=n.BitXor(), right=rhs, lineno=meta.line)
        return node

    def and_bitwise(self, meta, children):
        node = children[0]
        for rhs in children[1:]:
            node = n.BinOp(left=node, op=n.BitAnd(), right=rhs, lineno=meta.line)
        return node

    def shift_expr_op(self, meta, children):
        left, op_text, right = children
        return n.BinOp(left=left, op=_ARITH_OPS[op_text](), right=right, lineno=meta.line)

    def arith_expr_op(self, meta, children):
        left, op_text, right = children
        return n.BinOp(left=left, op=_ARITH_OPS[op_text](), right=right, lineno=meta.line)

    def term_op(self, meta, children):
        left, op_text, right = children
        return n.BinOp(left=left, op=_ARITH_OPS[op_text](), right=right, lineno=meta.line)

    def factor_op(self, meta, children):
        op_text = str(children[0])
        operand = children[1]
        return (op_text, operand)

    def unary_op(self, meta, children):
        op_text, operand = children[0]
        return n.UnaryOp(op=_UNARY_OPS[op_text](), operand=operand, lineno=meta.line)

    def power(self, meta, children):
        base, exponent = children
        if exponent is None:
            return base
        return n.BinOp(left=base, op=n.Pow(), right=exponent, lineno=meta.line)

    # -- postfix chain: attr access / call / subscript --------------------

    def attr_access(self, meta, children):
        return ("attr", str(children[0]))

    def call(self, meta, children):
        args, keywords = children[0] if children[0] is not None else ([], [])
        return ("call", args, keywords)

    def subscript(self, meta, children):
        return ("subscript", children[0])

    def atom_trailer(self, meta, children):
        node = children[0]
        for op in children[1:]:
            if op[0] == "attr":
                node = n.Attribute(value=node, attr=op[1], ctx=n.Load(), lineno=meta.line)
            elif op[0] == "call":
                node = n.Call(func=node, args=op[1], keywords=op[2], lineno=meta.line)
            elif op[0] == "subscript":
                node = n.Subscript(value=node, slice=op[1], ctx=n.Load(), lineno=meta.line)
        return node

    def index_subscript(self, meta, children):
        return children[0]

    def slice_subscript(self, meta, children):
        lower, upper, step = children
        return n.Slice(lower=lower, upper=upper, step=step, lineno=meta.line)

    def subscriptlist(self, meta, children):
        if len(children) == 1:
            return children[0]
        return n.Tuple(elts=children, ctx=n.Load(), lineno=meta.line)

    # -- call arguments -----------------------------------------------------

    def pos_arg(self, meta, children):
        return ("pos", children[0])

    def kw_arg(self, meta, children):
        name, value = children
        return ("kw", n.keyword(arg=str(name), value=value, lineno=meta.line))

    def star_arg(self, meta, children):
        return ("star", children[0])

    def dstar_arg(self, meta, children):
        return ("dstar", children[0])

    def genexpr_arg(self, meta, children):
        elt, comp = children
        return ("pos", n.GeneratorExp(elt=elt, generators=[comp], lineno=meta.line))

    def arglist(self, meta, children):
        args, keywords = [], []
        for kind, value in children:
            if kind == "pos":
                args.append(value)
            elif kind == "star":
                args.append(n.Starred(value=value, ctx=n.Load(), lineno=meta.line))
            elif kind == "kw":
                keywords.append(value)
            elif kind == "dstar":
                keywords.append(n.keyword(arg=None, value=value, lineno=meta.line))
        return (args, keywords)

    # -- atoms --------------------------------------------------------------

    def name_atom(self, meta, children):
        return n.Name(id=str(children[0]), ctx=n.Load(), lineno=meta.line)

    def number_atom(self, meta, children):
        return n.Constant(value=_parse_number(str(children[0])), lineno=meta.line)

    def string_atom(self, meta, children):
        return n.Constant(value="".join(_parse_string_literal(str(t)) for t in children), lineno=meta.line)

    def const_none(self, meta, children):
        return n.Constant(value=None, lineno=meta.line)

    def const_true(self, meta, children):
        return n.Constant(value=True, lineno=meta.line)

    def const_false(self, meta, children):
        return n.Constant(value=False, lineno=meta.line)

    def const_ellipsis(self, meta, children):
        return n.Constant(value=Ellipsis, lineno=meta.line)

    def tuple_atom(self, meta, children):
        inner = children[0]
        if inner is None:
            return n.Tuple(elts=[], ctx=n.Load(), lineno=meta.line)
        # A parenthesized generator expression `(x for x in y)` already
        # comes back as a bare GeneratorExp from testlist_comp_tuple below -
        # passed through as-is, correctly not wrapped as a Tuple.
        return inner

    def testlist_comp_tuple(self, meta, children):
        if len(children) == 2 and isinstance(children[1], n.comprehension):
            elt, comp = children
            return n.GeneratorExp(elt=elt, generators=[comp], lineno=meta.line)
        if len(children) == 1:
            return children[0]
        return n.Tuple(elts=children, ctx=n.Load(), lineno=meta.line)

    def list_atom(self, meta, children):
        inner = children[0]
        if isinstance(inner, n.ListComp):
            return inner
        return n.List(elts=(inner if inner is not None else []), ctx=n.Load(), lineno=meta.line)

    def testlist_comp_list(self, meta, children):
        if len(children) == 2 and isinstance(children[1], n.comprehension):
            elt, comp = children
            return n.ListComp(elt=elt, generators=[comp], lineno=meta.line)
        return children

    def comp_for(self, meta, children):
        target, iter_, ifs = children
        _set_ctx(target, n.Store)
        return n.comprehension(target=target, iter=iter_, ifs=ifs or [])

    def comp_if(self, meta, children):
        test, chained = children
        return [test] + (chained or [])

    def dict_or_set_atom(self, meta, children):
        inner = children[0]
        if inner is None:
            return n.Dict(keys=[], values=[], lineno=meta.line)
        return inner

    def set_comp(self, meta, children):
        elt, comp = children
        return n.SetComp(elt=elt, generators=[comp], lineno=meta.line)

    def dict_comp(self, meta, children):
        key, value, comp = children
        return n.DictComp(key=key, value=value, generators=[comp], lineno=meta.line)

    def dict_pair(self, meta, children):
        if len(children) == 2:
            return (children[0], children[1])
        raise NotImplementedError(
            "dict unpacking ('**' inside a {..} literal) is not yet supported"
        )

    def dict_literal(self, meta, children):
        pairs = [c for c in children if c is not None]
        keys = [k for k, _ in pairs]
        values = [v for _, v in pairs]
        return n.Dict(keys=keys, values=values, lineno=meta.line)

    # -- fallback -------------------------------------------------------

    def __default__(self, data, children, meta):
        raise NotImplementedError(
            f"grammar construct '{data}' is not yet supported by the IR "
            "(tracked for a later increment, see ast_transformer.py docstring)"
        )


def build_ast(tree):
    return ASTTransformer().transform(tree)
