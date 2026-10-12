import unittest

from nbcore.parser import ast_nodes as n
from nbcore.parser.ast_transformer import build_ast
from nbcore.parser.lark_parser import parse_to_tree
from nbcore.resource_utils.utils import TEST_RES_PATH, load_notebook, read_json


def parse(src):
    return build_ast(parse_to_tree(src))


class TestParserAST(unittest.TestCase):
    def test_simple_assign(self):
        mod = parse("x = 1")
        (stmt,) = mod.body
        self.assertIsInstance(stmt, n.Assign)
        self.assertEqual(stmt.targets[0].id, "x")
        self.assertIsInstance(stmt.targets[0].ctx, n.Store)
        self.assertEqual(stmt.value.value, 1)

    def test_tuple_unpack_assign(self):
        mod = parse("x, y = 1, 2")
        (stmt,) = mod.body
        self.assertIsInstance(stmt.targets[0], n.Tuple)
        self.assertEqual([e.id for e in stmt.targets[0].elts], ["x", "y"])
        self.assertTrue(all(isinstance(e.ctx, n.Store) for e in stmt.targets[0].elts))

    def test_attribute_and_subscript_targets(self):
        mod = parse("x.y = 1\nx[0] = 1\n")
        attr_assign, sub_assign = mod.body
        self.assertIsInstance(attr_assign.targets[0], n.Attribute)
        self.assertIsInstance(attr_assign.targets[0].ctx, n.Store)
        self.assertIsInstance(attr_assign.targets[0].value.ctx, n.Load)
        self.assertIsInstance(sub_assign.targets[0], n.Subscript)
        self.assertIsInstance(sub_assign.targets[0].ctx, n.Store)

    def test_aug_and_ann_assign(self):
        mod = parse("x += 1\nx: int = 1\n")
        aug, ann = mod.body
        self.assertIsInstance(aug, n.AugAssign)
        self.assertIsInstance(aug.op, n.Add)
        self.assertIsInstance(ann, n.AnnAssign)
        self.assertEqual(ann.annotation.id, "int")

    def test_call_chain_and_slice_subscript(self):
        mod = parse("y = df.iloc[1:5, ::2]")
        (stmt,) = mod.body
        sub = stmt.value
        self.assertIsInstance(sub, n.Subscript)
        self.assertIsInstance(sub.value, n.Attribute)
        self.assertEqual(sub.value.attr, "iloc")
        self.assertIsInstance(sub.slice, n.Tuple)
        first, second = sub.slice.elts
        self.assertIsInstance(first, n.Slice)
        self.assertEqual(first.lower.value, 1)
        self.assertEqual(first.upper.value, 5)
        self.assertIsInstance(second, n.Slice)
        self.assertIsNone(second.lower)
        self.assertEqual(second.step.value, 2)

    def test_import_and_import_from(self):
        mod = parse(
            "import pandas as pd\n"
            "from sklearn.model_selection import train_test_split\n"
            "from . import foo\n"
            "from ..pkg import bar as b\n"
        )
        imp, frm, frm_rel, frm_rel2 = mod.body
        self.assertIsInstance(imp, n.Import)
        self.assertEqual(imp.names[0].name, "pandas")
        self.assertEqual(imp.names[0].asname, "pd")
        self.assertIsInstance(frm, n.ImportFrom)
        self.assertEqual(frm.module, "sklearn.model_selection")
        self.assertEqual(frm.names[0].name, "train_test_split")
        self.assertEqual(frm_rel.level, 1)
        self.assertIsNone(frm_rel.module)
        self.assertEqual(frm_rel2.level, 2)
        self.assertEqual(frm_rel2.module, "pkg")

    def test_if_elif_else_nests_as_orelse(self):
        mod = parse(
            "if x > 5 and y < 3:\n"
            "    z = 1\n"
            "elif x == 0:\n"
            "    z = 2\n"
            "else:\n"
            "    z = 3\n"
        )
        (if_node,) = mod.body
        self.assertIsInstance(if_node, n.If)
        self.assertIsInstance(if_node.test, n.BoolOp)
        self.assertIsInstance(if_node.test.op, n.And)
        elif_node = if_node.orelse[0]
        self.assertIsInstance(elif_node, n.If)
        self.assertIsInstance(elif_node.test, n.Compare)
        self.assertEqual(elif_node.orelse[0].value.value, 3)

    def test_for_loop_with_else_and_in_disambiguation(self):
        mod = parse(
            "for i in range(10):\n"
            "    total += i\n"
            "else:\n"
            "    print(total)\n"
        )
        (for_node,) = mod.body
        self.assertIsInstance(for_node, n.For)
        self.assertEqual(for_node.target.id, "i")
        self.assertIsInstance(for_node.target.ctx, n.Store)
        self.assertIsInstance(for_node.iter, n.Call)
        self.assertEqual(len(for_node.orelse), 1)

    def test_while_and_break(self):
        mod = parse("while x > 0:\n    x -= 1\n    if x == 5:\n        break\n")
        (while_node,) = mod.body
        self.assertIsInstance(while_node, n.While)
        inner_if = while_node.body[1]
        self.assertIsInstance(inner_if.body[0], n.Break)

    def test_try_except_else_finally(self):
        mod = parse(
            "try:\n"
            "    risky()\n"
            "except ValueError as e:\n"
            "    handle(e)\n"
            "except (TypeError, KeyError):\n"
            "    pass\n"
            "else:\n"
            "    ok()\n"
            "finally:\n"
            "    cleanup()\n"
        )
        (try_node,) = mod.body
        self.assertIsInstance(try_node, n.Try)
        self.assertEqual(len(try_node.handlers), 2)
        self.assertEqual(try_node.handlers[0].type.id, "ValueError")
        self.assertEqual(try_node.handlers[0].name, "e")
        self.assertIsInstance(try_node.handlers[1].type, n.Tuple)
        self.assertIsNone(try_node.handlers[1].name)
        self.assertEqual(len(try_node.orelse), 1)
        self.assertEqual(len(try_node.finalbody), 1)

    def test_try_finally_only(self):
        mod = parse("try:\n    f()\nfinally:\n    g()\n")
        (try_node,) = mod.body
        self.assertEqual(try_node.handlers, [])
        self.assertEqual(len(try_node.finalbody), 1)

    def test_with_statement(self):
        mod = parse('with open("f.txt") as f:\n    data = f.read()\n')
        (with_node,) = mod.body
        self.assertIsInstance(with_node, n.With)
        self.assertEqual(with_node.items[0].optional_vars.id, "f")
        self.assertIsInstance(with_node.items[0].optional_vars.ctx, n.Store)

    def test_funcdef_parameter_kinds(self):
        mod = parse("def foo(a, b=1, *args, c, d=2, **kwargs):\n    return a + b\n")
        (fn,) = mod.body
        self.assertIsInstance(fn, n.FunctionDef)
        self.assertEqual([a.arg for a in fn.args.args], ["a", "b"])
        self.assertEqual([d.value for d in fn.args.defaults], [1])
        self.assertEqual(fn.args.vararg.arg, "args")
        self.assertEqual([a.arg for a in fn.args.kwonlyargs], ["c", "d"])
        self.assertEqual(fn.args.kwarg.arg, "kwargs")
        self.assertIsInstance(fn.body[0], n.Return)

    def test_classdef_with_bases_and_nested_method(self):
        mod = parse("class Foo(Base):\n    def method(self):\n        pass\n")
        (cls,) = mod.body
        self.assertIsInstance(cls, n.ClassDef)
        self.assertEqual(cls.bases[0].id, "Base")
        method = cls.body[0]
        self.assertIsInstance(method, n.FunctionDef)
        self.assertEqual(method.args.args[0].arg, "self")

    def test_call_with_star_and_kwargs(self):
        mod = parse("z = f(1, 2, x=3, *rest, **more)")
        call = mod.body[0].value
        self.assertIsInstance(call, n.Call)
        self.assertEqual([type(a).__name__ for a in call.args], ["Constant", "Constant", "Starred"])
        self.assertEqual(call.keywords[0].arg, "x")
        self.assertIsNone(call.keywords[1].arg)  # **more

    def test_operator_precedence_shape(self):
        # "+" is loosest here, so it must sit at the root; everything at
        # */%// precedence binds into its right operand first.
        mod = parse("z = a + b * (c - d) ** 2 % 3 // 4")
        expr = mod.body[0].value
        self.assertIsInstance(expr, n.BinOp)
        self.assertIsInstance(expr.op, n.Add)
        self.assertEqual(expr.left.id, "a")
        term_chain = expr.right
        self.assertIsInstance(term_chain.op, n.FloorDiv)
        self.assertIsInstance(term_chain.left.op, n.Mod)

    def test_decorator_is_attached(self):
        mod = parse("@decorator\ndef foo():\n    pass\n")
        (fn,) = mod.body
        self.assertEqual(len(fn.decorator_list), 1)
        self.assertEqual(fn.decorator_list[0].id, "decorator")

    def test_numeric_literals(self):
        mod = parse("a = 0x1e\nb = 1_000_000\nc = 1.5e10\n")
        a, b, c = (s.value.value for s in mod.body)
        self.assertEqual(a, 30)
        self.assertEqual(b, 1000000)
        self.assertEqual(c, 1.5e10)

    def test_fstring_is_opaque_literal_known_gap(self):
        # Documented phase-1 limitation: no interpolation, kept literal.
        mod = parse("s = f'value is {x}'")
        self.assertEqual(mod.body[0].value.value, "value is {x}")

    def test_list_comprehension(self):
        mod = parse("y = [x * x for x in z]")
        (stmt,) = mod.body
        comp = stmt.value
        self.assertIsInstance(comp, n.ListComp)
        self.assertIsInstance(comp.elt, n.BinOp)
        (generator,) = comp.generators
        self.assertEqual(generator.target.id, "x")
        self.assertIsInstance(generator.target.ctx, n.Store)
        self.assertEqual(generator.iter.id, "z")
        self.assertEqual(generator.ifs, [])

    def test_list_comprehension_with_if_filter(self):
        mod = parse("y = [x for x in z if x > 0]")
        (stmt,) = mod.body
        (generator,) = stmt.value.generators
        (if_test,) = generator.ifs
        self.assertIsInstance(if_test, n.Compare)

    def test_set_comprehension(self):
        mod = parse("s = {x for x in z}")
        (stmt,) = mod.body
        self.assertIsInstance(stmt.value, n.SetComp)

    def test_dict_comprehension_with_tuple_target(self):
        mod = parse("d = {k: v for k, v in z}")
        (stmt,) = mod.body
        comp = stmt.value
        self.assertIsInstance(comp, n.DictComp)
        (generator,) = comp.generators
        self.assertIsInstance(generator.target, n.Tuple)
        self.assertEqual([e.id for e in generator.target.elts], ["k", "v"])
        self.assertTrue(all(isinstance(e.ctx, n.Store) for e in generator.target.elts))

    def test_parenthesized_generator_expression(self):
        mod = parse("g = (x for x in z)")
        (stmt,) = mod.body
        self.assertIsInstance(stmt.value, n.GeneratorExp)

    def test_generator_expression_as_bare_call_argument(self):
        mod = parse("total = sum(x for x in z)")
        (stmt,) = mod.body
        call = stmt.value
        self.assertIsInstance(call, n.Call)
        (arg,) = call.args
        self.assertIsInstance(arg, n.GeneratorExp)

    def test_plain_tuple_and_list_literals_unaffected_by_comprehension_support(self):
        # Regression check: comprehension disambiguation in
        # testlist_comp_tuple/testlist_comp_list must not misclassify a
        # plain multi-element literal.
        mod = parse("a = [1, 2, 3]\nb = (1, 2)\n")
        self.assertIsInstance(mod.body[0].value, n.List)
        self.assertEqual(len(mod.body[0].value.elts), 3)
        self.assertIsInstance(mod.body[1].value, n.Tuple)
        self.assertEqual(len(mod.body[1].value.elts), 2)

    def test_unsupported_construct_becomes_placeholder_statement(self):
        # lambda remains a genuinely unsupported construct: the statement
        # containing it is replaced by n.Unsupported naming the construct,
        # and the rest of the cell still parses.
        mod = parse("a = 1\nf = lambda x: x + 1\nb = 2")
        self.assertEqual([type(s).__name__ for s in mod.body], ["Assign", "Unsupported", "Assign"])
        self.assertEqual(mod.body[1].lineno, 2)
        self.assertIn("lambda", mod.body[1].construct)

    def test_victim_notebook_loads_without_raising(self):
        # Regression test for the original bug report: this real-world
        # notebook has a list comprehension (cell 68:
        # `n_estimators = [int(x) for x in np.linspace(...)]`) that used to
        # raise NotImplementedError and abort loading the whole notebook.
        cells = read_json(TEST_RES_PATH + "victim.ipynb")["cells"]
        notebook_IR = load_notebook(cells)
        self.assertGreater(len(notebook_IR), 0)
        comprehension_cell = next(
            ir for ir in notebook_IR.values() if "n_estimators" in ir.cell_code
        )
        self.assertIsInstance(comprehension_cell.AST, n.Module)


if __name__ == "__main__":
    unittest.main()
