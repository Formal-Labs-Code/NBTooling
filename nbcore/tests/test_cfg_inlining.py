import unittest

from nbcore.parser.ast_transformer import build_ast
from nbcore.parser.cfg_builder import get_cfg
from nbcore.parser.cfg_nodes import AssignmentNode, BBorBInode, RestoreNode
from nbcore.parser.lark_parser import parse_to_tree


def build_cfg(src):
    return get_cfg(build_ast(parse_to_tree(src)), "test")


class TestReducedInlining(unittest.TestCase):
    def test_inline_names_map_back_to_originals(self):
        cfg = build_cfg("def f(a):\n    b = a + 1\n    return b\nx = f(1)\nf(2)\n")
        self.assertEqual(cfg.inline_names, {
            "~inline_0_f_a": ("a", "f"),
            "~inline_0_f_b": ("b", "f"),
            "~inline_0_f___return__": ("f()", "f"),
            "~inline_1_f_a": ("a", "f"),
            "~inline_1_f_b": ("b", "f"),
            "~inline_1_f___return__": ("f()", "f"),
        })

    def test_nested_local_call_inlined_when_eligible_else_blackbox(self):
        eligible = build_cfg("def f(a):\n    return a + 1\nprint(f(2))\n")
        self.assertIn("~inline_0_f_a", eligible.inline_names)
        # *args: not inlinable yet, but nested -> black-box, not rejected
        ineligible = build_cfg("def f(*a):\n    return a\nprint(f(2))\n")
        self.assertFalse(any(k.startswith("~inline_") for k in ineligible.inline_names))
        self.assertIn("f(2)", [node.label for node in ineligible.nodes])

    def test_simple_call_is_inlined_not_blackbox(self):
        cfg = build_cfg("def f(x):\n    return x + 1\ny = f(5)\n")
        # def-node, param-bind, return-expr, restore-to-target
        labels = [type(node).__name__ for node in cfg.nodes]
        self.assertNotIn("BBorBInode", labels)
        bind, ret, restore = cfg.nodes[2], cfg.nodes[3], cfg.nodes[4]
        self.assertIsInstance(bind, AssignmentNode)
        self.assertTrue(bind.left_hand_side.endswith("_x"))
        self.assertIsInstance(restore, RestoreNode)
        self.assertEqual(restore.left_hand_side, "y")
        self.assertEqual(restore.right_hand_side, [ret.left_hand_side])

    def test_bare_call_no_assignment_no_return_required(self):
        cfg = build_cfg("def f(x):\n    print(x)\nf(5)\n")
        # def-node, param-bind, then the print() call as a normal blackbox
        from nbcore.parser.cfg_nodes import BBorBInode
        self.assertTrue(any(isinstance(node, BBorBInode) and node.func_name == "print" for node in cfg.nodes))
        self.assertFalse(any(isinstance(node, RestoreNode) for node in cfg.nodes))

    def test_two_call_sites_do_not_collide(self):
        cfg = build_cfg("def f(x):\n    return x + 1\na = f(1)\nb = f(2)\n")
        restores = [n for n in cfg.nodes if isinstance(n, RestoreNode)]
        self.assertEqual([r.left_hand_side for r in restores], ["a", "b"])
        binds = [n for n in cfg.nodes if isinstance(n, AssignmentNode) and n.left_hand_side.endswith("_x")]
        self.assertEqual(len(binds), 2)
        self.assertNotEqual(binds[0].left_hand_side, binds[1].left_hand_side)  # renamed uniquely per call

    def test_caller_variable_same_name_as_param_not_clobbered(self):
        cfg = build_cfg("def f(x):\n    return x + 1\nx = 10\ny = f(x)\nprint(x)\n")
        outer_x_assign = cfg.nodes[2]
        self.assertEqual(outer_x_assign.left_hand_side, "x")
        self.assertEqual(outer_x_assign.label, "x = 10")
        param_bind = cfg.nodes[3]
        self.assertNotEqual(param_bind.left_hand_side, "x")  # renamed, doesn't collide
        self.assertEqual(param_bind.right_hand_side, ["x"])  # but still reads the caller's x

    def test_control_flow_inside_inlined_function_works(self):
        cfg = build_cfg(
            "def f(x):\n"
            "    if x > 0:\n"
            "        y = 1\n"
            "    else:\n"
            "        y = 2\n"
            "    return y\n"
            "z = f(5)\n"
        )
        from nbcore.parser.cfg_nodes import CondNode
        self.assertTrue(any(isinstance(node, CondNode) for node in cfg.nodes))
        restore = next(n for n in cfg.nodes if isinstance(n, RestoreNode))
        self.assertEqual(restore.left_hand_side, "z")


class TestBroadenedInlining(unittest.TestCase):
    def labels(self, cfg):
        return [node.label for node in cfg.nodes[1:-1]]

    def test_keyword_arguments_and_defaults_bind_parameters(self):
        cfg = build_cfg("def f(x, y=2, *, z=3):\n    return x + y + z\nr = f(1, z=w)\n")
        self.assertEqual(self.labels(cfg)[1:4], [  # [0] is the def node
            "~inline_0_f_x = 1", "~inline_0_f_y = 2", "~inline_0_f_z = w",
        ])
        self.assertEqual(cfg.warnings, [])

    def test_nested_local_calls_are_inlined(self):
        cfg = build_cfg(
            "def load(p):\n    return read(p)\n"
            "def main():\n    X = load('a.csv')\n    return X\n"
            "main()\n"
        )
        self.assertIn("~inline_0_main_X", cfg.inline_names)
        self.assertIn("~inline_1_load_p", cfg.inline_names)
        self.assertEqual(cfg.inline_names["~inline_1_load_p"], ("p", "load"))
        self.assertIn("read(~inline_1_load_p)", self.labels(cfg))
        self.assertEqual(cfg.warnings, [])

    def test_every_return_joins_at_the_restore_node(self):
        cfg = build_cfg(
            "def f(x):\n"
            "    if x:\n"
            "        return 1\n"
            "    return 2\n"
            "y = f(c)\n"
            "print(y)\n"
        )
        restore = next(node for node in cfg.nodes if isinstance(node, RestoreNode))
        self.assertEqual(sorted(p.label for p in restore.ingoing),
                         ["~inline_0_f___return__ = 1", "~inline_0_f___return__ = 2"])
        # `return 1` doesn't fall through into `return 2`
        early = next(node for node in cfg.nodes if node.label == "~inline_0_f___return__ = 1")
        self.assertEqual(early.outgoing, [restore])

    def test_recursive_call_is_inlined_once_then_black_box(self):
        cfg = build_cfg("def f(x):\n    return f(x)\ny = f(1)\n")
        self.assertIn("~inline_0_f_x", cfg.inline_names)
        self.assertFalse(any(k.startswith("~inline_1") for k in cfg.inline_names))
        self.assertTrue(any("recursive call" in w["message"] for w in cfg.warnings))

    def test_inlining_depth_is_limited(self):
        chain = "".join(f"def f{i}(x):\n    return f{i + 1}(x)\n" for i in range(12))
        cfg = build_cfg(chain + "def f12(x):\n    return x\ny = f0(1)\n")
        depths = {k.split("_")[2] for k in cfg.inline_names if k.startswith("~inline_")}
        self.assertEqual(len(depths), 8)
        self.assertTrue(any("more than 8" in w["message"] for w in cfg.warnings))


class TestUninlinableCallsDegrade(unittest.TestCase):
    """A local call outside the inlining scope is treated as a black-box
    call plus a CFG warning naming why, instead of rejecting the cell."""

    def assert_degraded(self, src, reason):
        cfg = build_cfg(src)
        self.assertFalse([k for k in cfg.inline_names if k.startswith("~inline_")], "should not be inlined")
        self.assertTrue(any(isinstance(node, BBorBInode) and node.func_name == "f" for node in cfg.nodes),
                        "the call should be a black-box call node")
        messages = [w["message"] for w in cfg.warnings]
        self.assertTrue(any(reason in m and "opaque call" in m for m in messages), messages)

    def test_star_args_parameter(self):
        self.assert_degraded("def f(*args):\n    return args\ny = f(1, 2)\n", "*args/**kwargs")

    def test_star_unpacking_at_call_site(self):
        self.assert_degraded("def f(x):\n    return x\ny = f(*a)\n", "unpacking at the call site")

    def test_missing_argument(self):
        self.assert_degraded("def f(x, y):\n    return x\nz = f(1)\n", "missing argument 'y'")

    def test_too_many_arguments(self):
        self.assert_degraded("def f(x):\n    return x\nz = f(1, 2)\n", "positional argument(s) but")

    def test_unknown_keyword(self):
        self.assert_degraded("def f(x):\n    return x\nz = f(q=1)\n", "keyword argument 'q'")

    def test_tuple_unpack_of_result(self):
        self.assert_degraded("def f(x):\n    return x\na, b = f(1)\n", "tuple/multi-target unpacking")

    def test_void_function_result_assigned(self):
        self.assert_degraded("def f(x):\n    print(x)\ny = f(5)\n", "never returns a value")

    def test_warning_carries_line(self):
        cfg = build_cfg("a = 1\ndef f(*xs):\n    return xs\nb = f(a)\n")
        self.assertEqual([w["line"] for w in cfg.warnings], [4])


class TestUnsupportedStatementsDegrade(unittest.TestCase):
    def test_unsupported_statement_is_a_noop_with_warning(self):
        cfg = build_cfg("x = 1\ndel x\ny = 2\n")
        labels = [node.label for node in cfg.nodes[1:-1]]
        self.assertEqual(labels, ["x = 1", "<unsupported>", "y = 2"])
        self.assertEqual([w["line"] for w in cfg.warnings], [2])
        self.assertIn("del_stmt", cfg.warnings[0]["message"])

    def test_unsupported_statement_in_function_body_only_skips_that_statement(self):
        cfg = build_cfg("def f(a):\n    del a\n    b = a\n    return b\nz = f(1)\n")
        self.assertIn("~inline_0_f_a", cfg.inline_names)  # f is still inlined
        self.assertIn("~inline_0_f_b = ~inline_0_f_a", [node.label for node in cfg.nodes])


if __name__ == "__main__":
    unittest.main()
