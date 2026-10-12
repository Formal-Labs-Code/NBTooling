import unittest

from nbcore.parser.ast_transformer import build_ast
from nbcore.parser.cfg_builder import get_cfg
from nbcore.parser.cfg_nodes import AssignmentNode, RestoreNode
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
        # default argument: not inlinable yet, but nested -> black-box, not rejected
        ineligible = build_cfg("def f(a, b=1):\n    return a + b\nprint(f(2))\n")
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


class TestReducedInliningRejections(unittest.TestCase):
    def assert_rejected(self, src):
        with self.assertRaises(NotImplementedError):
            build_cfg(src)

    def test_rejects_keyword_argument(self):
        self.assert_rejected("def f(x):\n    return x\ny = f(x=1)\n")

    def test_rejects_default_parameter(self):
        self.assert_rejected("def f(x, y=2):\n    return x\ny = f(1)\n")

    def test_rejects_star_args_parameter(self):
        self.assert_rejected("def f(*args):\n    return args\ny = f(1, 2)\n")

    def test_rejects_wrong_arg_count(self):
        self.assert_rejected("def f(x, y):\n    return x\nz = f(1)\n")

    def test_rejects_recursion(self):
        self.assert_rejected("def f(x):\n    return f(x)\ny = f(1)\n")

    def test_rejects_tuple_unpack_of_result(self):
        self.assert_rejected("def f(x):\n    return x\na, b = f(1)\n")

    def test_rejects_non_trailing_or_multiple_return(self):
        self.assert_rejected(
            "def f(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "    y = 2\n"
            "    return y\n"
            "z = f(1)\n"
        )

    def test_rejects_void_function_result_assigned(self):
        self.assert_rejected("def f(x):\n    print(x)\ny = f(5)\n")


if __name__ == "__main__":
    unittest.main()
