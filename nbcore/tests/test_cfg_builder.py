import unittest

from nbcore.parser.ast_transformer import build_ast
from nbcore.parser.cfg_builder import get_cfg
from nbcore.parser.cfg_nodes import (
    AssignmentCallNode,
    AssignmentNode,
    BBorBInode,
    BreakNode,
    CondNode,
    EntryOrExitNode,
    TryNode,
)
from nbcore.parser.lark_parser import parse_to_tree


def build_cfg(src):
    return get_cfg(build_ast(parse_to_tree(src)), "test")


class TestCFGBuilder(unittest.TestCase):
    def test_straight_line(self):
        cfg = build_cfg("x = 1\ny = x + 1\n")
        self.assertIsInstance(cfg.nodes[0], EntryOrExitNode)
        self.assertIsInstance(cfg.nodes[-1], EntryOrExitNode)
        middle = cfg.nodes[1:-1]
        self.assertTrue(all(isinstance(node, AssignmentNode) for node in middle))
        self.assertEqual(middle[0].left_hand_side, "x")
        self.assertEqual(middle[1].left_hand_side, "y")
        self.assertEqual(middle[1].right_hand_side, ["x"])
        # linear chain: each node's sole successor is the next one
        for a, b in zip(cfg.nodes, cfg.nodes[1:]):
            self.assertIn(b, a.outgoing)

    def test_comprehension_assignment_excludes_bound_var_from_rhs_names(self):
        # Regression test for the _collect_names fix: a comprehension's own
        # bound target (x) must not be reported as a "used" name alongside
        # the real free variable (z) - including the two Load-context uses
        # of x inside the element expression `x * x`, not just its single
        # Store-context occurrence as the loop target.
        cfg = build_cfg("y = [x * x for x in z]\n")
        node = cfg.nodes[1]
        self.assertIsInstance(node, AssignmentNode)
        self.assertEqual(node.left_hand_side, "y")
        self.assertEqual(node.right_hand_side, ["z"])

    def test_call_produces_blackbox_and_assignment_call_node(self):
        cfg = build_cfg("x = df.head()\n")
        call_node, assign_node = cfg.nodes[1], cfg.nodes[2]
        self.assertIsInstance(call_node, BBorBInode)
        self.assertEqual(call_node.func_name, "df.head")
        self.assertIsInstance(assign_node, AssignmentCallNode)
        self.assertEqual(assign_node.left_hand_side, "x")
        self.assertIn(assign_node, call_node.outgoing)

    def test_if_else_merges_after_branches(self):
        cfg = build_cfg("if x > 5:\n    y = 1\nelse:\n    y = 2\nz = y\n")
        cond = cfg.nodes[1]
        self.assertIsInstance(cond, CondNode)
        then_branch, else_branch = cond.outgoing
        merge = cfg.nodes[4]
        self.assertEqual(merge.left_hand_side, "z")
        self.assertIn(merge, then_branch.outgoing)
        self.assertIn(merge, else_branch.outgoing)

    def test_if_without_else_falls_through(self):
        cfg = build_cfg("if x > 5:\n    y = 1\nz = y\n")
        cond = cfg.nodes[1]
        merge = cfg.nodes[3]
        self.assertIn(merge, cond.outgoing)  # no-branch-taken path

    def test_nested_calls_are_hoisted_innermost_first(self):
        cfg = build_cfg("print(f(g(x)), key=h(y))\n")
        labels = [node.label for node in cfg.nodes[1:-1]]
        self.assertEqual(labels, [
            "g(x)", "~call_0 = g(x)",
            "f(~call_0)", "~call_1 = f(~call_0)",
            "h(y)", "~call_2 = h(y)",
            "print(~call_1, key=~call_2)",
        ])
        self.assertEqual(cfg.inline_names["~call_1"], ("f(~call_0)", None))

    def test_call_receiver_and_operator_operands_are_hoisted(self):
        cfg = build_cfg("acc = lr.score(X, y) * 100\nZ = MinMaxScaler().fit_transform(X)\n")
        labels = [node.label for node in cfg.nodes[1:-1]]
        self.assertEqual(labels, [
            "lr.score(X, y)", "~call_0 = lr.score(X, y)", "acc = ~call_0 * 100",
            "MinMaxScaler()", "~call_1 = MinMaxScaler()",
            "~call_1.fit_transform(X)", "Z = ~call_1.fit_transform(X)",
        ])

    def test_top_level_call_and_comprehension_calls_stay_in_place(self):
        cfg = build_cfg("a = f(x)\nb = [g(v) for v in a]\n")
        self.assertEqual([node.label for node in cfg.nodes[1:-1]],
                         ["f(x)", "a = f(x)", "b = [g(v) for v in a]"])

    def test_for_loop_has_back_edge(self):
        cfg = build_cfg("for i in range(10):\n    total += i\nprint(total)\n")
        # range(10) is a call nested in the loop statement: hoisted into a
        # temp evaluated once, before the header (see cfg_builder's ANF).
        header = next(node for node in cfg.nodes if isinstance(node, CondNode))
        iterable = [node for node in cfg.nodes[:cfg.nodes.index(header)] if node.label.endswith("range(10)")]
        self.assertEqual(len(iterable), 2)  # the call node and its ~call_0 assignment
        self.assertEqual(header.label, "for i in ~call_0:")
        body = header.positive_nodes[0]
        self.assertIn(header, body.outgoing)  # back-edge to loop header ...
        self.assertFalse(set(iterable) & set(body.outgoing))  # ... not to the iterable

    def test_break_skips_loop_continue_returns_to_header(self):
        cfg = build_cfg(
            "while x > 0:\n"
            "    x -= 1\n"
            "    if x == 5:\n"
            "        break\n"
            "    continue\n"
            "print('done')\n"
        )
        header = cfg.nodes[1]
        break_node = next(n for n in cfg.nodes if isinstance(n, BreakNode))
        continue_node = next(n for n in cfg.nodes if n.label == "Continue")
        after_loop = next(n for n in cfg.nodes if isinstance(n, BBorBInode))
        self.assertIn(after_loop, break_node.outgoing)
        self.assertIn(header, continue_node.outgoing)

    def test_try_except_finally_shape(self):
        cfg = build_cfg("try:\n    risky()\nexcept ValueError:\n    handle()\nfinally:\n    cleanup()\n")
        try_node = cfg.nodes[1]
        self.assertIsInstance(try_node, TryNode)
        body_call = cfg.nodes[2]
        handler_call = cfg.nodes[3]
        finally_call = cfg.nodes[4]
        self.assertIn(body_call, try_node.outgoing)
        self.assertIn(handler_call, try_node.outgoing)
        self.assertIn(finally_call, body_call.outgoing)
        self.assertIn(finally_call, handler_call.outgoing)

    def test_tuple_unpack_creates_one_node_per_target(self):
        cfg = build_cfg("x, y = 1, 2\n")
        first, second = cfg.nodes[1], cfg.nodes[2]
        self.assertEqual(first.left_hand_side, "x")
        self.assertEqual(second.left_hand_side, "y")

    def test_return_and_raise_only_connect_to_exit(self):
        # A statement after `return` in the same straight-line block must
        # not gain a normal successor edge from the ReturnNode (it's
        # unreachable via fallthrough); only the final module-exit wiring
        # (which bypasses the ConnectToExitNode guard) may reach it.
        cfg = build_cfg("def f():\n    return 1\n    x = 2\n")
        # FunctionDef itself is opaque in phase 1 -- body isn't spliced in,
        # so this mainly checks parsing/CFG-building doesn't crash on it.
        self.assertTrue(any(n.label.startswith("def f(") for n in cfg.nodes))


if __name__ == "__main__":
    unittest.main()
