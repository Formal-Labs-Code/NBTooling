import unittest

from nbcore.ir.intermediate_representations import IntermediateRepresentations

from nbcompile.converter import (
    ScriptBlock,
    convert_to_script,
    find_sccs,
    find_start_cells,
    linearize,
    render_script,
)


def make_notebook(cells: dict[int, str]) -> dict:
    return {
        cell_id: IntermediateRepresentations(code, cell_id)
        for cell_id, code in cells.items()
    }


class TestFindStartCells(unittest.TestCase):
    def test_cells_with_no_deps_are_start_cells(self):
        edges = {0: set(), 1: {0}, 2: {0}}
        self.assertEqual(find_start_cells(edges), [0])

    def test_multiple_start_cells(self):
        edges = {0: set(), 1: set(), 2: {0, 1}}
        self.assertEqual(find_start_cells(edges), [0, 1])

    def test_no_start_cells_in_a_pure_cycle(self):
        edges = {0: {1}, 1: {0}}
        self.assertEqual(find_start_cells(edges), [])


class TestFindSccs(unittest.TestCase):
    def test_linear_chain_is_all_singletons(self):
        edges = {0: set(), 1: {0}, 2: {1}}
        sccs = find_sccs(edges)
        self.assertEqual(sorted(sccs), [[0], [1], [2]])

    def test_mutual_dependency_is_one_scc(self):
        edges = {0: {1}, 1: {0}}
        sccs = find_sccs(edges)
        self.assertEqual(sccs, [[0, 1]])

    def test_three_cell_cycle_is_one_scc(self):
        edges = {0: {2}, 1: {0}, 2: {1}}
        sccs = find_sccs(edges)
        self.assertEqual(sccs, [[0, 1, 2]])

    def test_cycle_with_external_dependencies_stays_separate(self):
        # 0 has no deps and feeds the cycle {1, 2}; 3 depends on the cycle.
        edges = {0: set(), 1: {0, 2}, 2: {1}, 3: {2}}
        sccs = find_sccs(edges)
        self.assertIn([0], sccs)
        self.assertIn([1, 2], sccs)
        self.assertIn([3], sccs)
        self.assertEqual(len(sccs), 3)


class TestLinearize(unittest.TestCase):
    def test_linear_chain_preserves_dependency_order(self):
        edges = {0: set(), 1: {0}, 2: {1}}
        blocks = linearize(edges)
        order = [b.cell_ids[0] for b in blocks]
        self.assertEqual(order, [0, 1, 2])
        self.assertTrue(all(not b.is_cycle for b in blocks))

    def test_branch_respects_partial_order_not_exact_order(self):
        # 0 and 1 both feed 2 - only relative order (0 before 2, 1 before 2)
        # is guaranteed, not which of 0/1 comes first (the deferred
        # "many subchains" ordering problem).
        edges = {0: set(), 1: set(), 2: {0, 1}}
        blocks = linearize(edges)
        flat = [cid for b in blocks for cid in b.cell_ids]
        self.assertLess(flat.index(0), flat.index(2))
        self.assertLess(flat.index(1), flat.index(2))

    def test_two_cell_cycle_becomes_one_cycle_block(self):
        edges = {0: {1}, 1: {0}}
        blocks = linearize(edges)
        self.assertEqual(len(blocks), 1)
        self.assertTrue(blocks[0].is_cycle)
        self.assertEqual(blocks[0].cell_ids, [0, 1])

    def test_cycle_stays_correctly_ordered_relative_to_its_neighbors(self):
        # 0 (no deps) feeds the {1, 2} cycle; 3 depends on the cycle.
        edges = {0: set(), 1: {0, 2}, 2: {1}, 3: {2}}
        blocks = linearize(edges)
        self.assertEqual(
            [ScriptBlock(cell_ids=[0], is_cycle=False),
             ScriptBlock(cell_ids=[1, 2], is_cycle=True),
             ScriptBlock(cell_ids=[3], is_cycle=False)],
            blocks,
        )


class TestRenderScript(unittest.TestCase):
    def test_non_cycle_blocks_emit_code_as_is(self):
        notebook_IR = make_notebook({0: "x = 1", 1: "y = x + 1"})
        blocks = [ScriptBlock([0], False), ScriptBlock([1], False)]
        script = render_script(notebook_IR, blocks)
        self.assertEqual(script, "x = 1\n\ny = x + 1")

    def test_cycle_block_gets_a_while_true_wrapper_with_a_todo(self):
        notebook_IR = make_notebook({0: "a = b", 1: "b = a"})
        blocks = [ScriptBlock([0, 1], True)]
        script = render_script(notebook_IR, blocks)
        self.assertIn("while True:", script)
        self.assertIn("TODO", script)
        self.assertIn("    a = b", script)
        self.assertIn("    b = a", script)


class TestConvertToScript(unittest.TestCase):
    def test_end_to_end_linear_notebook(self):
        notebook_IR = make_notebook({0: "x = 1", 1: "y = x + 1", 2: "z = y + 1"})
        script = convert_to_script(notebook_IR)
        self.assertEqual(script, "x = 1\n\ny = x + 1\n\nz = y + 1")

    def test_end_to_end_with_a_real_cycle(self):
        # Same fixture as nbcore's own
        # test_a_genuine_cross_cell_cycle_resolves_without_hanging -
        # confirmed to actually produce mutual edges, not a hypothetical.
        notebook_IR = make_notebook({0: "a = b", 1: "b = a"})
        script = convert_to_script(notebook_IR)
        self.assertIn("while True:", script)
        self.assertIn("TODO", script)


if __name__ == "__main__":
    unittest.main()
