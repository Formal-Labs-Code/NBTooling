import unittest

from nbcore.analyses import dependency_analysis
from nbcore.analyses.dependency_analysis import build_fixpoint_dependency_edges
from nbcore.ir.intermediate_representations import IntermediateRepresentations


def make_notebook(cells: dict[int, str]) -> dict:
    return {
        cell_id: IntermediateRepresentations(code, cell_id)
        for cell_id, code in cells.items()
    }


class TestBuildFixpointDependencyEdges(unittest.TestCase):
    def test_no_dependencies(self):
        notebook_IR = make_notebook({0: "x = 1", 1: "z = 2"})
        edges = build_fixpoint_dependency_edges(notebook_IR)
        self.assertEqual(edges, {0: set(), 1: set()})

    def test_direct_dependency_when_order_respected(self):
        notebook_IR = make_notebook({0: "x = 1", 1: "y = x + 1"})
        edges = build_fixpoint_dependency_edges(notebook_IR)
        self.assertEqual(edges, {0: set(), 1: {0}})

    def test_finds_a_backward_read_before_defined_edge(self):
        """
        The actual gap this module exists to close: a single-pass,
        ID-ordered walk (the kind context_builder.build_dependency_edges
        used to be, before it became a plain alias for this function -
        see that module's comment) can only ever record an edge pointing
        *backward* to an already-visited definer - a cell reading a name
        only defined in a later cell is structurally invisible to it.
        This is exactly the `order_dependent` bug class's shape (see
        experiments.md findings 10-11).
        """
        notebook_IR = make_notebook({0: "y = x + 1", 1: "x = 1"})
        edges = build_fixpoint_dependency_edges(notebook_IR)
        self.assertEqual(edges, {0: {1}, 1: set()})

    def test_same_cell_read_before_write_reassignment_still_seeds_correctly(self):
        """
        Regression guard for a real bug found while building this: a cell
        like `data = len(data)` compiles to a Call CFG node followed by
        an Assign CFG node. Seeding the fixpoint from a truly empty
        abstract state let intra_fixpoint_runner's "did the state change"
        convergence check treat the no-op Call node as "nothing changed"
        and never reach the Assign node at all - silently losing the
        cell's own definition. Fixed by pre-seeding with the cell's own
        UDA.defined_vars (mirroring StaleCellAnalysis._prepare_init_as).
        """
        notebook_IR = make_notebook({
            0: "data = [1, 2, 3]",
            1: "data = len(data)",
            2: "data.append(4)",
        })
        edges = build_fixpoint_dependency_edges(notebook_IR)
        self.assertIn(1, edges[2])  # depends on the reassignment...
        self.assertIn(0, edges[2])  # ...and the original definition

    def test_never_produces_a_self_edge(self):
        notebook_IR = make_notebook({0: "x = x + 1"})
        edges = build_fixpoint_dependency_edges(notebook_IR)
        self.assertEqual(edges, {0: set()})

    def test_definer_provenance_affects_state_equality_and_contains(self):
        left = dependency_analysis.DependencyAS({"x": 0})
        right = dependency_analysis.DependencyAS({"x": 1})
        self.assertNotEqual(left, right)
        self.assertFalse(left.contains(right))

    def test_deeper_k_does_not_misattribute_a_transitive_chain(self):
        """
        Regression guard for a real bug found while investigating whether
        raising _DIRECT_EDGE_DEPTH beyond 2 would change anything: without
        per-name provenance tracking in DependencyAS, a cell several hops
        downstream got attributed back to the *original seed* for every
        name reachable along the path, not just the ones the seed itself
        defines. Here, seeding from cell 0 and propagating with K=6 must
        still report cell 2 as depending on cell 1 (which actually defines
        y) - not cell 0, which cell 2 never reads at all.
        """
        notebook_IR = make_notebook({0: "x = 1", 1: "y = x + 1", 2: "z = y + 1"})
        original_depth = dependency_analysis._DIRECT_EDGE_DEPTH
        try:
            dependency_analysis._DIRECT_EDGE_DEPTH = 6
            edges = dependency_analysis.build_fixpoint_dependency_edges(notebook_IR)
        finally:
            dependency_analysis._DIRECT_EDGE_DEPTH = original_depth
        self.assertEqual(edges, {0: set(), 1: {0}, 2: {1}})

    def test_a_genuine_cross_cell_cycle_resolves_without_hanging(self):
        """
        A mutual dependency (cell 0 reads b, defined by cell 1; cell 1
        reads a, defined by cell 0) exercises inter_fixpoint_runner's own
        cycle-detection (the `contains()` check already proven safe by
        StaleCellAnalysis) rather than anything specific to this module -
        confirms that mechanism still applies correctly once seeding
        happens from every cell independently, and that raising K past 2
        doesn't change the result even in the presence of a real cycle.
        """
        notebook_IR = make_notebook({0: "a = b", 1: "b = a"})
        original_depth = dependency_analysis._DIRECT_EDGE_DEPTH
        try:
            edges_k2 = dependency_analysis.build_fixpoint_dependency_edges(notebook_IR)
            dependency_analysis._DIRECT_EDGE_DEPTH = 8
            edges_k8 = dependency_analysis.build_fixpoint_dependency_edges(notebook_IR)
        finally:
            dependency_analysis._DIRECT_EDGE_DEPTH = original_depth
        self.assertEqual(edges_k2, {0: {1}, 1: {0}})
        self.assertEqual(edges_k2, edges_k8)


if __name__ == "__main__":
    unittest.main()
