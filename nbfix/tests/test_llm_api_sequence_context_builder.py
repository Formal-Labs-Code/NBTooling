import unittest

from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbfix.llm.api_sequence_context_builder import build_api_sequence_context


def make_notebook(cells: dict[int, tuple]) -> dict:
    """cells: cell_id -> (current_code, last_ran_code)"""
    return {
        cell_id: IntermediateRepresentations(current, cell_id, last_ran)
        for cell_id, (current, last_ran) in cells.items()
    }


class TestBuildApiSequenceContext(unittest.TestCase):
    def test_captures_every_cells_current_code(self):
        notebook_IR = make_notebook({
            0: ("model = M()", "model = M()"),
            1: ("model.predict(X)", "model.predict(X)"),
        })
        context = build_api_sequence_context(notebook_IR)
        self.assertEqual(context.cells, {0: "model = M()", 1: "model.predict(X)"})

    def test_never_run_cell_is_marked_not_yet_run(self):
        """
        last_ran_code defaults to "" for a cell that's never been
        executed - must not be confused with an actually-run cell whose
        code happens to differ. Uses a method call (model.fit) so the
        component survives the has-a-method-call filter - this test is
        about not_yet_run, not about filtering.
        """
        notebook_IR = make_notebook({0: ("model.fit(X)", "")})
        context = build_api_sequence_context(notebook_IR)
        self.assertEqual(context.not_yet_run, {0})

    def test_edited_since_last_run_is_marked_not_yet_run(self):
        """
        A cell that WAS run, but has since been edited without being
        re-run, has NOT actually executed its current code - the code
        sitting in the kernel is still the old version.
        """
        notebook_IR = make_notebook({0: ("model.fit(X, y)  # added fit", "model = M()")})
        context = build_api_sequence_context(notebook_IR)
        self.assertEqual(context.not_yet_run, {0})

    def test_run_with_current_code_is_not_marked(self):
        notebook_IR = make_notebook({0: ("model.fit(X)", "model.fit(X)")})
        context = build_api_sequence_context(notebook_IR)
        self.assertEqual(context.not_yet_run, set())

    def test_focus_cell_none_includes_every_cell(self):
        notebook_IR = make_notebook({
            0: ("model.fit(X)", "model.fit(X)"),
            1: ("model.predict(Y)", "model.predict(Y)"),
        })
        context = build_api_sequence_context(notebook_IR, focus_cell=None)
        self.assertEqual(set(context.cells), {0, 1})

    def test_focus_cell_narrows_to_connected_component(self):
        """
        Cell 2 (unrelated - defines its own name, reads nothing from
        cell 0/1) must not be pulled into a scan focused on cell 0's
        component, even though it exists in the same notebook.
        """
        notebook_IR = make_notebook({
            0: ("model = M()", "model = M()"),
            1: ("model.fit(X)", "model.fit(X)"),
            2: ("other.run()", "other.run()"),
        })
        context = build_api_sequence_context(notebook_IR, focus_cell=0)
        self.assertEqual(set(context.cells), {0, 1})
        self.assertNotIn(2, context.cells)

    def test_focus_cell_component_still_reports_not_yet_run(self):
        notebook_IR = make_notebook({
            0: ("model = M()", "model = M()"),
            1: ("model.fit(X)", ""),
        })
        context = build_api_sequence_context(notebook_IR, focus_cell=0)
        self.assertEqual(context.not_yet_run, {1})


class TestMethodCallFilter(unittest.TestCase):
    """
    Regression coverage for the explicit user-requested optimization:
    decompose the notebook into its dependency-connected components and
    skip sending a component to the LLM at all if none of its cells
    contain a method call (obj.method(...)) anywhere - this bug class is
    inherently about stateful-object call order, so a component built
    entirely from plain assignments/builtins provably cannot violate it.
    """

    def test_full_scan_drops_a_component_with_no_method_calls(self):
        notebook_IR = make_notebook({
            0: ("x = 1", "x = 1"),
            1: ("y = x + 1", "y = x + 1"),
            2: ("model = M()", "model = M()"),
            3: ("model.predict(X)", "model.predict(X)"),
        })
        context = build_api_sequence_context(notebook_IR, focus_cell=None)
        self.assertEqual(set(context.cells), {2, 3})
        self.assertNotIn(0, context.cells)
        self.assertNotIn(1, context.cells)

    def test_full_scan_considered_cells_includes_filtered_out_ones(self):
        """
        A filtered-out component's "no violation" answer is confidently
        known without asking the LLM, so it still counts as considered -
        the caller needs this to correctly clear any stale finding on
        those cells too (see DetectApiSequenceEvent.checked_cells).
        """
        notebook_IR = make_notebook({
            0: ("x = 1", "x = 1"),
            1: ("model.predict(X)", "model.predict(X)"),
        })
        context = build_api_sequence_context(notebook_IR, focus_cell=None)
        self.assertEqual(context.considered_cells, {0, 1})
        self.assertEqual(set(context.cells), {1})

    def test_focus_cell_component_with_no_method_calls_yields_empty_cells(self):
        notebook_IR = make_notebook({
            0: ("x = 1", "x = 1"),
            1: ("y = x + 1", "y = x + 1"),
        })
        context = build_api_sequence_context(notebook_IR, focus_cell=0)
        self.assertEqual(context.cells, {})
        self.assertEqual(context.considered_cells, {0, 1})

    def test_component_with_call_on_only_one_cell_keeps_the_whole_component(self):
        """
        Filtering is per-component, not per-cell - a plain setup cell
        (no call of its own) must still be included if some OTHER cell
        in its component has a method call, since the model needs to see
        the setup code to reason about the call.
        """
        notebook_IR = make_notebook({
            0: ("model = M()", "model = M()"),
            1: ("model.fit(X)", "model.fit(X)"),
        })
        context = build_api_sequence_context(notebook_IR, focus_cell=None)
        self.assertEqual(set(context.cells), {0, 1})

    def test_constructor_call_alone_does_not_count(self):
        """
        `StandardScaler()` is a call, but not an attribute call on an
        object - a bare constructor/function call doesn't establish
        relevance on its own (only obj.method(...) does).
        """
        notebook_IR = make_notebook({0: ("scaler = StandardScaler()", "scaler = StandardScaler()")})
        context = build_api_sequence_context(notebook_IR, focus_cell=None)
        self.assertEqual(context.cells, {})


if __name__ == "__main__":
    unittest.main()
