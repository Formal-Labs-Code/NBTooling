import unittest

from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbfix.llm.stale_context_builder import build_stale_context


def make_notebook(cells: dict[int, tuple]) -> dict:
    """cells: cell_id -> (current_code, last_ran_code)"""
    return {
        cell_id: IntermediateRepresentations(current, cell_id, last_ran)
        for cell_id, (current, last_ran) in cells.items()
    }


class TestBuildStaleContext(unittest.TestCase):
    def test_captures_edited_cell_and_all_current_code(self):
        notebook_IR = make_notebook({
            0: ("x = 99", "x = 10"),
            1: ("y = x + 1", "y = x + 1"),
        })
        context = build_stale_context(notebook_IR, 0, original_code="x = 10")
        self.assertEqual(context.edited_cell, 0)
        self.assertEqual(context.original_code, "x = 10")
        self.assertEqual(context.current_code, "x = 99")
        self.assertEqual(context.cells, {0: "x = 99", 1: "y = x + 1"})

    def test_original_code_comes_from_the_caller_not_last_ran_code(self):
        """
        Regression guard for a real ordering bug found via a live
        end-to-end check: last_ran_code can't be read internally here,
        because events.RunCellEvent overwrites it to match cell_code as
        part of the same call that actually executes the cell - by the
        time a cell's kernel value is genuinely fresh, the diff this
        context needs is already gone from last_ran_code. The caller
        must capture it beforehand and pass it in explicitly.
        """
        notebook_IR = make_notebook({0: ("x = 99", "x = 99")})  # last_ran_code already reset
        context = build_stale_context(notebook_IR, 0, original_code="x = 10")
        self.assertEqual(context.original_code, "x = 10")


if __name__ == "__main__":
    unittest.main()
