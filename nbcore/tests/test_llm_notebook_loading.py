import unittest

from nbcore.llm.notebook_loading import load_notebook_resilient


class TestLoadNotebookResilient(unittest.TestCase):
    def test_all_cells_parse_normally(self):
        cells = [
            {"cell_type": "code", "source": "x = 1"},
            {"cell_type": "code", "source": "y = x + 1"},
        ]
        notebook_IR = load_notebook_resilient(cells)
        self.assertEqual(set(notebook_IR.keys()), {0, 1})
        self.assertEqual(notebook_IR[1].cell_code, "y = x + 1")

    def test_unparsable_cell_is_skipped_not_fatal(self):
        cells = [
            {"cell_type": "code", "source": "x = 1"},
            {"cell_type": "code", "source": "x = = 1"},  # a genuine syntax error
            {"cell_type": "code", "source": "y = x + 1"},
        ]
        notebook_IR = load_notebook_resilient(cells)

        self.assertEqual(set(notebook_IR.keys()), {0, 2})
        self.assertEqual(notebook_IR[2].cell_code, "y = x + 1")

    def test_unsupported_construct_loads_with_a_warning(self):
        # lambda isn't supported by ast_transformer.py yet: its statement is
        # a no-op placeholder and the cell loads, with a CFG warning.
        cells = [{"cell_type": "code", "source": "f = lambda n: n * n\ny = 1"}]
        notebook_IR = load_notebook_resilient(cells)

        self.assertEqual(set(notebook_IR.keys()), {0})
        self.assertEqual([w["line"] for w in notebook_IR[0].CFG.warnings], [1])

    def test_non_code_and_empty_cells_are_skipped(self):
        cells = [
            {"cell_type": "markdown", "source": "# heading"},
            {"cell_type": "code", "source": ""},
            {"cell_type": "code", "source": "x = 1"},
        ]
        notebook_IR = load_notebook_resilient(cells)
        self.assertEqual(set(notebook_IR.keys()), {2})

    def test_empty_notebook(self):
        self.assertEqual(load_notebook_resilient([]), {})
        self.assertEqual(load_notebook_resilient(None), {})


if __name__ == "__main__":
    unittest.main()
