import unittest

from nbcore.resource_utils.script_loading import (
    cell_start_lines, file_line, first_code_cell, has_cell_markers, is_script, script_to_notebook,
)
from nbcore.resource_utils.utils import load_notebook


class TestScriptLoading(unittest.TestCase):
    def test_script_without_markers_is_one_cell(self):
        nb = script_to_notebook("import os\n\nx = 1\ny = x\n")
        self.assertTrue(is_script(nb))
        self.assertFalse(has_cell_markers(nb))
        self.assertEqual([c["cell_type"] for c in nb["cells"]], ["code"])
        self.assertEqual(cell_start_lines(nb), {0: 1})

    def test_percent_markers_split_cells_and_markdown_is_skipped(self):
        src = (
            "#!/usr/bin/env python\n"      # 1  preamble: comments only, not a cell
            "# %% [markdown]\n"            # 2
            "# Title\n"                    # 3
            "# %%\n"                       # 4
            "\n"                           # 5  leading blank line dropped
            "x = 1\n"                      # 6
            "# %% load data\n"             # 7
            "y = x\n"                      # 8
        )
        nb = script_to_notebook(src)
        self.assertTrue(has_cell_markers(nb))
        self.assertEqual([c["cell_type"] for c in nb["cells"]], ["markdown", "code", "code"])
        self.assertEqual([c["source"] for c in nb["cells"][1:]], ["x = 1", "y = x"])
        self.assertEqual(cell_start_lines(nb), {0: 3, 1: 6, 2: 8})
        self.assertEqual(first_code_cell(nb), 1)
        # load_notebook numbers cells by position, skipping the markdown one
        self.assertEqual(sorted(load_notebook(nb["cells"])), [1, 2])

    def test_jupyter_export_in_markers(self):
        src = "#!/usr/bin/env python\n# coding: utf-8\n\n# In[1]:\n\n\nimport os\n\n\n# In[ ]:\n\n\nprint(os.sep)\n"
        nb = script_to_notebook(src)
        self.assertTrue(has_cell_markers(nb))
        self.assertEqual([c["source"].strip() for c in nb["cells"]], ["import os", "print(os.sep)"])
        self.assertEqual(cell_start_lines(nb), {0: 7, 1: 13})

    def test_file_line_maps_cell_relative_lines(self):
        starts = {1: 6, 2: 8}
        self.assertEqual(file_line(starts, 1, 1), 6)
        self.assertEqual(file_line(starts, 2, 3), 10)
        self.assertIsNone(file_line(starts, 0, 1))
        self.assertIsNone(file_line(starts, 1, 0))

    def test_cell_line_numbers_match_file_after_mapping(self):
        # end to end through the IR: a node's line, mapped, is its file line
        src = "# %%\n\n\na = 1\n\nb = a\n"
        nb = script_to_notebook(src)
        cells = load_notebook(nb["cells"])
        starts = cell_start_lines(nb)
        lines = {n.label: file_line(starts, 0, n.line_number) for n in cells[0].CFG.nodes[1:-1]}
        self.assertEqual(lines, {"a = 1": 4, "b = a": 6})


if __name__ == "__main__":
    unittest.main()
