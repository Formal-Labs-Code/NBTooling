import unittest

from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

from nbharness.mcp_server import analyze


class TestNBHarnessMCPServer(unittest.TestCase):
    def test_analyze_detects_known_data_leak(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_true.ipynb")

        output = analyze(notebook)

        self.assertEqual(set(output), {"findings", "warnings"})
        self.assertEqual(output["warnings"], [])
        result = output["findings"]
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        finding = result[0]
        self.assertEqual(finding["leakage_type"], "Data Leak Analysis")
        self.assertEqual(finding["sink"], "X_selected_test")
        self.assertEqual(
            finding["source"],
            {"cell": 1, "line": 3, "call": "min_max_scaler.fit_transform", "variable": "X"},
        )
        self.assertEqual(finding["path"], [0, 1, 2, 3, 4])
        self.assertEqual(
            finding["locations"],
            [{"role": "sink", "cell": 4, "line": 2}, {"role": "source", "cell": 1, "line": 3}],
        )
        self.assertTrue(finding["message"].startswith("Preprocessing leak: `X_selected_test` passed to `lr.predict`"))
        self.assertIn("`min_max_scaler.fit_transform` (cell 1, line 3)", finding["message"])

    def test_analyze_reports_no_findings_for_clean_notebook(self):
        self.assertEqual(analyze(read_json(TEST_RES_PATH + "dataleak_false.ipynb")), {"findings": [], "warnings": []})

    def test_analyze_script_path(self):
        result = analyze(TEST_RES_PATH + "dataleak_script_flat_true.py")["findings"]
        self.assertEqual(len(result), 1)
        finding = result[0]
        self.assertIsNone(finding["path"])  # single-cell script: no cell path
        self.assertEqual(
            finding["locations"],
            [{"role": "sink", "cell": 0, "line": 17, "file_line": 17},
             {"role": "source", "cell": 0, "line": 11, "file_line": 11}],
        )

    def test_analyze_script_with_cells_starts_at_first_code_cell(self):
        result = analyze(TEST_RES_PATH + "dataleak_script_cells_true.py")["findings"]
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["path"], [1, 2, 3, 4, 5])
        self.assertEqual(result[0]["locations"][0]["file_line"], 33)

    def test_analyze_script_source_string_and_object(self):
        with open(TEST_RES_PATH + "dataleak_script_flat_true.py") as f:
            src = f.read()
        self.assertEqual(analyze(src), analyze({"script": src}))
        self.assertEqual(len(analyze(src)["findings"]), 1)

    def test_analyze_notebook_starting_with_markdown(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_true.ipynb")
        notebook["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": ["# Title"]})
        result = analyze(notebook)["findings"]
        self.assertEqual(len(result), 1)
        self.assertTrue(result[0]["message"].startswith("Preprocessing leak"))

    def test_analyze_reports_degraded_code_as_warnings(self):
        src = (
            "import numpy as np\n"                       # 1
            "def helper(*xs):\n"                          # 2
            "    return xs\n"                             # 3
            "def main():\n"                               # 4
            "    X = np.genfromtxt('d.csv')\n"            # 5
            "    del X\n"                                 # 6  unsupported statement
            "    helper(1)\n"                             # 7  *args: not inlined
            "main()\n"
            "main()\n"                                    # main inlined twice: warnings once
        )
        output = analyze({"script": src})
        self.assertEqual(output["findings"], [])
        self.assertEqual([(w["cell"], w["line"], w["file_line"]) for w in output["warnings"]],
                         [(0, 6, 6), (0, 7, 7)])
        self.assertIn("del_stmt", output["warnings"][0]["message"])
        self.assertIn("*args/**kwargs", output["warnings"][1]["message"])


if __name__ == "__main__":
    unittest.main()
