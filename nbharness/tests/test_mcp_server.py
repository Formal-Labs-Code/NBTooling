import unittest

from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

from nbharness.mcp_server import analyze


class TestNBHarnessMCPServer(unittest.TestCase):
    def test_analyze_detects_known_data_leak(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_true.ipynb")

        result = analyze(notebook)

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
        self.assertEqual(analyze(read_json(TEST_RES_PATH + "dataleak_false.ipynb")), [])


if __name__ == "__main__":
    unittest.main()
