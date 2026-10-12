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
        self.assertIsNone(finding["source"])
        self.assertEqual(finding["path"], [0, 1, 2, 3, 4])
        self.assertEqual(finding["locations"], ["cell 4", "line 2"])
        self.assertEqual(finding["message"], "Training model with data leak.")


if __name__ == "__main__":
    unittest.main()
