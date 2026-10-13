import unittest
from nbfix.analyses.isolated_cell_analysis import IsolatedCellAnalysis
from nbcore.resource_utils.utils import load_notebook, read_json, TEST_RES_PATH
import json


class TestIsolatedCellAnalysis(unittest.TestCase):
    def test_analyze_notebook(self):
        test_IR = load_notebook(
            read_json(TEST_RES_PATH + "Test.ipynb")["cells"]
        )
        analyzer_test = IsolatedCellAnalysis()
        self.assertEqual(analyzer_test.analyze_notebook(test_IR).dumps(), '[{"cell_id":6,"errors":[{"line":1,"label":"f", "error_type":"ErrorType.TERMINAL", "message":"This cell contains only unused variables."}]}]')

        basic_IR = load_notebook(
            read_json(TEST_RES_PATH + "Basic.ipynb")["cells"]
        )
        analyzer_basic = IsolatedCellAnalysis()
        self.assertEqual(analyzer_basic.analyze_notebook(basic_IR).dumps(), '[{"cell_id":0,"errors":[{"line":1,"label":"x", "error_type":"ErrorType.TERMINAL", "message":"This cell contains only unused variables."}]},{"cell_id":1,"errors":[{"line":1,"label":"y", "error_type":"ErrorType.TERMINAL", "message":"This cell contains only unused variables."}]}]')

if __name__ == "__main__":
    unittest.main()