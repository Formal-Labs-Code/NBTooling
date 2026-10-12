import json
import os
import unittest

from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

from nbharness.analyses.stale_cell_analysis import STALE
from nbharness.cli import nbharness


def run(name):
    raw = nbharness(filename=os.path.join(TEST_RES_PATH, name), notebook=None,
                    analyses=[DATA_LEAK], start=0, level=10)
    return json.loads(raw) if raw else []


class TestNBHarnessScripts(unittest.TestCase):
    def test_flat_script_leak_reports_file_lines(self):
        results = run("dataleak_script_flat_true.py")
        self.assertEqual(len(results), 1)
        err = results[0]["errors"][0]
        self.assertEqual((results[0]["cell_id"], err["line"], err["file_line"]), (0, 17, 17))
        self.assertEqual(err["source"]["call"], "scaler.fit_transform")
        self.assertEqual(err["source"]["file_line"], 11)

    def test_flat_script_clean(self):
        self.assertEqual(run("dataleak_script_flat_false.py"), [])

    def test_main_guard_script(self):
        results = run("dataleak_script_main_true.py")
        self.assertEqual(len(results), 1)
        err = results[0]["errors"][0]
        self.assertEqual((err["label"], err["file_line"]), ("X_test", 14))
        self.assertEqual((err["source"]["function"], err["source"]["file_line"]), ("main", 10))

    def test_percent_cells_match_the_notebook(self):
        """The `# %%` script is dataleak_true.ipynb plus a leading markdown
        cell: same finding, cell ids shifted by one, plus file lines."""
        script = run("dataleak_script_cells_true.py")
        notebook = json.loads(nbharness(filename=None, notebook=read_json(TEST_RES_PATH + "dataleak_true.ipynb"),
                                        analyses=[DATA_LEAK], start=0, level=10))
        s_err, n_err = script[0]["errors"][0], notebook[0]["errors"][0]
        self.assertEqual(script[0]["cell_id"], notebook[0]["cell_id"] + 1)
        self.assertEqual(script[0]["path"], [c + 1 for c in notebook[0]["path"]])
        self.assertEqual((s_err["label"], s_err["line"]), (n_err["label"], n_err["line"]))
        self.assertEqual(s_err["source"]["call"], n_err["source"]["call"])
        self.assertEqual((s_err["file_line"], s_err["source"]["file_line"]), (33, 20))

    def test_evaluation_call_nested_in_other_calls(self):
        results = run("dataleak_script_nested_call_true.py")
        self.assertEqual(len(results), 1)
        err = results[0]["errors"][0]
        self.assertEqual((err["label"], err["file_line"]), ("X_test", 15))
        self.assertIn("passed to `lr.predict`", err["message"])
        # the hoisted receiver temp is reported under its original text
        self.assertEqual(err["source"]["call"], "MinMaxScaler().fit_transform")

    def test_notebook_only_analysis_rejected_without_markers(self):
        with self.assertRaises(ValueError):
            nbharness(filename=os.path.join(TEST_RES_PATH, "dataleak_script_flat_true.py"), notebook=None,
                      analyses=[STALE], start=0, level=10)


if __name__ == "__main__":
    unittest.main()
