import json
import unittest
from nbfix.cli import *
from nbcore.resource_utils.utils import read_json, TEST_RES_PATH
from nbcore.analyses.dataleak_analysis import DATA_LEAK

class TestNBFixCLI(unittest.TestCase):
    def test_dataleak(self):
        results_true = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":4,"errors":[{"line":2,"label":"X_selected_test", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_selected_test` passed to `lr.predict` is contaminated because `min_max_scaler.fit_transform` (cell 1, line 3) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 1, "line": 3, "call": "min_max_scaler.fit_transform", "variable": "X"}}],"path":[0, 1, 2, 3, 4]}]')
        results_false = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_false.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_false, '')

    def test_dataleak_smote_fit_resample(self):
        """
        Regression coverage for adding 'fit_resample' to taintKB (imblearn's
        SMOTE and friends) - same taint-propagation shape as fit_transform/
        normalize, but a different method name, previously invisible to
        this analysis. Verified directly (not assumed) that the 2-value
        tuple-unpack return (X_res, y_res = smote.fit_resample(X, y)) is
        handled correctly before adding this fixture.
        """
        results_true = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_smote_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":4,"errors":[{"line":1,"label":"X_test", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_test` passed to `lr.predict` is contaminated because `smote.fit_resample` (cell 1, line 5) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 1, "line": 5, "call": "smote.fit_resample", "variable": "X"}}],"path":[0, 1, 2, 3, 4]}]')
        results_false = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_smote_false.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_false, '')

    def test_dataleak_read_excel_source_and_score_sink(self):
        """
        Regression coverage for adding 'read_excel' to resetKB and 'score'
        to testKB - same shape as the existing genfromtxt/predict handling
        respectively, just different real-world entry points (pandas'
        Excel loader, and evaluating via .score() instead of .predict()).
        """
        results_true = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_readexcel_score_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":4,"errors":[{"line":2,"label":"X_test", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_test` passed to `lr.score` is contaminated because `scaler.fit_transform` (cell 1, line 3) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 1, "line": 3, "call": "scaler.fit_transform", "variable": "X"}}],"path":[0, 1, 2, 3, 4]}]')
        results_false = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_readexcel_score_false.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_false, '')

    def test_dataleak_fit_then_transform_split(self):
        """
        Regression coverage for the .fit()-then-separate-.transform()
        leak pattern, arguably the most common real one:
        scaler.fit(X_all) on the whole, unsplit dataset, then
        scaler.transform(X_train)/scaler.transform(X_test) called
        separately per split. This has the identical call *shape* as the
        correct, recommended pattern (fit on the train split only, then
        transform() both splits) - what actually distinguishes them is
        whether the fit() argument's rows were still the full, untouched
        range at fit time, not just "was fit called before two
        transforms." Both directions are tested here specifically because
        a false positive on the *correct* pattern (test_false below) would
        be worse than the original false negative this fixes - verified
        directly, not assumed, before landing.
        """
        results_true = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_fit_transform_split_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":5,"errors":[{"line":1,"label":"X_test_scaled", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_test_scaled` passed to `lr.predict` is contaminated because `scaler.fit` (cell 2, line 3) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 2, "line": 3, "call": "scaler.fit", "variable": "X"}}],"path":[0, 1, 2, 3, 4, 5]}]')
        results_false = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_fit_transform_split_false.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_false, '')

    def test_dataleak_overlap_reports_training_site(self):
        """
        Training on the whole X and then evaluating on X_test is a
        train/test overlap leak - no preprocessing involved - so the
        finding's source is where the model was fit, not a taint origin.
        """
        results_true = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_overlap_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":3,"errors":[{"line":1,"label":"X_test", "error_type":"ErrorType.TERMINAL", "message":"Train/test overlap: `X_test` passed to `lr.predict` shares rows with `X` used by `lr.fit` (cell 2, line 3), so evaluation scores are inflated.", "source":{"cell": 2, "line": 3, "call": "lr.fit", "variable": "X"}}],"path":[0, 1, 2, 3]}]')
        results_false = nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_overlap_false.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_false, '')
    def test_dataleak_param_shadowing_global(self):
        """
        rtests/test7.ipynb's shape: a function parameter named `model`
        shadows the global `model` passed into it. def_use used to treat the
        parameter as binding `model` for the whole cell, so the runner
        explored cell 3 straight after cell 1 (skipping the cell defining
        `model`) and crashed with KeyError. The leak must be reported on
        the real path, through cell 2, using the user's names rather than
        the parser's ~inline_ renames.
        """
        results = json.loads(nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_shadowed_param_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            ))
        self.assertEqual(len(results), 1)
        self.assertEqual(results[0]["path"], [0, 1, 2, 3])
        err = results[0]["errors"][0]
        self.assertEqual((results[0]["cell_id"], err["line"]), (3, 2))
        self.assertEqual(err["label"], "X_local_test")
        self.assertEqual(err["message"],
            "Train/test overlap: `X_local_test` passed to `model.predict` (in `test_model`) shares rows "
            "with `X_local_train` used by `model.fit` (in `train_model`, cell 2, line 5), so evaluation "
            "scores are inflated.")
        self.assertEqual(err["source"], {"cell": 2, "line": 5, "call": "model.fit",
                                         "variable": "X_local_train", "function": "train_model"})
    def test_dataleak_rebind_before_split(self):
        """
        `X = scaler.fit_transform(X)` reads and rebinds X. def_use used to
        cancel the read against the binding, giving the cell no inputs, so
        the runner never visited it and the preprocessing leak was missed.
        """
        results = json.loads(nbfix(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_rebind_before_split_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            ))
        self.assertEqual(len(results), 1)
        err = results[0]["errors"][0]
        self.assertEqual((results[0]["cell_id"], err["line"], err["label"]), (4, 3, "X_test"))
        self.assertTrue(err["message"].startswith("Preprocessing leak:"))
        self.assertEqual(err["source"], {"cell": 2, "line": 2, "call": "scaler.fit_transform", "variable": "X"})
    def test_dataleak_found_when_cell_starts_with_noop_node(self):
        """
        A cell whose first node is a no-op (if/def header) after imports
        used to strand the rest of the cell: the node's output equalled the
        runner's empty default state, so its successors were never queued.
        """
        imports = ("import numpy as np\n"
                   "from sklearn.preprocessing import MinMaxScaler\n"
                   "from sklearn.model_selection import train_test_split\n"
                   "from sklearn.linear_model import LogisticRegression\n")
        body = ("X = np.genfromtxt('data.csv', delimiter=',', dtype=None)\n"
                "y = np.random.choice(3, size=4)\n"
                "X_selected = MinMaxScaler().fit_transform(X)\n"
                "X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.25)\n"
                "lr = LogisticRegression()\n"
                "a = lr.fit(X_train, y_train)\n"
                "y_pred = lr.predict(X_test)\n")
        indented = "".join("    " + line + "\n" for line in body.splitlines())
        shapes = {
            "if-guard": imports + "if __name__ == '__main__':\n" + indented,
            "def-main": imports + "def main():\n" + indented + "main()\n",
        }
        for name, src in shapes.items():
            with self.subTest(shape=name):
                result = nbfix(filename=None, notebook={"cells": [{"cell_type": "code", "source": src}]},
                                   analyses=[DATA_LEAK], start=0, level=10)
                self.assertIn("Preprocessing leak", result)
    def test_dataleak_two_way_split(self):
        """Two one-array splits in a row. The split position used to come
        from a visit counter assuming four targets per split, so the second
        split read a nonexistent second argument and its results went dull."""
        src = ("import numpy as np\nfrom sklearn.model_selection import train_test_split\n"
               "from sklearn.linear_model import LogisticRegression\n"
               "X = np.genfromtxt('d.csv')\n"
               "A = np.genfromtxt('a.csv')\n"
               "A_train, A_test = train_test_split(A, test_size=0.25)\n"
               "X_train, X_test = train_test_split(X, test_size=0.25)\n"
               "lr = LogisticRegression()\na = lr.fit(X, X)\np = lr.predict(X_test)\n")
        result = nbfix(filename=None, notebook={"cells": [{"cell_type": "code", "source": src}]},
                           analyses=[DATA_LEAK], start=0, level=10)
        self.assertIn("Train/test overlap: `X_test` passed to `lr.predict`", result)
        clean = src.replace("a = lr.fit(X, X)", "a = lr.fit(X_train, X_train)")
        self.assertEqual(nbfix(filename=None, notebook={"cells": [{"cell_type": "code", "source": clean}]},
                                   analyses=[DATA_LEAK], start=0, level=10), "")

if __name__ == "__main__":
    unittest.main()