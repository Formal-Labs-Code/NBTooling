import unittest
from nbharness.cli import *
from nbcore.resource_utils.utils import read_json, TEST_RES_PATH
from nbcore.analyses.dataleak_analysis import DATA_LEAK

class TestNBHarnessCLI(unittest.TestCase):
    def test_dataleak(self):
        results_true = nbharness(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":4,"errors":[{"line":2,"label":"X_selected_test", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_selected_test` passed to `lr.predict` is contaminated because `min_max_scaler.fit_transform` (cell 1, line 3) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 1, "line": 3, "call": "min_max_scaler.fit_transform", "variable": "X"}}],"path":[0, 1, 2, 3, 4]}]')
        results_false = nbharness(
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
        results_true = nbharness(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_smote_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":4,"errors":[{"line":1,"label":"X_test", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_test` passed to `lr.predict` is contaminated because `smote.fit_resample` (cell 1, line 5) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 1, "line": 5, "call": "smote.fit_resample", "variable": "X"}}],"path":[0, 1, 2, 3, 4]}]')
        results_false = nbharness(
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
        results_true = nbharness(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_readexcel_score_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":4,"errors":[{"line":2,"label":"X_test", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_test` passed to `lr.score` is contaminated because `scaler.fit_transform` (cell 1, line 3) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 1, "line": 3, "call": "scaler.fit_transform", "variable": "X"}}],"path":[0, 1, 2, 3, 4]}]')
        results_false = nbharness(
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
        results_true = nbharness(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_fit_transform_split_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":5,"errors":[{"line":1,"label":"X_test_scaled", "error_type":"ErrorType.TERMINAL", "message":"Preprocessing leak: `X_test_scaled` passed to `lr.predict` is contaminated because `scaler.fit` (cell 2, line 3) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.", "source":{"cell": 2, "line": 3, "call": "scaler.fit", "variable": "X"}}],"path":[0, 1, 2, 3, 4, 5]}]')
        results_false = nbharness(
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
        results_true = nbharness(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_overlap_true.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_true, '[{"cell_id":3,"errors":[{"line":1,"label":"X_test", "error_type":"ErrorType.TERMINAL", "message":"Train/test overlap: `X_test` passed to `lr.predict` shares rows with `X` used by `lr.fit` (cell 2, line 3), so evaluation scores are inflated.", "source":{"cell": 2, "line": 3, "call": "lr.fit", "variable": "X"}}],"path":[0, 1, 2, 3]}]')
        results_false = nbharness(
                filename=None,
                notebook=read_json(TEST_RES_PATH + "dataleak_overlap_false.ipynb"),
                analyses=[DATA_LEAK],
                start=0,
                level=10
            )
        self.assertEqual(results_false, '')

if __name__ == "__main__":
    unittest.main()