import copy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from nbcore.llm.client import LLMClientError
from nbcore.resource_utils.script_loading import script_to_notebook
from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

from nbfix import repair
from nbfix.findings import analyze_data_leaks
from nbfix.repair import fix


def _source(cell):
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _client(*responses):
    client = MagicMock()
    client.chat_json.side_effect = list(responses)
    return client


def _edits(*pairs):
    return {"edits": [{"cell": cell, "code": code} for cell, code in pairs], "explanation": "fixed"}


def _prompt(client, call=0):
    return client.chat_json.call_args_list[call][0][1]


class TestFixVerifiesKnownGoodRepairs(unittest.TestCase):
    """These dataleak_*_true notebooks have a dataleak_*_false twin with the
    same cells and the leak fixed - answering with the twin's changed cells
    must be verified. (dataleak_false.ipynb is an unrelated notebook, not
    dataleak_true.ipynb's twin.)"""

    PAIRS = ["dataleak_fit_transform_split", "dataleak_overlap", "dataleak_smote",
             "dataleak_readexcel_score"]

    def test_clean_twin_is_verified(self):
        for name in self.PAIRS:
            with self.subTest(name):
                leaky = read_json(f"{TEST_RES_PATH}{name}_true.ipynb")
                clean = read_json(f"{TEST_RES_PATH}{name}_false.ipynb")
                changed = [(i, _source(c)) for i, (a, c) in enumerate(zip(leaky["cells"], clean["cells"]))
                           if _source(a) != _source(c)]
                client = _client(_edits(*changed))

                result = fix(leaky, client=client)

                self.assertEqual(result["status"], "verified", result.get("problems"))
                self.assertEqual(result["attempts"], 1)
                self.assertEqual([e["cell"] for e in result["edits"]], [i for i, _ in changed])
                self.assertEqual(len(result["findings"]), 1)


class TestFix(unittest.TestCase):
    def setUp(self):
        self.leaky = read_json(TEST_RES_PATH + "dataleak_fit_transform_split_true.ipynb")
        self.fixed_cell_2 = "scaler = StandardScaler()\nscaler.fit(X_train)"

    def test_no_findings_never_calls_the_llm(self):
        client = _client()
        result = fix(read_json(TEST_RES_PATH + "dataleak_fit_transform_split_false.ipynb"), client=client)
        self.assertEqual(result["status"], "no_findings")
        client.chat_json.assert_not_called()

    def test_prompt_lists_cells_leak_and_locations(self):
        client = _client(_edits((2, self.fixed_cell_2)))
        fix(self.leaky, client=client)
        prompt = _prompt(client)
        self.assertIn("### Cell 2 (editable)", prompt)
        self.assertIn("Preprocessing leak: `X_test_scaled` passed to `lr.predict`", prompt)
        self.assertIn("Reported at: cell 5, line 1: `y_pred = lr.predict(X_test_scaled)`", prompt)
        self.assertIn("Caused by: cell 2, line 3: `scaler.fit(X)`", prompt)

    def test_input_notebook_is_not_modified(self):
        before = copy.deepcopy(self.leaky)
        fix(self.leaky, client=_client(_edits((2, self.fixed_cell_2))))
        self.assertEqual(self.leaky, before)

    def test_notebook_diff_is_per_cell(self):
        result = fix(self.leaky, client=_client(_edits((2, self.fixed_cell_2))), filename="nb.ipynb")
        self.assertIn("--- a/nb.ipynb (cell 2)", result["diff"])
        self.assertIn("-scaler.fit(X)\n+scaler.fit(X_train)", result["diff"])

    def test_code_fence_around_cell_code_is_stripped(self):
        result = fix(self.leaky, client=_client(_edits((2, f"```python\n{self.fixed_cell_2}\n```"))))
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["edits"][0]["code"], self.fixed_cell_2)

    def test_formatting_only_changes_are_undone(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_fit_transform_split_true.ipynb")
        notebook["cells"][2]["source"] = "# fit\n\nscaler = StandardScaler( )\n\nscaler.fit(X)"
        reformatted = "# fit\nscaler = StandardScaler()\nscaler.fit(X_train)"
        result = fix(notebook, client=_client(_edits((2, reformatted))))
        self.assertEqual(result["status"], "verified")
        # Only the fit line changed; the spacing and blank lines are the original's.
        self.assertEqual(result["edits"][0]["code"], "# fit\n\nscaler = StandardScaler( )\n\nscaler.fit(X_train)")

    def test_still_leaky_patch_is_retried_with_feedback_then_unverified(self):
        still_leaky = "scaler = StandardScaler()\nscaler.fit(X)  # unchanged fit"
        client = _client(*[_edits((2, still_leaky))] * 3)

        result = fix(self.leaky, client=client)

        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["attempts"], 3)
        self.assertEqual(len(result["remaining_findings"]), 1)
        self.assertIn("NBFix still reports: Preprocessing leak", result["problems"][0])
        self.assertIn("## Your previous attempt was rejected", _prompt(client, 1))
        self.assertIn("NBFix still reports", _prompt(client, 1))

    def test_deleting_the_sink_call_is_rejected(self):
        client = _client(_edits((5, "y_pred = None")), _edits((2, self.fixed_cell_2)))
        result = fix(self.leaky, client=client)
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["attempts"], 2)
        self.assertIn("The `lr.predict` call where a leak was reported is gone", _prompt(client, 1))

    def test_stubbing_out_the_source_step_is_rejected(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_smote_true.ipynb")
        stub = ("class SMOTE:\n    def fit_resample(self, X, y):\n        return X, y\n"
                "smote = SMOTE()\nX_res, y_res = X, y")
        result = fix(notebook, client=_client(*[_edits((1, stub))] * 3))
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["remaining_findings"], [])
        self.assertIn("The `smote.fit_resample` step a leak was traced to is no longer called", result["problems"][0])

    def test_reading_a_name_before_it_is_defined_is_rejected(self):
        # Resampling the train split in cell 1, before cell 2 splits the data.
        notebook = read_json(TEST_RES_PATH + "dataleak_smote_true.ipynb")
        edits = _edits(
            (1, "smote = SMOTE()\nX_train, y_train = smote.fit_resample(X_train, y_train)"),
            (2, "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)"),
        )
        result = fix(notebook, client=_client(*[edits] * 3))
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["problems"], [
            "Cell 1 reads `X_train` before any earlier code defines it, so the patched code would fail "
            "with a NameError; keep the cells in an order that runs top to bottom.",
            "Cell 1 reads `y_train` before any earlier code defines it, so the patched code would fail "
            "with a NameError; keep the cells in an order that runs top to bottom.",
        ])

    def test_unused_preprocessing_result_is_rejected(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_smote_true.ipynb")
        edits = _edits(
            (1, "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)"),
            (2, "smote = SMOTE()\nX_train_res, y_train_res = smote.fit_resample(X_train, y_train)"),
        )  # cell 3 still trains on X_train, not X_train_res
        result = fix(notebook, client=_client(*[edits] * 3))
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(len(result["problems"]), 2)
        self.assertIn("`X_train_res` is computed by the preprocessing step but never used", result["problems"][0])

    def test_fit_transform_source_may_become_fit_plus_transform(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_readexcel_score_true.ipynb")
        client = _client(_edits(
            (1, "scaler = MinMaxScaler()"),
            (2, "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2)\n"
                "scaler.fit(X_train)\nX_train = scaler.transform(X_train)\nX_test = scaler.transform(X_test)"),
        ))
        self.assertEqual(fix(notebook, client=client)["status"], "verified")

    def test_new_analysis_warning_is_rejected(self):
        unmodelled = self.fixed_cell_2 + "\nasync def later():\n    pass"
        result = fix(self.leaky, client=_client(*[_edits((2, unmodelled))] * 3))
        self.assertEqual(result["status"], "unverified")
        self.assertEqual(result["remaining_findings"], [])
        self.assertIn("uses code NBFix cannot fully analyze", result["problems"][0])

    def test_syntax_error_is_rejected_then_retried(self):
        client = _client(_edits((2, "scaler = StandardScaler(\n")), _edits((2, self.fixed_cell_2)))
        result = fix(self.leaky, client=client)
        self.assertEqual(result["status"], "verified")
        self.assertIn("Cell 2 has a syntax error", _prompt(client, 1))

    def test_non_editable_and_unknown_cells_are_rejected(self):
        notebook = copy.deepcopy(self.leaky)
        notebook["cells"].append({"cell_type": "code", "source": "print(y_pred)", "metadata": {}})
        # Cell 6 runs after the sink, so it is not on the leak's path.
        result = fix(notebook, client=_client(*[_edits((6, "print(1)"), (9, "x = 1"))] * 3))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["problems"], ["Cell 6 is not editable.", "Cell 9 is not a code cell."])

    def test_unchanged_response_fails(self):
        result = fix(self.leaky, client=_client(*[_edits()] * 3))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["problems"], ["The response changes no cells."])

    def test_malformed_response_fails(self):
        result = fix(self.leaky, client=_client(*[{"patch": "..."}] * 3))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["problems"], ['The response has no "edits" list.'])

    def test_analysis_error_on_patch_is_rejected(self):
        calls = []

        def analyze(payload):
            calls.append(payload)
            if len(calls) > 1:
                raise ValueError("unsupported\nmore detail")
            return analyze_data_leaks(payload)

        with patch.object(repair, "analyze_data_leaks", side_effect=analyze):
            result = fix(self.leaky, client=_client(*[_edits((2, self.fixed_cell_2))] * 3))
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["problems"], ["NBFix could not analyze the patched code (ValueError: unsupported)."])

    def test_llm_client_error_raises_runtime_error(self):
        client = MagicMock()
        client.chat_json.side_effect = LLMClientError("Could not reach LLM endpoint")
        with self.assertRaisesRegex(RuntimeError, "Could not reach LLM endpoint"):
            fix(self.leaky, client=client)


class TestFixScripts(unittest.TestCase):
    def _git_apply(self, name, source, diff):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, name).write_text(source, encoding="utf-8")
            subprocess.run(["git", "apply", "-"], cwd=tmp, input=diff, text=True, check=True, capture_output=True)
            return Path(tmp, name).read_text(encoding="utf-8")

    def test_single_cell_script_diff_applies_to_the_file(self):
        source = Path(TEST_RES_PATH + "dataleak_script_flat_true.py").read_text(encoding="utf-8")
        clean = Path(TEST_RES_PATH + "dataleak_script_flat_false.py").read_text(encoding="utf-8")

        result = fix(script_to_notebook(source), client=_client(_edits((0, clean))),
                     source=source, filename="flat.py")

        self.assertEqual(result["status"], "verified")
        self.assertEqual(self._git_apply("flat.py", source, result["diff"]), clean)

    def test_cell_marker_script_diff_keeps_markers_and_removes_the_leak(self):
        source = Path(TEST_RES_PATH + "dataleak_script_cells_true.py").read_text(encoding="utf-8")
        split_then_scale = (
            "# then split\n"
            "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.025, random_state=2)\n"
            "X_selected_train = min_max_scaler.fit_transform(X_train)\n"
            "X_selected_test = min_max_scaler.transform(X_test)"
        )
        client = _client(_edits((2, "min_max_scaler = MinMaxScaler()"), (3, split_then_scale)))

        result = fix(script_to_notebook(source), client=client, source=source, filename="cells.py")

        self.assertEqual(result["status"], "verified", result.get("problems"))
        patched = self._git_apply("cells.py", source, result["diff"])
        self.assertEqual(patched.count("# %%"), source.count("# %%"))
        self.assertIn("\n\n# %%\n# then split\n", patched)
        self.assertEqual(analyze_data_leaks(script_to_notebook(patched))["findings"], [])


if __name__ == "__main__":
    unittest.main()
