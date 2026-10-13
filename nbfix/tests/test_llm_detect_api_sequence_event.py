import unittest
from unittest.mock import MagicMock

import pytest

# Same guard as test_llm_detect_stale_cells_event.py: importing this
# module transitively imports openai (via config.py -> client.py), even
# though tests inject a mock client.
pytest.importorskip("openai")

from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbfix.llm.detect_api_sequence_event import DetectApiSequenceEvent


def make_notebook(cells: dict[int, str]) -> dict:
    return {
        cell_id: IntermediateRepresentations(code, cell_id)
        for cell_id, code in cells.items()
    }


class FakeNBFix:
    def __init__(self, notebook_IR):
        self.cells = notebook_IR


class TestDetectApiSequenceEvent(unittest.TestCase):
    def test_calls_llm_with_every_cell(self):
        notebook_IR = make_notebook({0: "model = M()", 1: "model.predict(X)"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"findings": []}
        event = DetectApiSequenceEvent(client=client)

        event.execute(nbfix)

        client.chat_json.assert_called_once()
        _, user_prompt = client.chat_json.call_args.args
        self.assertIn("model = M()", user_prompt)
        self.assertIn("model.predict(X)", user_prompt)

    def test_maps_llm_response_to_result(self):
        notebook_IR = make_notebook({0: "model = M()", 1: "model.predict(X)"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"findings": [{"cell_id": 1, "message": "call model.fit() first"}]}
        event = DetectApiSequenceEvent(client=client)

        result = event.execute(nbfix)

        self.assertEqual(len(result.path_results), 1)
        error = result.path_results[0].error_infos[0]
        self.assertEqual(error.cell_id, 1)
        self.assertEqual(error.error_type, "LLM_API_SEQUENCE")

    def test_no_violations_returns_empty_result(self):
        notebook_IR = make_notebook({0: "model.fit(X)"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"findings": []}
        event = DetectApiSequenceEvent(client=client)

        result = event.execute(nbfix)

        self.assertEqual(result.path_results, [])

    def test_never_mutates_notebook_ir(self):
        """
        Same discipline as DetectStaleCellsEvent/DetectBugsEvent - this
        is a read-only check, it must never touch cell_code/last_ran_code.
        """
        notebook_IR = make_notebook({0: "model.fit(X)"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"findings": []}
        event = DetectApiSequenceEvent(client=client)

        event.execute(nbfix)

        self.assertEqual(nbfix.cells[0].cell_code, "model.fit(X)")
        self.assertEqual(nbfix.cells[0].last_ran_code, "")

    def test_no_method_calls_anywhere_skips_the_llm_call(self):
        """
        Regression guard for the explicit user-requested optimization:
        a notebook (or component) with zero method calls anywhere
        provably cannot contain this bug class, so there's no reason to
        spend an LLM call on it - see api_sequence_context_builder.py's
        docstring. checked_cells still reflects everything considered
        (not just what got filtered in), so a stale finding on these
        cells is still safely cleared by the caller's merge.
        """
        notebook_IR = make_notebook({0: "x = 1", 1: "y = x + 1"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        event = DetectApiSequenceEvent(client=client)

        result = event.execute(nbfix)

        client.chat_json.assert_not_called()
        self.assertEqual(event.checked_cells, {0, 1})
        self.assertEqual(result.path_results, [])

    def test_checked_cells_covers_everything_when_no_focus_cell(self):
        notebook_IR = make_notebook({0: "x = 1", 1: "y = 2"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"findings": []}
        event = DetectApiSequenceEvent(client=client)

        event.execute(nbfix)

        self.assertEqual(event.checked_cells, {0, 1})

    def test_checked_cells_narrows_to_focus_cells_component(self):
        notebook_IR = make_notebook({0: "x = 1", 1: "y = x + 1", 2: "z = 99"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"findings": []}
        event = DetectApiSequenceEvent(focus_cell=0, client=client)

        event.execute(nbfix)

        self.assertEqual(event.checked_cells, {0, 1})

    def test_checked_cells_empty_before_execute(self):
        event = DetectApiSequenceEvent(client=MagicMock())
        self.assertEqual(event.checked_cells, set())

    def test_focus_cell_not_in_notebook_ir_returns_empty_result(self):
        """
        Regression guard for a live crash: notebook_IR only ever contains
        code cells (NBFix.load_notebook skips markdown), but the frontend
        watches every cell for edits regardless of type. Editing a
        markdown cell sent its position as focus_cell and crashed with a
        bare KeyError ("Event execution failed: 6" - Python's KeyError
        stringifies to just the key), since nothing guarded against a
        focus_cell that was never a real notebook_IR entry.
        """
        notebook_IR = make_notebook({0: "x = 1", 1: "y = 2"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        event = DetectApiSequenceEvent(focus_cell=6, client=client)

        result = event.execute(nbfix)

        client.chat_json.assert_not_called()
        self.assertEqual(event.checked_cells, set())
        self.assertEqual(result.path_results, [])


if __name__ == "__main__":
    unittest.main()
