import unittest
from unittest.mock import MagicMock

import pytest

# Same guard as test_llm_detect_bugs_event.py: importing this module
# transitively imports openai (via config.py -> client.py), even though
# tests inject a mock client.
pytest.importorskip("openai")

from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbfix.llm.detect_stale_cells_event import DetectStaleCellsEvent


def make_notebook(cells: dict[int, str]) -> dict:
    return {
        cell_id: IntermediateRepresentations(code, cell_id)
        for cell_id, code in cells.items()
    }


class FakeNBFix:
    def __init__(self, notebook_IR):
        self.cells = notebook_IR


class TestDetectStaleCellsEvent(unittest.TestCase):
    def test_no_edit_returns_empty_result_without_calling_client(self):
        notebook_IR = make_notebook({0: "x = 10"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        event = DetectStaleCellsEvent(0, original_code="x = 10", client=client)

        result = event.execute(nbfix)

        self.assertEqual(result.path_results, [])
        client.chat_json.assert_not_called()

    def test_never_run_cell_with_real_code_is_treated_as_an_edit(self):
        """
        Matches the real StaleCellAnalysis's own behavior (confirmed
        empirically this session, not assumed): a cell's very first run
        compares its code against an empty prior code, which looks like a
        genuine change. Faithfully mirroring that here rather than
        special-casing "never run" keeps this event consistent with what
        the deterministic analysis already does.
        """
        notebook_IR = make_notebook({0: "x = 10"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"stale_cells": []}
        event = DetectStaleCellsEvent(0, original_code="", client=client)

        event.execute(nbfix)

        client.chat_json.assert_called_once()

    def test_edit_triggers_llm_call_with_correct_prompt(self):
        notebook_IR = make_notebook({0: "x = 99", 1: "y = x + 1"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"stale_cells": []}
        event = DetectStaleCellsEvent(0, original_code="x = 10", client=client)

        event.execute(nbfix)

        client.chat_json.assert_called_once()
        _, user_prompt = client.chat_json.call_args.args
        self.assertIn("x = 10", user_prompt)
        self.assertIn("x = 99", user_prompt)
        self.assertIn("y = x + 1", user_prompt)

    def test_maps_llm_response_to_result(self):
        notebook_IR = make_notebook({0: "x = 99", 1: "y = x + 1", 2: "z = y + 1"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"stale_cells": [2]}
        event = DetectStaleCellsEvent(0, original_code="x = 10", client=client)

        result = event.execute(nbfix)

        self.assertEqual(len(result.path_results), 1)
        error = result.path_results[0].error_infos[0]
        self.assertEqual(error.cell_id, 2)
        self.assertEqual(error.error_type, "LLM_STALE")

    def test_never_mutates_notebook_ir(self):
        """
        Regression guard matching DetectBugsEvent's own
        active_analyses-unchanged test: this event must never update
        last_ran_code/cell_code itself - that's events.RunCellEvent's
        job, and doing it here too would silently double-apply state
        changes if a caller runs both.
        """
        notebook_IR = make_notebook({0: "x = 99"})
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"stale_cells": []}
        event = DetectStaleCellsEvent(0, original_code="x = 10", client=client)

        event.execute(nbfix)

        self.assertEqual(nbfix.cells[0].last_ran_code, "")
        self.assertEqual(nbfix.cells[0].cell_code, "x = 99")

    def test_uses_explicit_original_code_not_last_ran_code(self):
        """
        Regression guard for a real ordering bug found via a live
        end-to-end check: events.RunCellEvent overwrites last_ran_code to
        match cell_code as part of the *same* call that actually executes
        a cell, so by the time it's safe to call this event (the cell's
        kernel value is genuinely fresh), last_ran_code no longer holds
        the pre-edit code - it already matches cell_code. An earlier
        version of this event read notebook_IR[cell_index].last_ran_code
        internally and was silently broken as a result: it would only
        ever see "no edit" once RunCellEvent had already run. Here,
        last_ran_code on the fake IR is already reset to match cell_code
        (simulating the state right after a real RunCellEvent), but the
        explicitly-passed original_code still correctly triggers a check.
        """
        notebook_IR = make_notebook({0: "x = 99"})  # last_ran_code defaults to "" here,
        notebook_IR[0].last_ran_code = "x = 99"  # then simulate RunCellEvent having already reset it
        nbfix = FakeNBFix(notebook_IR)
        client = MagicMock()
        client.chat_json.return_value = {"stale_cells": []}
        event = DetectStaleCellsEvent(0, original_code="x = 10", client=client)

        event.execute(nbfix)

        client.chat_json.assert_called_once()


if __name__ == "__main__":
    unittest.main()
