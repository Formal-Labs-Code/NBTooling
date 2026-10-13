import json
import unittest

from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

from nbfix.analyses.idle_cell_analysis import IDLE
from nbfix.analyses.isolated_cell_analysis import ISOLATED
from nbfix.analyses.stale_cell_analysis import STALE
from nbfix.events import AddActiveAnalysesEvent, ChangeCellCodeEvent, OpenNotebookEvent, RunCellEvent
from nbfix.session_factory import new_session

FIXED_CELL_2 = "lr = LogisticRegression()\na = lr.fit(X_train, y_train)"


def _messages(result):
    # What the server extension sends - must be valid JSON.
    entries = json.loads(result.dumps() or "[]")
    return [(entry["cell_id"], error["line"], error["message"]) for entry in entries for error in entry["errors"]]


def _leaks(results):
    """Data leaks the UI shows after the last run (each response replaces the
    previous diagnostics) - every response must still be valid JSON."""
    messages = [_messages(result) for result in results]
    return {m[:2] for m in messages[-1] if m[2].startswith(("Train/test overlap", "Preprocessing leak"))}


class TestLiveDataLeakAcrossReruns(unittest.TestCase):
    """The live session (JupyterLab) analyzes data leaks as cells run, keeping
    an abstract state of the kernel; re-running edited cells must update it."""

    def setUp(self):
        notebook = read_json(TEST_RES_PATH + "dataleak_overlap_true.ipynb")
        self.session = new_session()
        self.session.execute_event(OpenNotebookEvent([
            {"cell_type": cell["cell_type"], "source": "".join(cell["source"])} for cell in notebook["cells"]
        ]))
        self.session.execute_event(AddActiveAnalysesEvent([DATA_LEAK, STALE, IDLE, ISOLATED]))

    def _run_all(self):
        return [self.session.execute_event(RunCellEvent(index)) for index in range(4)]

    def test_leak_is_reported_when_the_cells_run(self):
        self.assertEqual(_leaks(self._run_all()), {(3, 1)})

    def test_rerunning_fixed_cells_clears_the_leak(self):
        self._run_all()
        self.session.execute_event(ChangeCellCodeEvent(FIXED_CELL_2, 2, with_result=True))
        self.assertEqual(_leaks(self._run_all()), set())

    def test_rerunning_unchanged_cells_still_reports_the_leak(self):
        self._run_all()
        self.assertEqual(_leaks(self._run_all()), {(3, 1)})


if __name__ == "__main__":
    unittest.main()
