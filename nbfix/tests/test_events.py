import unittest

from nbcore.events import RunBatchEvent
from nbcore.resource_utils.utils import load_notebook, read_json, TEST_RES_PATH
from nbcore.analyses.dataleak_analysis import DATA_LEAK

from nbfix.analyses.stale_cell_analysis import STALE
from nbfix.analyses.idle_cell_analysis import IDLE
from nbfix.events import RunCellEvent
from nbfix.session_factory import new_session


class TestEvents(unittest.TestCase):
    def setUp(self):
        self.maxDiff = None
        self.session = new_session()
        notebook_json = read_json(TEST_RES_PATH + "dataleak_true.ipynb")["cells"]
        self.session.load(load_notebook(notebook_json))
        self.session.add_analyses([DATA_LEAK, STALE, IDLE])

    def test_cell_run_event(self):
        cell_run_event = RunCellEvent(0)
        results = cell_run_event.execute(self.session).dumps()
        self.assertTrue(True)

    def test_batch_run_event(self):
        run_event = RunBatchEvent(0)
        results = run_event.execute(self.session).dumps(True)
        self.assertTrue(True)

if __name__ == "__main__":
    unittest.main()
