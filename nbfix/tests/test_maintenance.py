import unittest

from nbcore.analyses.dataleak_analysis import DataLeakAnalysis, DATA_LEAK
from nbcore.resource_utils.utils import load_notebook, read_json, TEST_RES_PATH

from nbfix.analyses.stale_cell_analysis import StaleCellAnalysis, STALE
from nbfix.session_factory import new_session


class TestNBFixMaintenance(unittest.TestCase):
    def setUp(self) -> None:
        self.notebook_json = read_json(TEST_RES_PATH + "dataleak_true.ipynb")[
            "cells"
        ]
        self.session = new_session()
        self.reference_cells = load_notebook(self.notebook_json)

    def test_load_notebook(self):
        reference_cells = load_notebook(self.notebook_json)
        self.session.load(load_notebook(self.notebook_json))
        for i in reference_cells.keys():
            self.assertEqual(reference_cells[i], self.session.cells[i])

    def test_add_analyses(self):
        self.session.add_analyses([DATA_LEAK])
        self.assertIn(DATA_LEAK, self.session.active_analyses)
        self.assertIsInstance(
            self.session.all_analyses[DATA_LEAK], DataLeakAnalysis
        )

        self.session.add_analyses([STALE])
        self.assertIn(STALE, self.session.active_analyses)
        self.assertIsInstance(
            self.session.all_analyses[STALE], StaleCellAnalysis
        )

    def test_update_abstract_states(self):
        """
        TODO: Update pickles to match new abstract state model
        """
        # self.session.add_analyses([DATA_LEAK])

        # cell_0 = self.reference_cells[0]
        # self.session.update_abstract_states(cell_0)
        # as_0 = pickle.loads(mngr.grab_remote("abstract_state_0.pickle"))
        # self.assertEqual(as_0, self.session.active_analyses[DATA_LEAK].abstract_state)

if __name__ == "__main__":
    unittest.main()
