import unittest

from nbfix.events import (
    AddCellEvent,
    ChangeCellCodeEvent,
    CloseNotebookEvent,
    OpenNotebookEvent,
    RemoveCellEvent,
    RunCellEvent,
)
from nbfix.serverextension.dispatch import InvalidEventError, build_event


class TestBuildEvent(unittest.TestCase):
    def test_unknown_event_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("not_a_real_event", {})

    def test_open_notebook(self):
        event = build_event("open_notebook", {"notebook_json": []})
        self.assertIsInstance(event, OpenNotebookEvent)
        self.assertEqual(event.notebook_json, [])

    def test_open_notebook_missing_param_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("open_notebook", {})

    def test_run_cell(self):
        event = build_event("run_cell", {"cell_index": 3})
        self.assertIsInstance(event, RunCellEvent)
        self.assertEqual(event.cell_index, 3)

    def test_add_cell(self):
        event = build_event(
            "add_cell", {"position": 1, "kind": 2, "content": "x = 1"}
        )
        self.assertIsInstance(event, AddCellEvent)
        self.assertEqual(event.position, 1)
        self.assertEqual(event.kind, 2)
        self.assertEqual(event.content, "x = 1")

    def test_remove_cell(self):
        event = build_event("remove_cell", {"position": 2})
        self.assertIsInstance(event, RemoveCellEvent)
        self.assertEqual(event.position, 2)

    def test_change_cell(self):
        event = build_event(
            "change_cell",
            {"new_code": "y = 2", "cell_index": 0, "with_result": True},
        )
        self.assertIsInstance(event, ChangeCellCodeEvent)
        self.assertEqual(event.new_code, "y = 2")
        self.assertEqual(event.cell_index, 0)
        self.assertTrue(event.with_result)

    def test_close_notebook(self):
        event = build_event("close_notebook", {})
        self.assertIsInstance(event, CloseNotebookEvent)

    def test_add_cell_missing_param_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("add_cell", {"position": 1, "kind": 2})


if __name__ == "__main__":
    unittest.main()
