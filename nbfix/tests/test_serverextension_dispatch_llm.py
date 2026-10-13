import unittest

import pytest

pytest.importorskip("openai")

from nbfix.llm.detect_api_sequence_event import DetectApiSequenceEvent
from nbcore.llm.detect_bugs_event import DetectBugsEvent
from nbfix.llm.detect_stale_cells_event import DetectStaleCellsEvent
from nbfix.serverextension.dispatch import InvalidEventError, build_event


class TestDetectBugsDispatch(unittest.TestCase):
    def test_full_scope(self):
        event = build_event("detect_bugs", {"scope": "full"})
        self.assertIsInstance(event, DetectBugsEvent)
        self.assertEqual(event.scope, "full")
        self.assertIsNone(event.cell_index)

    def test_defaults_to_full_scope(self):
        event = build_event("detect_bugs", {})
        self.assertEqual(event.scope, "full")

    def test_cell_scope_with_cell_index(self):
        event = build_event("detect_bugs", {"scope": "cell", "cell_index": 2})
        self.assertEqual(event.scope, "cell")
        self.assertEqual(event.cell_index, 2)

    def test_subgraph_scope_with_cell_index(self):
        event = build_event("detect_bugs", {"scope": "subgraph", "cell_index": 0})
        self.assertEqual(event.scope, "subgraph")
        self.assertEqual(event.cell_index, 0)

    def test_invalid_scope_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("detect_bugs", {"scope": "not_a_scope"})

    def test_cell_scope_without_cell_index_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("detect_bugs", {"scope": "cell"})

    def test_subgraph_scope_without_cell_index_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("detect_bugs", {"scope": "subgraph"})

    def test_context_mode_defaults_to_deps(self):
        event = build_event("detect_bugs", {"scope": "full"})
        self.assertEqual(event.context_mode, "deps")

    def test_context_mode_none_accepted(self):
        event = build_event("detect_bugs", {"scope": "full", "context_mode": "none"})
        self.assertEqual(event.context_mode, "none")

    def test_invalid_context_mode_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("detect_bugs", {"scope": "full", "context_mode": "bogus"})

    def test_finding_types_defaults_to_none(self):
        event = build_event("detect_bugs", {"scope": "full"})
        self.assertIsNone(event.finding_types)

    def test_finding_types_accepted(self):
        event = build_event(
            "detect_bugs", {"scope": "full", "finding_types": ["Idle Cells Analysis"]}
        )
        self.assertEqual(event.finding_types, ["Idle Cells Analysis"])

    def test_invalid_finding_types_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("detect_bugs", {"scope": "full", "finding_types": ["Not A Real Analysis"]})


class TestDetectStaleCellsLLMDispatch(unittest.TestCase):
    def test_builds_event_with_cell_index_and_original_code(self):
        event = build_event(
            "detect_stale_cells_llm", {"cell_index": 2, "original_code": "x = 10"}
        )
        self.assertIsInstance(event, DetectStaleCellsEvent)
        self.assertEqual(event.cell_index, 2)
        self.assertEqual(event.original_code, "x = 10")

    def test_missing_cell_index_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("detect_stale_cells_llm", {"original_code": "x = 10"})

    def test_missing_original_code_raises(self):
        with self.assertRaises(InvalidEventError):
            build_event("detect_stale_cells_llm", {"cell_index": 0})

    def test_empty_original_code_is_accepted(self):
        # "" is a real, meaningful value (a never-run cell) - must not be
        # rejected the same way a genuinely missing param is.
        event = build_event(
            "detect_stale_cells_llm", {"cell_index": 0, "original_code": ""}
        )
        self.assertEqual(event.original_code, "")


class TestDetectApiSequenceLLMDispatch(unittest.TestCase):
    def test_builds_event_with_no_focus_cell(self):
        # No focus_cell - scans the whole notebook in one call.
        event = build_event("detect_api_sequence_llm", {})
        self.assertIsInstance(event, DetectApiSequenceEvent)
        self.assertIsNone(event.focus_cell)

    def test_builds_event_with_focus_cell(self):
        event = build_event("detect_api_sequence_llm", {"focus_cell": 2})
        self.assertEqual(event.focus_cell, 2)


if __name__ == "__main__":
    unittest.main()
