import json
import unittest

from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbfix.llm.stale_result_mapping import map_stale_findings_to_result


def make_notebook(cells: dict[int, str]) -> dict:
    return {
        cell_id: IntermediateRepresentations(code, cell_id)
        for cell_id, code in cells.items()
    }


class TestMapStaleFindingsToResult(unittest.TestCase):
    def setUp(self):
        self.notebook_IR = make_notebook({0: "x = 1", 1: "y = x + 1", 2: "z = y + 1"})

    def test_valid_finding(self):
        findings_json = {"stale_cells": [2]}
        result = map_stale_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(len(result.path_results), 1)
        error = result.path_results[0].error_infos[0]
        self.assertEqual(error.cell_id, 2)
        self.assertEqual(error.error_type, "LLM_STALE")
        self.assertTrue(error.error_message)

    def test_multiple_valid_findings(self):
        findings_json = {"stale_cells": [1, 2]}
        result = map_stale_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual({pr.error_infos[0].cell_id for pr in result.path_results}, {1, 2})

    def test_dumps_produces_valid_json(self):
        findings_json = {"stale_cells": [1]}
        result = map_stale_findings_to_result(findings_json, self.notebook_IR)
        parsed = json.loads(result.dumps())
        self.assertEqual(parsed[0]["cell_id"], 1)
        self.assertEqual(parsed[0]["errors"][0]["error_type"], "LLM_STALE")

    def test_missing_stale_cells_key_returns_empty_result(self):
        result = map_stale_findings_to_result({"not_stale_cells": []}, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_empty_list_returns_empty_result(self):
        result = map_stale_findings_to_result({"stale_cells": []}, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_unknown_cell_id_is_dropped(self):
        findings_json = {"stale_cells": [99]}
        result = map_stale_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_non_int_cell_id_is_dropped(self):
        findings_json = {"stale_cells": ["1"]}
        result = map_stale_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_bool_cell_id_is_dropped(self):
        # bool is a subclass of int in Python - must not sneak through.
        findings_json = {"stale_cells": [True]}
        result = map_stale_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_one_invalid_finding_does_not_drop_valid_ones(self):
        findings_json = {"stale_cells": [1, 99]}
        result = map_stale_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(len(result.path_results), 1)
        self.assertEqual(result.path_results[0].error_infos[0].cell_id, 1)


if __name__ == "__main__":
    unittest.main()
