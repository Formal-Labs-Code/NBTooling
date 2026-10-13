import json
import unittest

from nbcore.ir.intermediate_representations import IntermediateRepresentations
from nbfix.llm.api_sequence_result_mapping import map_api_sequence_findings_to_result


def make_notebook(cells: dict[int, str]) -> dict:
    return {
        cell_id: IntermediateRepresentations(code, cell_id)
        for cell_id, code in cells.items()
    }


class TestMapApiSequenceFindingsToResult(unittest.TestCase):
    def setUp(self):
        self.notebook_IR = make_notebook({0: "model = M()", 1: "model.predict(X)"})

    def test_valid_finding(self):
        findings_json = {"findings": [{"cell_id": 1, "message": "call model.fit() first"}]}
        result = map_api_sequence_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(len(result.path_results), 1)
        error = result.path_results[0].error_infos[0]
        self.assertEqual(error.cell_id, 1)
        self.assertEqual(error.error_type, "LLM_API_SEQUENCE")
        self.assertEqual(error.error_message, "call model.fit() first")

    def test_multiple_valid_findings(self):
        notebook_IR = make_notebook({0: "a", 1: "b", 2: "c"})
        findings_json = {"findings": [
            {"cell_id": 1, "message": "x"},
            {"cell_id": 2, "message": "y"},
        ]}
        result = map_api_sequence_findings_to_result(findings_json, notebook_IR)
        self.assertEqual({pr.error_infos[0].cell_id for pr in result.path_results}, {1, 2})

    def test_dumps_produces_valid_json(self):
        findings_json = {"findings": [{"cell_id": 1, "message": "example"}]}
        result = map_api_sequence_findings_to_result(findings_json, self.notebook_IR)
        parsed = json.loads(result.dumps())
        self.assertEqual(parsed[0]["cell_id"], 1)
        self.assertEqual(parsed[0]["errors"][0]["error_type"], "LLM_API_SEQUENCE")

    def test_missing_findings_key_returns_empty_result(self):
        result = map_api_sequence_findings_to_result({"not_findings": []}, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_empty_list_returns_empty_result(self):
        result = map_api_sequence_findings_to_result({"findings": []}, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_unknown_cell_id_is_dropped(self):
        findings_json = {"findings": [{"cell_id": 99, "message": "bad cell"}]}
        result = map_api_sequence_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_missing_message_is_dropped(self):
        findings_json = {"findings": [{"cell_id": 1}]}
        result = map_api_sequence_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_non_int_cell_id_is_dropped(self):
        findings_json = {"findings": [{"cell_id": "1", "message": "bad type"}]}
        result = map_api_sequence_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(result.path_results, [])

    def test_one_invalid_finding_does_not_drop_valid_ones(self):
        findings_json = {"findings": [
            {"cell_id": 1, "message": "valid"},
            {"cell_id": 99, "message": "invalid - unknown cell"},
        ]}
        result = map_api_sequence_findings_to_result(findings_json, self.notebook_IR)
        self.assertEqual(len(result.path_results), 1)
        self.assertEqual(result.path_results[0].error_infos[0].error_message, "valid")


if __name__ == "__main__":
    unittest.main()
