import json
import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

# scripts/ isn't a package (it's a dev-only tool, not shipped/installed) -
# add it to sys.path so the scoring function can be imported for testing.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

from benchmark_llm import (  # noqa: E402
    ModelConfig,
    _discover_eval_notebooks,
    _load_expected,
    _load_nbfix,
    parse_config,
    run_once,
    score_findings,
)


class TestParseConfig(unittest.TestCase):
    def test_parses_name_model_base_url(self):
        config = parse_config("small=qwen2.5-coder:14b@http://localhost:11434/v1")
        self.assertEqual(config.name, "small")
        self.assertEqual(config.model, "qwen2.5-coder:14b")
        self.assertEqual(config.base_url, "http://localhost:11434/v1")

    def test_missing_parts_raises(self):
        with self.assertRaises(ValueError):
            parse_config("small=qwen2.5-coder:14b")  # no @base_url

        with self.assertRaises(ValueError):
            parse_config("qwen2.5-coder:14b@http://localhost:11434/v1")  # no name=


class TestScoreFindings(unittest.TestCase):
    def test_exact_match(self):
        score = score_findings([(0, 3)], [(0, 3)])
        self.assertEqual(score["true_positives"], 1)
        self.assertEqual(score["false_positives"], 0)
        self.assertEqual(score["false_negatives"], 0)
        self.assertEqual(score["precision"], 1.0)
        self.assertEqual(score["recall"], 1.0)
        self.assertEqual(score["f1"], 1.0)

    def test_within_line_tolerance_still_matches(self):
        score = score_findings([(0, 5)], [(0, 3)], line_tolerance=2)
        self.assertEqual(score["true_positives"], 1)

    def test_outside_line_tolerance_does_not_match(self):
        score = score_findings([(0, 6)], [(0, 3)], line_tolerance=2)
        self.assertEqual(score["true_positives"], 0)

    def test_different_cell_id_does_not_match_even_if_line_close(self):
        score = score_findings([(1, 3)], [(0, 3)])
        self.assertEqual(score["true_positives"], 0)

    def test_extra_actual_findings_are_false_positives(self):
        score = score_findings([(0, 1), (0, 2)], [(0, 1)])
        self.assertEqual(score["true_positives"], 1)
        self.assertEqual(score["false_positives"], 1)
        self.assertEqual(score["precision"], 0.5)
        self.assertEqual(score["recall"], 1.0)

    def test_missing_expected_findings_are_false_negatives(self):
        score = score_findings([(0, 1)], [(0, 1), (1, 5)])
        self.assertEqual(score["true_positives"], 1)
        self.assertEqual(score["false_negatives"], 1)
        self.assertEqual(score["recall"], 0.5)

    def test_no_double_counting_same_actual_finding_twice(self):
        # Two expected findings both near the same single actual finding -
        # it should only count as one true positive, not two.
        score = score_findings([(0, 3)], [(0, 2), (0, 4)])
        self.assertEqual(score["true_positives"], 1)
        self.assertEqual(score["false_negatives"], 1)

    def test_empty_actual_and_expected(self):
        score = score_findings([], [])
        self.assertEqual(score["precision"], 0.0)
        self.assertEqual(score["recall"], 0.0)
        self.assertEqual(score["f1"], 0.0)

    def test_empty_expected_with_actual_findings_zero_recall(self):
        score = score_findings([(0, 1)], [])
        self.assertEqual(score["recall"], 0.0)
        self.assertEqual(score["false_positives"], 1)


class TestDiscoverEvalNotebooks(unittest.TestCase):
    def test_walks_one_level_deep_by_bug_class(self):
        with tempfile.TemporaryDirectory() as eval_dir:
            os.makedirs(os.path.join(eval_dir, "class_a"))
            os.makedirs(os.path.join(eval_dir, "class_b"))
            open(os.path.join(eval_dir, "class_a", "ex1.ipynb"), "w").close()
            open(os.path.join(eval_dir, "class_a", "ex1.ipynb.expected.json"), "w").close()
            open(os.path.join(eval_dir, "class_b", "clean1.ipynb"), "w").close()

            found = sorted(_discover_eval_notebooks(eval_dir))
            # the .expected.json sidecar must not be picked up as a notebook.
            self.assertEqual(found, [("class_a", "ex1.ipynb"), ("class_b", "clean1.ipynb")])

    def test_ignores_non_directory_entries(self):
        with tempfile.TemporaryDirectory() as eval_dir:
            open(os.path.join(eval_dir, "README.md"), "w").close()
            os.makedirs(os.path.join(eval_dir, "class_a"))
            open(os.path.join(eval_dir, "class_a", "ex1.ipynb"), "w").close()

            found = list(_discover_eval_notebooks(eval_dir))
            self.assertEqual(found, [("class_a", "ex1.ipynb")])


class TestLoadExpected(unittest.TestCase):
    def test_returns_none_when_sidecar_absent(self):
        with tempfile.TemporaryDirectory() as eval_dir:
            os.makedirs(os.path.join(eval_dir, "class_a"))
            self.assertIsNone(_load_expected(eval_dir, "class_a", "ex1.ipynb"))

    def test_loads_sidecar_when_present(self):
        with tempfile.TemporaryDirectory() as eval_dir:
            os.makedirs(os.path.join(eval_dir, "class_a"))
            sidecar = os.path.join(eval_dir, "class_a", "ex1.ipynb.expected.json")
            with open(sidecar, "w") as f:
                json.dump([{"cell_id": 0, "path": [0], "errors": []}], f)

            self.assertEqual(
                _load_expected(eval_dir, "class_a", "ex1.ipynb"),
                [{"cell_id": 0, "path": [0], "errors": []}],
            )

    def test_empty_list_sidecar_loads_as_empty_not_none(self):
        # Clean llm_bench fixtures ship an explicit [] - must be
        # distinguishable from "no sidecar at all" (None).
        with tempfile.TemporaryDirectory() as eval_dir:
            os.makedirs(os.path.join(eval_dir, "class_a"))
            sidecar = os.path.join(eval_dir, "class_a", "clean1.ipynb.expected.json")
            with open(sidecar, "w") as f:
                json.dump([], f)

            self.assertEqual(_load_expected(eval_dir, "class_a", "clean1.ipynb"), [])


class TestLoadNBFix(unittest.TestCase):
    def test_loads_via_real_fast_path(self):
        nbfix = _load_nbfix({"cells": [{"cell_type": "code", "source": "x = 1"}]})
        self.assertEqual(set(nbfix.cells), {0})

    def test_falls_back_to_resilient_loader_on_unsupported_construct(self):
        # A lambda is a real, currently-unsupported construct (see
        # parser/README.md's backlog) - the real loader should raise, and
        # _load_nbfix should recover via the resilient loader rather than
        # propagating.
        nbfix = _load_nbfix({
            "cells": [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "code", "source": "f = lambda y: y + 1"},
            ]
        })
        self.assertIn(0, nbfix.cells)


class TestRunOnce(unittest.TestCase):
    def setUp(self):
        self.nbfix = _load_nbfix({
            "cells": [
                {"cell_type": "code", "source": "x = 1"},
                {"cell_type": "code", "source": "y = x + 1"},
            ]
        })
        self.config = ModelConfig(name="test", model="test-model", base_url="http://localhost:1234/v1")

    def test_context_config_none_omits_dependency_graph_from_prompt(self):
        with patch("benchmark_llm.LLMClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat_json_with_usage.return_value = ({"findings": []}, {})
            mock_cls.return_value = mock_client

            run_once("ex1.ipynb", "class_a", self.nbfix, self.config, "none", expected=None)

            _, user_prompt = mock_client.chat_json_with_usage.call_args.args
            self.assertNotIn("## Dependency graph", user_prompt)

    def test_context_config_deps_includes_dependency_graph(self):
        with patch("benchmark_llm.LLMClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat_json_with_usage.return_value = ({"findings": []}, {})
            mock_cls.return_value = mock_client

            run_once("ex1.ipynb", "class_a", self.nbfix, self.config, "deps", expected=None)

            _, user_prompt = mock_client.chat_json_with_usage.call_args.args
            self.assertIn("## Dependency graph", user_prompt)

    def test_records_bug_class_and_context_config(self):
        with patch("benchmark_llm.LLMClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat_json_with_usage.return_value = ({"findings": []}, {})
            mock_cls.return_value = mock_client

            result = run_once("ex1.ipynb", "class_a", self.nbfix, self.config, "deps", expected=None)

            self.assertEqual(result.bug_class, "class_a")
            self.assertEqual(result.context_config, "deps")

    def test_captures_token_usage(self):
        usage = {"prompt_tokens": 20, "completion_tokens": 8, "total_tokens": 28}
        with patch("benchmark_llm.LLMClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat_json_with_usage.return_value = ({"findings": []}, usage)
            mock_cls.return_value = mock_client

            result = run_once("ex1.ipynb", "class_a", self.nbfix, self.config, "deps", expected=None)

            self.assertEqual(result.tokens, usage)

    def test_scores_against_expected_when_given(self):
        findings_json = {
            "findings": [
                {"cell_ids": [1], "line": 1, "label": "y", "severity": "critical", "message": "bug"}
            ]
        }
        expected = [{"cell_id": 1, "path": [1], "errors": [{"line": 1, "label": "y", "error_type": "x", "message": "bug"}]}]
        with patch("benchmark_llm.LLMClient") as mock_cls:
            mock_client = MagicMock()
            mock_client.chat_json_with_usage.return_value = (findings_json, {})
            mock_cls.return_value = mock_client

            result = run_once("ex1.ipynb", "class_a", self.nbfix, self.config, "deps", expected=expected)

            self.assertEqual(result.score["true_positives"], 1)
            self.assertEqual(result.score["precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
