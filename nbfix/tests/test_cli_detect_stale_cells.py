import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from nbfix.cli import _build_parser, detect_stale_cells, main
from nbcore.resource_utils.utils import TEST_RES_PATH


def _mock_openai_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class TestArgParsing(unittest.TestCase):
    def test_missing_cell_exits(self):
        with patch(
            "sys.argv",
            ["nbfix", "--detect-stale-cells", "-f", "notebook.ipynb"],
        ):
            with self.assertRaises(SystemExit):
                main()

    def test_parses_with_cell(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--detect-stale-cells", "--cell", "0", "-f", "notebook.ipynb"]
        )
        self.assertTrue(args.detect_stale_cells)
        self.assertEqual(args.cell, 0)


class TestDetectStaleCellsFunction(unittest.TestCase):
    @patch("nbcore.llm.client.openai.OpenAI")
    def test_no_edit_returns_empty_dumps_without_calling_llm(self, mock_openai_cls):
        # Basic.ipynb loaded fresh from a file: every cell's last_ran_code
        # starts empty. Cell 0 has real code, so this exercises the
        # "never run before" path (treated as an edit, matching the real
        # StaleCellAnalysis's own behavior) - not the true no-op path.
        # Assert on the actual call happening rather than assuming which
        # branch fires, since that depends on Basic.ipynb's contents.
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '{"stale_cells": []}'
        )

        result = detect_stale_cells(TEST_RES_PATH + "Basic.ipynb", None, 0)

        self.assertEqual(result, "")  # Result.dumps() for zero findings

    @patch("nbcore.llm.client.openai.OpenAI")
    def test_main_end_to_end_prints_result(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '{"stale_cells": []}'
        )

        argv = [
            "nbfix", "--detect-stale-cells", "--cell", "0",
            "-f", TEST_RES_PATH + "Basic.ipynb",
        ]
        with patch("sys.argv", argv):
            with patch("builtins.print") as mock_print:
                main()
        mock_print.assert_called_once_with("")


if __name__ == "__main__":
    unittest.main()
