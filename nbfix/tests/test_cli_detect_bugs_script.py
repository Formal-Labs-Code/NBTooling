import unittest
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from nbfix.cli import _build_parser, detect_bugs, main
from nbcore.resource_utils.utils import TEST_RES_PATH


def _mock_openai_response(content: str):
    response = MagicMock()
    response.choices = [MagicMock(message=MagicMock(content=content))]
    return response


class TestArgParsing(unittest.TestCase):
    def test_scope_cell_without_cell_index_exits(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["--detect-bugs", "--scope", "cell", "-f", "sample_script.py"]
        )
        self.assertIsNone(args.cell)  # parsing itself succeeds

        # main()'s post-parse validation is what actually rejects this.
        with patch(
            "sys.argv",
            ["nbfix", "--detect-bugs", "--scope", "cell", "-f", "sample_script.py"],
        ):
            with self.assertRaises(SystemExit):
                main()

    def test_scope_full_does_not_require_cell(self):
        parser = _build_parser()
        args = parser.parse_args(["--detect-bugs", "-f", "sample_script.py"])
        self.assertEqual(args.scope, "full")
        self.assertIsNone(args.cell)


class TestDetectBugsFunction(unittest.TestCase):
    @patch("nbcore.llm.client.openai.OpenAI")
    def test_detect_bugs_returns_dumps_output(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '{"findings": []}'
        )

        result = detect_bugs(TEST_RES_PATH + "sample_script.py", None, "full", None)

        self.assertEqual(result, "")  # Result.dumps() for zero findings

    @patch("nbcore.llm.client.openai.OpenAI")
    def test_main_end_to_end_prints_result(self, mock_openai_cls):
        mock_client = mock_openai_cls.return_value
        mock_client.chat.completions.create.return_value = _mock_openai_response(
            '{"findings": []}'
        )

        argv = [
            "nbfix",
            "--detect-bugs",
            "--scope",
            "full",
            "-f",
            TEST_RES_PATH + "sample_script.py",
        ]
        with patch("sys.argv", argv):
            with patch("builtins.print") as mock_print:
                main()
        mock_print.assert_called_once_with("")


if __name__ == "__main__":
    unittest.main()
