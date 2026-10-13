import asyncio
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from nbcore.resource_utils.utils import TEST_RES_PATH, read_json

from nbfix.mcp_server import fix, mcp

SCRIPT = TEST_RES_PATH + "dataleak_script_flat_true.py"
CLEAN = Path(TEST_RES_PATH + "dataleak_script_flat_false.py").read_text(encoding="utf-8")


def _fix(notebook, *responses):
    client = MagicMock()
    client.chat_json.side_effect = list(responses)
    with patch("nbcore.llm.config.default_client", return_value=client):
        return fix(notebook)


def _edit(cell, code):
    return {"edits": [{"cell": cell, "code": code}], "explanation": "fixed"}


class TestMCPFix(unittest.TestCase):
    def test_fix_is_an_mcp_tool(self):
        names = [tool.name for tool in asyncio.run(mcp.list_tools())]
        self.assertEqual(sorted(names), ["analyze", "fix"])

    def test_script_path_diff_is_of_the_file(self):
        result = _fix(SCRIPT, _edit(0, CLEAN))
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["diff"].startswith(
            "--- a/dataleak_script_flat_true.py\n+++ b/dataleak_script_flat_true.py\n"))

    def test_script_source_diff_is_of_the_script(self):
        source = Path(SCRIPT).read_text(encoding="utf-8")
        for notebook in (source, {"script": source}):
            with self.subTest(type(notebook).__name__):
                result = _fix(notebook, _edit(0, CLEAN))
                self.assertEqual(result["status"], "verified")
                self.assertTrue(result["diff"].startswith("--- a/script.py\n+++ b/script.py\n"))

    def test_notebook_diff_is_per_cell(self):
        result = _fix(read_json(TEST_RES_PATH + "dataleak_overlap_true.ipynb"),
                      _edit(2, "lr = LogisticRegression()\na = lr.fit(X_train, y_train)"))
        self.assertEqual(result["status"], "verified")
        self.assertEqual(result["edits"], [{"cell": 2, "code": "lr = LogisticRegression()\na = lr.fit(X_train, y_train)"}])
        self.assertIn("--- a/notebook (cell 2)", result["diff"])


if __name__ == "__main__":
    unittest.main()
