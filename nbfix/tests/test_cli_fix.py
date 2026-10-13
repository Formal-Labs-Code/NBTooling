import io
import json
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

pytest.importorskip("openai")

from nbcore.resource_utils.utils import TEST_RES_PATH

from nbfix.cli import _build_parser, main

SCRIPT = TEST_RES_PATH + "dataleak_script_flat_true.py"
CLEAN = Path(TEST_RES_PATH + "dataleak_script_flat_false.py").read_text(encoding="utf-8")


def _run(argv, *responses):
    client = MagicMock()
    client.chat_json.side_effect = list(responses)
    out = io.StringIO()
    with patch("nbcore.llm.config.default_client", return_value=client), \
            patch("sys.argv", ["nbfix", *argv]), redirect_stdout(out):
        main()
    return out.getvalue()


def _edit(code):
    return {"edits": [{"cell": 0, "code": code}], "explanation": "split first"}


class TestCliFix(unittest.TestCase):
    def test_attempts_default(self):
        self.assertEqual(_build_parser().parse_args(["--fix", "-f", SCRIPT]).attempts, 3)

    def test_fix_prints_json_result(self):
        result = json.loads(_run(["-f", SCRIPT, "--fix"], _edit(CLEAN)))
        self.assertEqual(result["status"], "verified")
        self.assertTrue(result["diff"].startswith(f"--- a/{SCRIPT}\n+++ b/{SCRIPT}\n"))

    def test_diff_prints_only_the_diff(self):
        out = _run(["-f", SCRIPT, "--fix", "--diff"], _edit(CLEAN))
        self.assertTrue(out.startswith(f"--- a/{SCRIPT}"))
        self.assertIn("+X_test_scaled = scaler.transform(X_test)\n", out)

    def test_diff_without_verified_repair_exits_with_problems(self):
        leaky = Path(SCRIPT).read_text(encoding="utf-8")
        with self.assertRaises(SystemExit) as exit_:
            _run(["-f", SCRIPT, "--fix", "--diff", "--attempts", "1"], _edit(leaky + "\n# no-op"))
        self.assertIn("No verified repair (status: unverified)", str(exit_.exception.code))
        self.assertIn("NBFix still reports", str(exit_.exception.code))


if __name__ == "__main__":
    unittest.main()
