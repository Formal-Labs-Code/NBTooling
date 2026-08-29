import unittest
from unittest.mock import patch

from nbcore.resource_utils.utils import TEST_RES_PATH

from nbcompile.cli import _build_parser, compile_notebook, main


class TestArgParsing(unittest.TestCase):
    def test_filename_required(self):
        parser = _build_parser()
        with self.assertRaises(SystemExit):
            parser.parse_args([])

    def test_output_defaults_to_none(self):
        parser = _build_parser()
        args = parser.parse_args(["-f", "notebook.ipynb"])
        self.assertIsNone(args.output)


class TestCompileNotebook(unittest.TestCase):
    def test_produces_a_linear_script(self):
        script = compile_notebook(TEST_RES_PATH + "dataleak_fit_transform_split_true.ipynb")
        self.assertIn("StandardScaler", script)
        self.assertIn("train_test_split", script)
        # No cycle in this fixture - no loop wrapper should appear.
        self.assertNotIn("while True:", script)


class TestMain(unittest.TestCase):
    def test_prints_to_stdout_by_default(self):
        argv = ["nbcompile", "-f", TEST_RES_PATH + "dataleak_fit_transform_split_true.ipynb"]
        with patch("sys.argv", argv):
            with patch("builtins.print") as mock_print:
                main()
        mock_print.assert_called_once()
        self.assertIn("StandardScaler", mock_print.call_args.args[0])

    def test_writes_to_output_file_when_given(self):
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = os.path.join(tmpdir, "out.py")
            argv = [
                "nbcompile",
                "-f", TEST_RES_PATH + "dataleak_fit_transform_split_true.ipynb",
                "-o", out_path,
            ]
            with patch("sys.argv", argv):
                main()
            with open(out_path) as f:
                content = f.read()
            self.assertIn("StandardScaler", content)


if __name__ == "__main__":
    unittest.main()
