import json
import unittest

from nbcore.analyses.runner.analysis_results import ErrorInfo, ErrorType, PathResult, Result


class TestResultDumps(unittest.TestCase):
    def test_error_without_a_line_dumps_valid_json(self):
        result = Result()
        result.add_path_result(PathResult([0], [ErrorInfo(0, None, "x", ErrorType.TERMINAL, "message")]))
        self.assertEqual(json.loads(result.dumps())[0]["errors"][0]["line"], None)


if __name__ == "__main__":
    unittest.main()
