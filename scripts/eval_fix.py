"""
Runs NBFix's LLM data-leak repair (nbfix.repair.fix) over every leaky fixture
in tests/resources/ and reports how many repairs NBFix could verify.

Not part of the installed nbfix package - a dev-only evaluation tool. Uses the
same LLM configuration as the product (NBCORE_LLM_BASE_URL/NBCORE_LLM_MODEL/
NBCORE_LLM_API_KEY, defaulting to a local Ollama), so comparing models is a
matter of setting env vars:

    python scripts/eval_fix.py
    NBCORE_LLM_MODEL=qwen2.5-coder:7b python scripts/eval_fix.py --out results.json

"verified" means NBFix's own analysis accepts the patch, not that the patch is
what a person would write - read the diffs (--out) before trusting the numbers.
"""
import argparse
import glob
import json
import os
import time

from nbcore.llm.config import DEFAULT_MODEL
from nbcore.resource_utils.script_loading import script_to_notebook
from nbcore.resource_utils.utils import read_json

from nbfix.repair import fix

FIXTURES = "tests/resources/dataleak*_true.*"


def _load(path):
    if path.endswith(".py"):
        with open(path, encoding="utf-8") as f:
            source = f.read()
        return script_to_notebook(source), source
    return read_json(path), None


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--attempts", type=int, default=3, help="Maximum LLM attempts per fixture (default 3).")
    parser.add_argument("--out", help="Write every result, including diffs and problems, to this JSON file.")
    parser.add_argument("fixtures", nargs="*", help=f"Fixtures to run (default: {FIXTURES}).")
    args = parser.parse_args()

    paths = args.fixtures or sorted(glob.glob(FIXTURES))
    print(f"model: {os.environ.get('NBCORE_LLM_MODEL', DEFAULT_MODEL)}, attempts: {args.attempts}\n")
    print(f"{'fixture':48} {'status':12} {'attempts':>8} {'seconds':>8}")

    results = []
    for path in paths:
        notebook, source = _load(path)
        started = time.monotonic()
        try:
            result = fix(notebook, max_attempts=args.attempts, source=source, filename=os.path.basename(path))
        except RuntimeError as exc:
            result = {"status": "error", "attempts": 0, "problems": [str(exc)]}
        result["fixture"] = path
        result["seconds"] = round(time.monotonic() - started, 1)
        results.append(result)
        print(f"{os.path.basename(path):48} {result['status']:12} {result['attempts']:>8} {result['seconds']:>8}")

    counts = {}
    for result in results:
        counts[result["status"]] = counts.get(result["status"], 0) + 1
    print("\n" + ", ".join(f"{status}: {count}" for status, count in sorted(counts.items())))

    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"results written to {args.out}")


if __name__ == "__main__":
    main()
