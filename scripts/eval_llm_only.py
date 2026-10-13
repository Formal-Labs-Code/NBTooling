"""
Baseline for NBFix's `fix`: the same LLM detecting and fixing data leaks on
its own - no NBFix analysis in the prompt, no checks, no retries.

Runs every leaky (`dataleak*_true.*`) and clean (`dataleak*_false.*`) fixture
through one generic prompt and records what the model reported and changed.
It does not judge correctness: the results are meant to be read (see
benchmark_results/llm_only/README.md). As evidence for that review, it also
records whether each patched notebook parses and what NBFix's data-leak
analysis finds in it.

Not part of the installed nbfix package - a dev-only evaluation tool. Uses
the same LLM configuration as the product (NBCORE_LLM_BASE_URL/MODEL/API_KEY):

    python scripts/eval_llm_only.py --out results.json
    NBCORE_LLM_MODEL=qwen2.5-coder:7b python scripts/eval_llm_only.py --out results.json
"""
import argparse
import copy
import glob
import json
import os
import time

from nbcore.llm.client import LLMClientError
from nbcore.llm.config import DEFAULT_MODEL, default_client
from nbcore.resource_utils.script_loading import script_to_notebook
from nbcore.resource_utils.utils import read_json

from nbfix.code_checks import parse_cell
from nbfix.findings import analyze_data_leaks
from nbfix.repair import _apply, _code_cells, _diff

FIXTURES = ["tests/resources/dataleak*_true.*", "tests/resources/dataleak*_false.*"]

SYSTEM_PROMPT = """You review Python data science code - Jupyter notebook cells, or the cells of a Python script - for data leakage between training and test data, and fix any you find.

Respond with a single JSON object of exactly this shape and nothing else:

{
  "leaks": [{"cell": <int>, "description": "<what leaks and why>"}],
  "edits": [{"cell": <int>, "code": "<complete new source of the cell>"}],
  "explanation": "<one or two sentences on what you changed and why>"
}

"leaks" lists every data leak you find, by the cell it is in; leave it empty if there is none. "edits" gives the complete new source of each cell you change to fix them; leave out cells you don't change, and leave "edits" empty if there is nothing to fix."""


def user_prompt(cells: dict[int, str]) -> str:
    return "## Cells\n\n" + "\n\n".join(f"### Cell {i}\n```python\n{code}\n```" for i, code in cells.items())


def _load(path):
    if path.endswith(".py"):
        with open(path, encoding="utf-8") as f:
            source = f.read()
        return script_to_notebook(source), source
    return read_json(path), None


def run_one(path, client) -> dict:
    notebook, source = _load(path)
    cells = _code_cells(notebook)
    started = time.monotonic()
    try:
        response = client.chat_json(SYSTEM_PROMPT, user_prompt(cells))
    except LLMClientError as exc:
        return {"fixture": path, "error": str(exc)}
    seconds = round(time.monotonic() - started, 1)

    edits = [e for e in response.get("edits") or []
             if isinstance(e, dict) and e.get("cell") in cells and isinstance(e.get("code"), str)]
    record = {
        "fixture": path,
        "seconds": seconds,
        "leaks_reported": response.get("leaks"),
        "explanation": response.get("explanation"),
        "edits": edits,
        "nbfix_findings_before": [f["message"] for f in analyze_data_leaks(copy.deepcopy(notebook))["findings"]],
    }
    if not edits:
        return record

    patched = _apply(notebook, edits)
    record["diff"] = _diff(notebook, cells, edits, source, os.path.basename(path))
    try:
        for code in _code_cells(patched).values():
            parse_cell(code)
        record["parses"] = True
    except SyntaxError as exc:
        record["parses"] = False
        record["syntax_error"] = str(exc)
        return record
    try:
        record["nbfix_findings_after"] = [f["message"] for f in analyze_data_leaks(copy.deepcopy(patched))["findings"]]
    except Exception as exc:  # the analysis rejecting the patched code
        record["nbfix_findings_after"] = f"analysis failed: {type(exc).__name__}: {exc}"
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--out", required=True, help="Write every result to this JSON file.")
    parser.add_argument("fixtures", nargs="*", help="Fixtures to run (default: all leaky and clean fixtures).")
    args = parser.parse_args()

    paths = args.fixtures or sorted(p for pattern in FIXTURES for p in glob.glob(pattern))
    client = default_client()
    print(f"model: {os.environ.get('NBCORE_LLM_MODEL', DEFAULT_MODEL)}\n")
    results = []
    for path in paths:
        record = run_one(path, client)
        results.append(record)
        print(f"{os.path.basename(path):48} leaks reported: {len(record.get('leaks_reported') or []):2}  "
              f"edited cells: {len(record.get('edits') or []):2}  {record.get('seconds', '-')}s")
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nresults written to {args.out}")


if __name__ == "__main__":
    main()
