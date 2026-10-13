import json
from argparse import ArgumentParser

from nbcore.events import RunBatchEvent
from nbcore.resource_utils.utils import load_notebook, read_json
from nbcore.resource_utils.script_loading import (
    cell_start_lines, file_line, first_code_cell, has_cell_markers, is_script, script_to_notebook,
)
from nbcore.analyses.dataleak_analysis import DATA_LEAK

from .analyses.idle_cell_analysis import IDLE
from .analyses.isolated_cell_analysis import ISOLATED
from .analyses.stale_cell_analysis import STALE
from .session_factory import new_session

# Batch/one-shot mode over a notebook or script file - the CLI counterpart
# to the live JupyterLab server extension.


# Analyses about notebook execution state (which cells ran, in what order) -
# meaningless for a script without cell markers, which is one cell run top
# to bottom.
NOTEBOOK_ONLY_ANALYSES = (STALE, IDLE, ISOLATED)


def read_input(filename, notebook):
    """The notebook dict for a .ipynb/.py file or an already-loaded dict; a
    .py script is converted with script_to_notebook (one cell per `# %%` /
    `# In[ ]:` marker, or a single cell without markers)."""
    assert not (filename and notebook)
    if notebook is not None:
        return notebook
    assert filename
    if filename.endswith(".py"):
        with open(filename, encoding="utf-8") as f:
            return script_to_notebook(f.read())
    return read_json(filename)


def _load_notebook(filename, notebook):
    session = new_session()
    notebook = read_input(filename, notebook)
    session.load(load_notebook(notebook["cells"]))
    return session


def add_file_lines(raw_results: str, notebook: dict) -> str:
    """For script input: adds `file_line` (the line in the .py file) to each
    error, and to its `source` when it has one."""
    starts = cell_start_lines(notebook)
    if not raw_results or not starts:
        return raw_results
    entries = json.loads(raw_results)
    for entry in entries:
        for err in entry.get("errors", []):
            err["file_line"] = file_line(starts, entry.get("cell_id"), err.get("line"))
            source = err.get("source")
            if isinstance(source, dict):
                source["file_line"] = file_line(starts, source.get("cell"), source.get("line"))
    return json.dumps(entries)


def nbfix(filename, notebook, analyses, start, level=5):
    results, _ = run_batch(filename, notebook, analyses, start, level)
    return results


def run_batch(filename, notebook, analyses, start, level=5):
    """nbfix(), also returning the loaded session (its cells carry
    per-cell CFG warnings - see analysis_warnings)."""
    notebook = read_input(filename, notebook)
    script = is_script(notebook)
    if script and not has_cell_markers(notebook):
        notebook_only = [a for a in analyses if a in NOTEBOOK_ONLY_ANALYSES]
        if notebook_only:
            raise ValueError(
                f"{', '.join(notebook_only)} only applies to notebooks or scripts with "
                "`# %%` cell markers; this script has none, so it is a single cell."
            )
    if script and start == 0:
        start = first_code_cell(notebook)

    session = _load_notebook(None, notebook)
    session.level = level
    session.add_analyses(analyses)
    event = RunBatchEvent(start)
    results = session.execute_event(event).dumps(True)

    return (add_file_lines(results, notebook) if script else results), session


def analysis_warnings(session, notebook) -> list[dict]:
    """What the parser degraded instead of rejecting - unsupported
    statements analyzed as no-ops, local calls it couldn't inline analyzed
    as opaque calls - as [{"cell", "line", "message"}] (+ "file_line" for
    scripts), deduplicated (a function inlined twice warns once)."""
    starts = cell_start_lines(notebook)
    out, seen = [], set()
    for cell_id, cell_IR in sorted(session.cells.items()):
        for warning in getattr(cell_IR.CFG, "warnings", []):
            key = (cell_id, warning.get("line"), warning.get("message"))
            if key in seen:
                continue
            seen.add(key)
            entry = {"cell": cell_id, "line": warning.get("line"), "message": warning.get("message")}
            mapped = file_line(starts, cell_id, warning.get("line"))
            if mapped is not None:
                entry["file_line"] = mapped
            out.append(entry)
    return out


def detect_bugs(filename, notebook, scope, cell_index, context_mode="deps", finding_types=None):
    """Batch/one-shot LLM bug detection over a notebook or script, via
    nbcore.llm.detect_bugs_event, which is generic over any cell dict."""
    try:
        from nbcore.llm.client import LLMClientError
        from nbcore.llm.detect_bugs_event import DetectBugsEvent
    except ImportError as exc:
        raise RuntimeError(
            "LLM bug detection is not installed - pip install nbcore[llm]"
        ) from exc

    session = _load_notebook(filename, notebook)
    if finding_types:
        session.add_analyses(list(finding_types))
        session.run_analyses(-1, list(finding_types))
    event = DetectBugsEvent(scope, cell_index, context_mode=context_mode, finding_types=finding_types)
    try:
        result = session.execute_event(event)
    except LLMClientError as exc:
        raise RuntimeError(str(exc)) from exc

    return result.dumps(True)


def detect_stale_cells(filename, notebook, cell_index):
    """Runs LLM-assisted stale-cell detection - an opt-in alternative to
    the deterministic StaleCellAnalysis (-a "Stale Cells Analysis"), not
    something layered on top of it.

    Loading a notebook fresh from a file leaves every cell's
    last_ran_code empty, the same characteristic the deterministic STALE
    analysis already has when driven from this CLI - meaningful edit-vs-
    original comparisons come from a live session (RunCellEvent/
    ChangeCellCodeEvent, as the JupyterLab server extension drives)."""
    try:
        from nbcore.llm.client import LLMClientError
        from .llm.detect_stale_cells_event import DetectStaleCellsEvent
    except ImportError as exc:
        raise RuntimeError(
            "LLM stale-cell detection is not installed - pip install nbfix[llm]"
        ) from exc

    session = _load_notebook(filename, notebook)
    original_code = session.cells[cell_index].last_ran_code
    event = DetectStaleCellsEvent(cell_index, original_code)
    try:
        result = session.execute_event(event)
    except LLMClientError as exc:
        raise RuntimeError(str(exc)) from exc

    return result.dumps(True)


def fix_leaks(filename, notebook, max_attempts=3):
    """LLM-suggested data-leak repair, verified by re-running the analysis
    (see repair.fix). Never writes the file - the result carries the
    proposed edits and a unified diff."""
    from .repair import fix  # repair builds on this module's run_batch

    source = None
    if filename and filename.endswith(".py"):
        with open(filename, encoding="utf-8") as f:
            source = f.read()
    return fix(read_input(filename, notebook), max_attempts=max_attempts, source=source, filename=filename)


def _build_parser() -> ArgumentParser:
    parser = ArgumentParser(description="NBFix - live and batch notebook diagnostics")
    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument("-f", "--filename", type=str, help='Notebook (.ipynb) or Python script (.py) file.')
    group.add_argument("-n", "--notebook", type=str, help='Notebook json string.')
    parser.add_argument("-a", "--analyses", nargs="+", type=str, default=[], help='Analyses to perform.')
    parser.add_argument("-s", "--start", type=int, default=0, help='Starting cell ID (default is 0).')
    parser.add_argument("-l", "--level", nargs="?", type=int, default=5, help='Depth level of the analysis (default is inf).')
    parser.add_argument(
        "--detect-bugs", action="store_true",
        help='Run LLM-assisted bug detection instead of -a. Requires the llm extra (pip install nbcore[llm]).',
    )
    parser.add_argument(
        "--scope", choices=["cell", "subgraph", "full"], default="full",
        help='Scope for --detect-bugs (default: full notebook).',
    )
    parser.add_argument(
        "--cell", type=int, default=None,
        help='Cell index to check. Required when --scope is cell or subgraph, or with --detect-stale-cells.',
    )
    parser.add_argument(
        "--context-mode", choices=["none", "deps"], default="deps",
        help='Context mode for --detect-bugs: "deps" includes the dependency '
             'graph (default), "none" omits it entirely.',
    )
    parser.add_argument(
        "--finding-types", nargs="+", choices=[DATA_LEAK, STALE, IDLE, ISOLATED], default=None,
        help='Deterministic analysis names to run and feed the LLM as extra '
             'context, for --detect-bugs.',
    )
    parser.add_argument(
        "--detect-stale-cells", action="store_true",
        help='Run LLM-assisted stale-cell detection instead of the deterministic '
             '"Stale Cells Analysis" (-a). Requires the llm extra (pip install '
             'nbfix[llm]). Requires --cell.',
    )
    parser.add_argument(
        "--fix", action="store_true",
        help='Suggest an LLM repair for every data leak, verified by re-running the '
             'analysis on the patched code. Prints the result as JSON; never writes '
             'the file. Requires the llm extra (pip install nbfix[llm]).',
    )
    parser.add_argument(
        "--attempts", type=int, default=3,
        help='Maximum LLM attempts for --fix (default 3).',
    )
    parser.add_argument(
        "--diff", action="store_true",
        help='With --fix, print only the unified diff of a verified repair (for '
             'git apply); exits with an error if no repair was verified.',
    )
    return parser


def main():
    parser = _build_parser()
    args = parser.parse_args()

    if args.detect_bugs:
        if args.scope in ("cell", "subgraph") and args.cell is None:
            parser.error(f"--cell is required when --scope {args.scope}")
        try:
            results = detect_bugs(
                args.filename, args.notebook, args.scope, args.cell,
                args.context_mode, args.finding_types,
            )
        except RuntimeError as exc:
            raise SystemExit(f"Error: {exc}")
        print(results)
        return

    if args.fix:
        if args.attempts < 1:
            parser.error("--attempts must be at least 1")
        try:
            result = fix_leaks(args.filename, args.notebook, args.attempts)
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"Error: {exc}")
        if not args.diff:
            print(json.dumps(result, indent=2))
        elif result["status"] == "verified":
            print(result["diff"], end="")
        else:
            problems = "".join(f"\n  - {p}" for p in result.get("problems", []))
            raise SystemExit(f"No verified repair (status: {result['status']}).{problems}")
        return

    if args.detect_stale_cells:
        if args.cell is None:
            parser.error("--cell is required for --detect-stale-cells")
        try:
            results = detect_stale_cells(args.filename, args.notebook, args.cell)
        except RuntimeError as exc:
            raise SystemExit(f"Error: {exc}")
        print(results)
        return

    try:
        results = nbfix(args.filename, args.notebook, args.analyses, args.start, args.level)
    except ValueError as exc:
        raise SystemExit(f"Error: {exc}")
    print(results)

if __name__ == "__main__":
    main()
