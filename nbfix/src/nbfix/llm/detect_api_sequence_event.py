from nbcore.analyses.runner.analysis_results import Result
from nbcore.events import Event
from .api_sequence_context_builder import build_api_sequence_context
from .api_sequence_prompts import API_SEQUENCE_SYSTEM_PROMPT, build_api_sequence_user_prompt
from .api_sequence_result_mapping import map_api_sequence_findings_to_result
from nbcore.llm.config import default_client


class DetectApiSequenceEvent(Event):
    """
    LLM-based check for whether any cell in the notebook violates a
    library API's call-order contract (e.g. calling .predict() on a
    scikit-learn estimator that was never .fit()) - scenario 1 of the
    two use cases discussed for this bug class: warn before the user
    runs a cell, not a separate "scan for all paths" feature (deferred -
    see this module's sibling api_sequence_prompts.py docstring).

    Scans the whole notebook in a single LLM call, not one target cell
    at a time - an earlier version took a cell_index and checked just
    that cell, but the only real caller (manager.ts's
    _scanAllCellsForApiSequence) always needed every cell checked
    anyway, making N single-cell calls per scan pure latency waste.

    focus_cell (optional) narrows the prompt to that cell's dependency-
    connected component instead of the whole notebook - see
    api_sequence_context_builder.py's docstring. checked_cells is set
    after execute() runs, to every cell_id actually *considered* -
    which, thanks to the method-call filter, can be larger than what
    actually went into the prompt (a component with no method calls at
    all is confidently resolved as "no violation" without ever asking
    the LLM, but still counts as checked). The caller needs this to know
    which cells' *prior* findings are safe to discard versus which are
    untouched by a narrowed scan (see dispatch.py/handlers.py for how
    this rides back to the frontend, and manager.ts's _sendChangeCell
    for the merge that depends on it). A plain attribute, not part of
    Result, since Result is a generic shape every event returns and this
    is specific to this one event's narrowing.

    Cares about execution history (not_yet_run in the context), but
    needs no explicit original_code parameter the way
    DetectStaleCellsEvent does - it isn't diffing against what a cell
    used to contain, just checking whether each cell's own current code
    has actually executed yet. Reads notebook_IR as-is at call time; no
    special ordering relative to RunCellEvent.

    Never mutates nbfix's state - same discipline as DetectBugsEvent/
    DetectStaleCellsEvent (both have their own regression tests for
    this).

    dependency_edges, if given, overrides build_api_sequence_context's own
    build_dependency_edges output entirely - same escape hatch as
    DetectBugsEvent's dependency_edges param.
    """

    def __init__(self, focus_cell: int = None, client=None, dependency_edges=None):
        self.focus_cell = focus_cell
        self.client = client or default_client()
        self.dependency_edges = dependency_edges
        self.checked_cells: set[int] = set()

    def execute(self, nbfix) -> Result:
        # focus_cell can be a non-code (e.g. markdown) cell's position -
        # the frontend watches every cell for edits, not just code ones,
        # and notebook_IR only ever contains code cells (see
        # NBFix.load_notebook). Nothing to check in that case - a plain
        # KeyError here (confirmed via a live report: editing a markdown
        # cell crashed with "Event execution failed: 6", Python's
        # KeyError stringifying to just the bare key) isn't a case to
        # propagate as a real failure.
        if self.focus_cell is not None and self.focus_cell not in nbfix.cells:
            return Result()

        context = build_api_sequence_context(nbfix.cells, self.focus_cell, self.dependency_edges)
        self.checked_cells = set(context.considered_cells)

        if not context.cells:
            # Every considered component had zero method calls - this
            # bug class provably cannot occur without one, so there is
            # nothing worth asking the LLM. Not an error case (unlike
            # the markdown-cell guard above); checked_cells is still
            # correctly non-empty so any stale finding on these cells
            # still gets cleared by the caller's merge.
            return Result()

        user_prompt = build_api_sequence_user_prompt(context)
        findings_json = self.client.chat_json(API_SEQUENCE_SYSTEM_PROMPT, user_prompt)
        return map_api_sequence_findings_to_result(findings_json, nbfix.cells)
