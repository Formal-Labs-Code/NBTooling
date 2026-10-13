from nbcore.analyses.runner.analysis_results import Result
from nbcore.events import Event
from nbcore.llm.config import default_client
from .stale_context_builder import build_stale_context
from .stale_prompts import STALE_SYSTEM_PROMPT, build_stale_user_prompt
from .stale_result_mapping import map_stale_findings_to_result


class DetectStaleCellsEvent(Event):
    """
    LLM-based alternative to StaleCellAnalysis, run *instead of* it, not
    alongside it as extra context - a different comparison than
    DetectBugsEvent's context_mode/finding_types axis, which is about
    whether structural context helps LLM bug-finding. This is about
    whether an LLM can replace a deterministic analysis NBFix already
    has, for a bug class defined by *execution history* rather than a
    static code snapshot - see stale_context_builder.py's docstring.

    Deliberately separate from StaleCellAnalysis/events.RunCellEvent, not
    a replacement wired into the same automatic path - the user asked for
    this specifically: "I'd like to be able to choose when I run the
    tool." RunCellEvent still owns updating last_ran_code and triggering
    the real deterministic check on every cell run; this event is an
    additional, opt-in check a caller (CLI flag, or a JupyterLab command)
    triggers explicitly. Never mutates nbfix's state itself - same
    discipline as DetectBugsEvent (see that class's docstring and its
    active_analyses-unchanged regression test).

    original_code is REQUIRED and must be the code cell_index was run
    with *before* the edit being checked - the caller must capture it
    themselves, before triggering the real re-run. This was NOT the
    original design (an earlier version read
    notebook_IR[cell_index].last_ran_code internally instead) - changed
    after a live end-to-end check caught a real ordering bug: RunCellEvent
    overwrites last_ran_code to match the new cell_code as part of the
    *same* call that actually executes the cell in the kernel. That means
    there's no single moment where both (a) the cell's kernel value is
    genuinely fresh (required for the "the edited cell was just
    re-executed" premise this event's prompt depends on - see
    stale_prompts.py) and (b) last_ran_code still holds the pre-edit
    code. Reading it internally would force a choice between an event
    that never sees a real diff, or one whose prompt premise is false.
    Requiring the caller to capture original_code explicitly removes the
    conflict entirely - the correct sequence is:

        original_code = nbfix.cells[cell_index].last_ran_code
        nbfix.execute_event(ChangeCellCodeEvent(new_code, cell_index, with_result=False))
        nbfix.execute_event(RunCellEvent(cell_index))  # actually re-executes; refreshes the kernel
        result = nbfix.execute_event(DetectStaleCellsEvent(cell_index, original_code))

    If original_code == the cell's current code, nothing changed - no
    LLM call is made and an empty Result is returned immediately, same
    as the real StaleCellAnalysis's own "no changed_vars, no
    propagation" behavior (including the same quirk it already has: a
    cell that's never been run before, where the caller has nothing
    meaningful to pass as original_code, looks identical to "changed
    from empty" and does trigger a check - see
    tests/resources/llm_bench_stale/_author_stale_fixtures.py's module
    docstring for where this was first confirmed empirically).
    """

    def __init__(self, cell_index: int, original_code: str, client=None):
        self.cell_index = cell_index
        self.original_code = original_code
        self.client = client or default_client()

    def execute(self, nbfix) -> Result:
        ir = nbfix.cells[self.cell_index]
        if self.original_code == ir.cell_code:
            return Result()

        context = build_stale_context(nbfix.cells, self.cell_index, self.original_code)
        user_prompt = build_stale_user_prompt(context)
        findings_json = self.client.chat_json(STALE_SYSTEM_PROMPT, user_prompt)
        return map_stale_findings_to_result(findings_json, nbfix.cells)
