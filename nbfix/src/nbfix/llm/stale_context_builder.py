"""
Context for LLM-based stale-cell detection - a different concern from
context_builder.py's BugDetectionContext, kept in its own module rather
than folded in (matching this package's one-concern-per-file convention,
e.g. repair_prompts.py being kept separate from prompts.py).

Staleness isn't a property of a single static code snapshot the way the
other bug classes context_builder.py serves are - it's a property of
*execution history*. NBFix already tracks the piece that matters
(IntermediateRepresentations.last_ran_code vs. .cell_code - the code a
cell was actually last executed with, vs. what's in it now, updated by
events.RunCellEvent/ChangeCellCodeEvent as a real session progresses),
but nothing before this fed it to the LLM - context_builder.py's
build_dependency_edges and friends only ever look at .cell_code.

original_code is an explicit parameter here, not read from
notebook_IR[cell_index].last_ran_code internally - found via a live
end-to-end check (not assumed) that reading it internally is actually
wrong for the real intended call sequence: RunCellEvent overwrites
last_ran_code to match cell_code as part of the *same* call that
actually executes the cell, so by the time a cell's kernel value is
genuinely fresh (i.e. it's actually safe to reason about "cell 0 was
just re-executed"), last_ran_code has already been reset and the diff is
gone. The caller must capture the prior code *before* triggering the
real re-run and pass it through explicitly - see
DetectStaleCellsEvent's docstring for the exact sequence.
"""
from dataclasses import dataclass


@dataclass
class StaleDetectionContext:
    edited_cell: int
    original_code: str
    current_code: str
    cells: dict[int, str]


def build_stale_context(notebook_IR, cell_index: int, original_code: str) -> StaleDetectionContext:
    """
    edited_cell is the cell that was just (re-)executed. current_code is
    read live from notebook_IR (it's safe to read at call time - nothing
    resets .cell_code the way .last_ran_code gets reset). original_code
    must be supplied by the caller, captured before the real re-run
    happened - see this module's docstring for why.

    cells captures every cell's *current* code - the only state relevant
    downstream of the edited cell, since every other cell's own
    last_ran_code, by construction, was whatever code it's currently
    showing (nothing else has changed since).
    """
    ir = notebook_IR[cell_index]
    return StaleDetectionContext(
        edited_cell=cell_index,
        original_code=original_code,
        current_code=ir.cell_code,
        cells={cid: cell_ir.cell_code for cid, cell_ir in notebook_IR.items()},
    )
