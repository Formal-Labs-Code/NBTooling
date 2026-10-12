import logging

from ..ir.intermediate_representations import IntermediateRepresentations

logger = logging.getLogger(__name__)


def load_notebook_resilient(cells_json) -> dict[int, IntermediateRepresentations]:
    """
    Like resource_utils.utils.load_notebook, but a cell that fails to load
    at all - e.g. a syntax error, or a construct missing from the grammar -
    is skipped rather than aborting the whole notebook.

    Constructs the grammar parses but the IR doesn't support yet (lambda,
    walrus, del, ...; see ast_transformer.py's docstring) no longer fail a
    cell anywhere: the statement using one becomes a no-op placeholder and
    the cell's CFG records a warning (IntermediateRepresentations.CFG.warnings),
    so partial results are visible rather than silent.
    """
    notebook_IR: dict[int, IntermediateRepresentations] = {}
    if not cells_json:
        return notebook_IR

    for cell_id, cell in enumerate(cells_json):
        if cell["cell_type"] != "code" or not cell["source"]:
            continue

        source = cell["source"]
        if isinstance(source, list):
            source = "".join(source)

        try:
            notebook_IR[cell_id] = IntermediateRepresentations(source, cell_id)
        except Exception as exc:
            logger.warning("Skipping cell %d - failed to parse: %s", cell_id, exc)

    return notebook_IR
