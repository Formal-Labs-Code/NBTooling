from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.events import RunBatchEvent

from ..analyses.idle_cell_analysis import IDLE
from ..analyses.isolated_cell_analysis import ISOLATED
from ..analyses.stale_cell_analysis import STALE
from ..events import (
    AddActiveAnalysesEvent,
    AddCellEvent,
    ChangeCellCodeEvent,
    CloseNotebookEvent,
    OpenNotebookEvent,
    RemoveCellEvent,
    RunCellEvent,
)

_VALID_FINDING_TYPES = {DATA_LEAK, STALE, IDLE, ISOLATED}


class InvalidEventError(ValueError):
    pass


def _build_detect_bugs_event(params):
    # Lazy import: nbcore.llm.detect_bugs_event (via config.py -> client.py)
    # imports openai, which the base/jupyter packages must never depend on
    # at module load time. Only import it once this specific event is
    # actually requested, and turn "not installed" into the same
    # InvalidEventError every other bad-request case already produces.
    # Shared with the --detect-bugs CLI flag - general bug detection
    # is generic over any dict of cells, notebook or script alike, so it
    # lives in nbcore rather than being duplicated per tool.
    try:
        from nbcore.llm.detect_bugs_event import DetectBugsEvent
    except ImportError as exc:
        raise InvalidEventError(
            "LLM bug detection is not installed - pip install nbcore[llm]"
        ) from exc

    scope = params.get("scope", "full")
    if scope not in ("cell", "subgraph", "full"):
        raise InvalidEventError(f"Invalid scope: {scope!r}")

    cell_index = params.get("cell_index")
    if scope != "full" and cell_index is None:
        raise InvalidEventError(f"cell_index is required for scope={scope!r}")

    context_mode = params.get("context_mode", "deps")
    if context_mode not in ("none", "deps"):
        raise InvalidEventError(f"Invalid context_mode: {context_mode!r}")

    finding_types = params.get("finding_types")
    if finding_types is not None:
        invalid = set(finding_types) - _VALID_FINDING_TYPES
        if invalid:
            raise InvalidEventError(f"Invalid finding_types: {sorted(invalid)}")

    return DetectBugsEvent(scope, cell_index, context_mode=context_mode, finding_types=finding_types)


def _build_detect_stale_cells_llm_event(params):
    # Same lazy-import guard as _build_detect_bugs_event, same reason.
    try:
        from ..llm.detect_stale_cells_event import DetectStaleCellsEvent
    except ImportError as exc:
        raise InvalidEventError(
            "LLM stale-cell detection is not installed - pip install nbfix[llm]"
        ) from exc

    cell_index = params.get("cell_index")
    if cell_index is None:
        raise InvalidEventError("cell_index is required for detect_stale_cells_llm")

    # original_code is required, not read from notebook_IR server-side -
    # see DetectStaleCellsEvent's own docstring for why: last_ran_code
    # gets overwritten by the same run_cell call that makes the cell's
    # kernel value fresh, so there's no single moment where reading it
    # here would be correct. The frontend tracks it itself (see
    # manager.ts's _lastConfirmedCode), the same way it already owns
    # cell identity/position for every other event.
    original_code = params.get("original_code")
    if original_code is None:
        raise InvalidEventError("original_code is required for detect_stale_cells_llm")

    return DetectStaleCellsEvent(int(cell_index), str(original_code))


def _build_detect_api_sequence_llm_event(params):
    # Same lazy-import guard as _build_detect_bugs_event, same reason.
    try:
        from ..llm.detect_api_sequence_event import DetectApiSequenceEvent
    except ImportError as exc:
        raise InvalidEventError(
            "LLM API-sequence detection is not installed - pip install nbfix[llm]"
        ) from exc

    # focus_cell is optional - omitted (or explicit null) scans the
    # whole notebook (used when the toggle is switched on); given, it
    # narrows the prompt to that cell's dependency-connected component
    # (used on edit - see DetectApiSequenceEvent's docstring).
    focus_cell = params.get("focus_cell")
    return DetectApiSequenceEvent(None if focus_cell is None else int(focus_cell))


_EVENT_BUILDERS = {
    "open_notebook": lambda params: OpenNotebookEvent(params["notebook_json"]),
    "run_cell": lambda params: RunCellEvent(params["cell_index"]),
    "run_batch": lambda params: RunBatchEvent(params["start_cell"]),
    "add_active_analyses": lambda params: AddActiveAnalysesEvent(params["active_analyses"]),
    "add_cell": lambda params: AddCellEvent(params["position"], params["kind"], params["content"]),
    "remove_cell": lambda params: RemoveCellEvent(params["position"]),
    "change_cell": lambda params: ChangeCellCodeEvent(
        str(params["new_code"]), int(params["cell_index"]), bool(params["with_result"])
    ),
    "close_notebook": lambda params: CloseNotebookEvent(),
    "detect_bugs": _build_detect_bugs_event,
    "detect_stale_cells_llm": _build_detect_stale_cells_llm_event,
    "detect_api_sequence_llm": _build_detect_api_sequence_llm_event,
}


def build_event(event_name: str, params: dict):
    """
    Constructs an Event instance for the given event name and params.

    Raises InvalidEventError if the event name is unknown or the params
    don't match what that event requires.
    """
    try:
        builder = _EVENT_BUILDERS[event_name]
    except KeyError:
        raise InvalidEventError(f"Unknown event: {event_name!r}")

    try:
        return builder(params or {})
    except (KeyError, TypeError, ValueError) as exc:
        raise InvalidEventError(f"Invalid parameters for event {event_name!r}: {exc}") from exc
