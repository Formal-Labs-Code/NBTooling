from __future__ import annotations

import logging

from nbcore.analyses.runner.analysis_results import ErrorInfo, PathResult, Result

logger = logging.getLogger(__name__)

# No natural "line" for a staleness finding the way a real code bug has
# one - the whole cell is stale, not one line of it. Fixed placeholder,
# same rationale as type_shape_analysis.py's own findings (which only
# need cell_id downstream, never a real source location).
_LINE = 1

# stale_prompts.STALE_SYSTEM_PROMPT now asks for a bare cell_id list, not
# a {cell_id, message} object per finding - dropping the free-text
# explanation was measured to cut latency ~3x (see that module's
# docstring). There's no per-cell explanation to surface anymore, so
# every finding gets this fixed message instead.
_MESSAGE = "LLM stale-cell check: this cell depends on a value that has not been refreshed since an upstream edit."


def map_stale_findings_to_result(findings_json: dict, notebook_IR) -> Result:
    """
    Builds a Result from parsed LLM JSON output (see
    stale_prompts.STALE_SYSTEM_PROMPT for the expected shape). A finding
    that fails validation is dropped, with a logged warning, rather than
    raising - same defensive-parsing discipline as result_mapping.py's
    map_findings_to_result, for the same reason: a single hallucinated
    finding must not break the rest of the result.
    """
    result = Result()

    findings = findings_json.get("stale_cells") if isinstance(findings_json, dict) else None
    if not isinstance(findings, list):
        logger.warning("LLM stale-cell response missing a 'stale_cells' list: %r", findings_json)
        return result

    for finding in findings:
        error_info = _build_error_info(finding, notebook_IR)
        if error_info is None:
            continue
        result.add_path_result(
            PathResult(path=[error_info.cell_id], error_infos=[error_info])
        )

    return result


def _build_error_info(cell_id, notebook_IR) -> ErrorInfo | None:
    if not isinstance(cell_id, int) or isinstance(cell_id, bool) or cell_id not in notebook_IR:
        logger.warning("Skipping stale finding with invalid/unknown cell_id: %r", cell_id)
        return None

    return ErrorInfo(
        cell_id=cell_id,
        line=_LINE,
        label="",
        error_type="LLM_STALE",
        error_message=_MESSAGE,
    )
