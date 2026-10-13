from __future__ import annotations

import logging

from nbcore.analyses.runner.analysis_results import ErrorInfo, PathResult, Result

logger = logging.getLogger(__name__)

# Same rationale as stale_result_mapping.py's _LINE - a call-sequence
# finding is about a cell's relationship to another cell, not one
# specific line within it.
_LINE = 1


def map_api_sequence_findings_to_result(findings_json: dict, notebook_IR) -> Result:
    """
    Builds a Result from the LLM's {"findings": [{"cell_id": int,
    "message": str}]} response (see
    api_sequence_prompts.API_SEQUENCE_SYSTEM_PROMPT). Same defensive-
    parsing discipline as stale_result_mapping.py: a finding that fails
    validation is dropped, with a logged warning, rather than raising -
    a single hallucinated/malformed finding must not break the rest of
    the result.
    """
    result = Result()

    findings = findings_json.get("findings") if isinstance(findings_json, dict) else None
    if not isinstance(findings, list):
        logger.warning("LLM api-sequence response missing a 'findings' list: %r", findings_json)
        return result

    for finding in findings:
        error_info = _build_error_info(finding, notebook_IR)
        if error_info is None:
            continue
        result.add_path_result(
            PathResult(path=[error_info.cell_id], error_infos=[error_info])
        )

    return result


def _build_error_info(finding, notebook_IR) -> ErrorInfo | None:
    if not isinstance(finding, dict):
        logger.warning("Skipping non-dict api-sequence finding: %r", finding)
        return None

    cell_id = finding.get("cell_id")
    if not isinstance(cell_id, int) or isinstance(cell_id, bool) or cell_id not in notebook_IR:
        logger.warning("Skipping api-sequence finding with invalid/unknown cell_id: %r", finding)
        return None

    message = finding.get("message")
    if not isinstance(message, str) or not message:
        logger.warning("Skipping api-sequence finding with missing message: %r", finding)
        return None

    return ErrorInfo(
        cell_id=cell_id,
        line=_LINE,
        label="",
        error_type="LLM_API_SEQUENCE",
        error_message=message,
    )
