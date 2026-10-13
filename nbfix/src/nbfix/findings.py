"""Data-leak findings in the schema MCP `analyze` returns, shared by the
MCP server and repair.py - kept out of mcp_server.py so it can be used
without the optional `mcp` extra installed."""
from __future__ import annotations

import json
from typing import Any

from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.resource_utils.script_loading import first_code_cell, has_cell_markers, is_script

from .cli import analysis_warnings, run_batch


def _location(role: str, cell, line, file_line=None) -> dict[str, Any]:
    location = {"role": role, "cell": cell, "line": line}
    if file_line is not None:
        location["file_line"] = file_line
    return location


def normalize_findings(raw_result: str, single_cell_script: bool = False) -> list[dict[str, Any]]:
    if not raw_result:
        return []

    try:
        payload = json.loads(raw_result)
    except json.JSONDecodeError:
        return []

    if not isinstance(payload, list):
        return []

    findings: list[dict[str, Any]] = []
    for entry in payload:
        if not isinstance(entry, dict):
            continue

        cell_id = entry.get("cell_id")
        # A script without cell markers is one cell: a cell path says nothing.
        path = None if single_cell_script else (entry.get("path") or None)
        for err in entry.get("errors") or []:
            if not isinstance(err, dict):
                continue

            source = err.get("source")
            locations = [_location("sink", cell_id, err.get("line"), err.get("file_line"))]
            if source:
                locations.append(_location("source", source.get("cell"), source.get("line"), source.get("file_line")))

            findings.append(
                {
                    "leakage_type": DATA_LEAK,
                    "sink": err.get("label"),
                    "source": source,
                    "path": path,
                    "locations": locations,
                    "message": err.get("message", ""),
                }
            )

    return findings


def analyze_data_leaks(payload: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Run the NBFix data-leak analysis on a notebook dict and normalize it to
    {"findings": [...], "warnings": [...]}."""
    raw_result, session = run_batch(
        filename=None,
        notebook=payload,
        analyses=[DATA_LEAK],
        start=first_code_cell(payload),
        level=10,
    )
    return {
        "findings": normalize_findings(
            raw_result, single_cell_script=is_script(payload) and not has_cell_markers(payload)),
        "warnings": analysis_warnings(session, payload),
    }
