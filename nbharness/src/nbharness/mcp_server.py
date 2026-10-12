from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.resource_utils.utils import read_json

from nbcore.resource_utils.script_loading import first_code_cell, has_cell_markers, is_script, script_to_notebook

from .cli import nbharness

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
    raise RuntimeError(
        "NBHarness MCP support requires the optional 'mcp' extra. "
        "Install it with: pip install -e './nbharness[mcp]'"
    ) from exc

mcp = FastMCP("NBHarness")


def _coerce_notebook(notebook: Any) -> dict[str, Any]:
    if notebook is None:
        raise ValueError("Notebook input is required.")

    if isinstance(notebook, (str, Path)):
        raw = str(notebook).strip()
        if not raw:
            raise ValueError("Notebook path or JSON string cannot be empty.")
        if raw.startswith("{") or raw.startswith("["):
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError("Notebook JSON string is not valid JSON.") from exc
            if not isinstance(payload, dict):
                raise TypeError("Notebook JSON payload must decode to a dict/object.")
            return _coerce_notebook(payload)

        if "\n" in raw:
            # Not JSON, and no path contains a newline: Python script source.
            return script_to_notebook(str(notebook))

        candidate = Path(raw)
        if not candidate.exists():
            raise FileNotFoundError(f"Notebook path does not exist: {raw}")
        if candidate.suffix == ".py":
            return script_to_notebook(candidate.read_text(encoding="utf-8"))
        return read_json(str(candidate))

    if hasattr(notebook, "read"):
        return json.load(notebook)

    if isinstance(notebook, dict):
        if isinstance(notebook.get("script"), str) and "cells" not in notebook:
            return script_to_notebook(notebook["script"])
        return notebook

    raise TypeError(
        "Notebook input must be a dict, a JSON string, a file path, or a file-like object."
    )


def _location(role: str, cell, line, file_line=None) -> dict[str, Any]:
    location = {"role": role, "cell": cell, "line": line}
    if file_line is not None:
        location["file_line"] = file_line
    return location


def _normalize_findings(raw_result: str, single_cell_script: bool = False) -> list[dict[str, Any]]:
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


def _execute_notebook_for_verification(notebook: Any) -> bool:
    try:
        import nbformat
        from nbclient import NotebookClient
    except ImportError:
        return False

    payload = _coerce_notebook(notebook)
    nb = nbformat.from_dict(payload)
    client = NotebookClient(nb, timeout=60, allow_errors=False)
    client.execute()
    return True


def _analyze_impl(notebook: Any) -> list[dict[str, Any]]:
    """Run the existing NBHarness data-leak analysis and normalize it to the PoC schema."""
    payload = _coerce_notebook(notebook)
    raw_result = nbharness(
        filename=None,
        notebook=payload,
        analyses=[DATA_LEAK],
        start=first_code_cell(payload),
        level=10,
    )
    return _normalize_findings(raw_result, single_cell_script=is_script(payload) and not has_cell_markers(payload))


def _verify_impl(notebook: Any) -> dict[str, Any]:
    """Run the analysis again and optionally execute the notebook end-to-end."""
    payload = _coerce_notebook(notebook)
    findings = _analyze_impl(payload)
    notes: list[str] = []
    runs: bool | None = None

    try:
        runs = _execute_notebook_for_verification(payload)
        if runs:
            notes.append("Notebook executed successfully end-to-end.")
        else:
            notes.append("Notebook execution was attempted but unavailable in this environment.")
    except Exception as exc:  # pragma: no cover - environment-specific
        runs = False
        notes.append(f"Notebook execution failed: {exc}")

    if findings:
        notes.append("NBHarness analysis reported leakage.")
    else:
        notes.append("NBHarness analysis reported no leakage.")

    return {
        "leakage_clean": not findings,
        "runs": runs,
        "findings": findings,
        "notes": " ".join(notes),
    }


@mcp.tool()
def analyze(notebook: str | dict[str, Any]) -> list[dict[str, Any]]:
    """Analyze a notebook or Python script and return leakage findings in the static-analysis schema.

    `notebook` is a .ipynb or .py path, notebook JSON (string or object),
    Python script source, or {"script": "<source>"}. A script is split into
    cells at `# %%` / `# In[ ]:` markers, or is a single cell without them;
    script locations also carry `file_line`, the line in the script."""
    return _analyze_impl(notebook)


def verify(notebook: str | dict[str, Any]) -> dict[str, Any]:
    """Optional best-effort runtime sanity check. This is not the main NBHarness contract."""
    return _verify_impl(notebook)


if __name__ == "__main__":
    mcp.run(transport="stdio")
