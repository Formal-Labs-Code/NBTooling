from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nbcore.resource_utils.utils import read_json

from nbcore.resource_utils.script_loading import script_to_notebook

from .findings import analyze_data_leaks

try:
    from mcp.server.fastmcp import FastMCP
except ImportError as exc:  # pragma: no cover - exercised only when the extra is absent
    raise RuntimeError(
        "NBFix MCP support requires the optional 'mcp' extra. "
        "Install it with: pip install -e './nbfix[mcp]'"
    ) from exc

mcp = FastMCP("NBFix")


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


def _analyze_impl(notebook: Any) -> dict[str, list[dict[str, Any]]]:
    """Run the existing NBFix data-leak analysis and normalize it to the PoC schema:
    {"findings": [...], "warnings": [...]}."""
    return analyze_data_leaks(_coerce_notebook(notebook))


def _verify_impl(notebook: Any) -> dict[str, Any]:
    """Run the analysis again and optionally execute the notebook end-to-end."""
    payload = _coerce_notebook(notebook)
    result = _analyze_impl(payload)
    findings = result["findings"]
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
        notes.append("NBFix analysis reported leakage.")
    else:
        notes.append("NBFix analysis reported no leakage.")

    return {
        "leakage_clean": not findings,
        "runs": runs,
        "findings": findings,
        "warnings": result["warnings"],
        "notes": " ".join(notes),
    }


@mcp.tool()
def analyze(notebook: str | dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Analyze a notebook or Python script and return leakage findings in the static-analysis schema.

    `notebook` is a .ipynb or .py path, notebook JSON (string or object),
    Python script source, or {"script": "<source>"}. A script is split into
    cells at `# %%` / `# In[ ]:` markers, or is a single cell without them;
    script locations also carry `file_line`, the line in the script.

    Returns {"findings": [...], "warnings": [...]}. Warnings list code the
    analysis could only partly model - an unsupported statement analyzed as
    a no-op, or a local call it couldn't inline analyzed as an opaque call
    - so "no findings" for that code is not a guarantee of no leak."""
    return _analyze_impl(notebook)


def verify(notebook: str | dict[str, Any]) -> dict[str, Any]:
    """Optional best-effort runtime sanity check. This is not the main NBFix contract."""
    return _verify_impl(notebook)


if __name__ == "__main__":
    mcp.run(transport="stdio")
