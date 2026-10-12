from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.resource_utils.utils import read_json

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
            return payload

        candidate = Path(raw)
        if not candidate.exists():
            raise FileNotFoundError(f"Notebook path does not exist: {raw}")
        return read_json(str(candidate))

    if hasattr(notebook, "read"):
        return json.load(notebook)

    if isinstance(notebook, dict):
        return notebook

    raise TypeError(
        "Notebook input must be a dict, a JSON string, a file path, or a file-like object."
    )


def _normalize_findings(raw_result: str) -> list[dict[str, Any]]:
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
        path = entry.get("path") or None
        for err in entry.get("errors") or []:
            if not isinstance(err, dict):
                continue

            source = err.get("source")
            locations = [{"role": "sink", "cell": cell_id, "line": err.get("line")}]
            if source:
                locations.append({"role": "source", "cell": source.get("cell"), "line": source.get("line")})

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
        start=0,
        level=10,
    )
    return _normalize_findings(raw_result)


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
    """Analyze a notebook and return leakage findings in the static-analysis schema."""
    return _analyze_impl(notebook)


def verify(notebook: str | dict[str, Any]) -> dict[str, Any]:
    """Optional best-effort runtime sanity check. This is not the main NBHarness contract."""
    return _verify_impl(notebook)


if __name__ == "__main__":
    mcp.run(transport="stdio")
