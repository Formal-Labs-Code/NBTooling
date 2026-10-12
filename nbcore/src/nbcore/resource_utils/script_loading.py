"""
Python scripts as notebook-shaped input.

A script with cell markers is split into one cell per marker, so it's
analyzed exactly like the notebook it represents:
  * `# %%` (VS Code / Jupytext percent format); `# %% [markdown]` or
    `# %% [md]` starts a markdown cell, skipped like any notebook markdown;
  * `# In[ ]:` / `# In[3]:` (Jupyter's own .py export).
A script without markers is a single code cell: it runs top to bottom, which
is how the analyses already treat code within one cell.

Each cell's metadata records the file line its (stored) source starts on, so
cell-relative line numbers can be mapped back into the file - see
file_line(). Leading blank lines are dropped from each cell up front because
IntermediateRepresentations.remove_magic drops them too, and line numbers
count from the first line it keeps.
"""

import re

START_LINE_KEY = "nbtooling_start_line"
SCRIPT_KEY = "nbtooling_script"

_PERCENT_MARKER = re.compile(r"^#\s*%%(?P<tag>.*)$")
_IN_MARKER = re.compile(r"^#\s*In\[[^\]]*\]:\s*$")
_MARKDOWN_TAG = re.compile(r"^\s*\[(markdown|md)\]", re.IGNORECASE)


def _is_code(text: str) -> bool:
    """True if the text has anything besides blank lines and comments - a
    shebang/encoding preamble before the first marker isn't a cell."""
    return any(line.strip() and not line.lstrip().startswith("#") for line in text.splitlines())


def script_to_notebook(source: str) -> dict:
    """A notebook dict ({"cells": [...], "metadata": {...}}) for a script."""
    cells = []
    has_markers = False
    lines = source.splitlines()

    def flush(body: list[str], start: int, cell_type: str):
        while body and body[0] == "":
            body = body[1:]
            start += 1
        text = "\n".join(body)
        if cell_type == "markdown" or _is_code(text):
            cells.append({
                "cell_type": cell_type,
                "source": text,
                "metadata": {START_LINE_KEY: start},
            })

    body, start, cell_type = [], 1, "code"
    for number, line in enumerate(lines, 1):
        percent = _PERCENT_MARKER.match(line)
        if percent or _IN_MARKER.match(line):
            has_markers = True
            flush(body, start, cell_type)
            body, start = [], number + 1
            cell_type = "markdown" if percent and _MARKDOWN_TAG.match(percent.group("tag")) else "code"
            continue
        body.append(line)
    flush(body, start, cell_type)

    return {"cells": cells, "metadata": {SCRIPT_KEY: {"cell_markers": has_markers}}}


def is_script(notebook: dict) -> bool:
    return SCRIPT_KEY in (notebook.get("metadata") or {})


def has_cell_markers(notebook: dict) -> bool:
    return bool((notebook.get("metadata") or {}).get(SCRIPT_KEY, {}).get("cell_markers"))


def cell_start_lines(notebook: dict) -> dict[int, int]:
    """Cell index (as load_notebook numbers them) -> file line of its first
    source line, for cells that carry one."""
    return {
        index: cell["metadata"][START_LINE_KEY]
        for index, cell in enumerate(notebook.get("cells", []))
        if START_LINE_KEY in (cell.get("metadata") or {})
    }


def file_line(start_lines: dict[int, int], cell_id, line):
    """File line for a cell-relative line, or None if it can't be mapped."""
    if cell_id not in start_lines or not isinstance(line, int) or line < 1:
        return None
    return start_lines[cell_id] + line - 1


def first_code_cell(notebook: dict) -> int:
    """Index of the first code cell with source (load_notebook's numbering),
    or 0 if there is none."""
    for index, cell in enumerate(notebook.get("cells", [])):
        if cell.get("cell_type") == "code" and cell.get("source"):
            return index
    return 0
