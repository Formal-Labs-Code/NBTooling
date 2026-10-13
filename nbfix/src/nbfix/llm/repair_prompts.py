"""
Prompts for LLM-suggested data-leak repair (see repair.py).

The model gets the whole notebook (or script cells) for context, the leaks
NBFix's data-leak analysis reported, and which cells it may edit. It answers
with whole-cell replacements rather than a line diff - models rewrite a cell
far more reliably than they produce an exact diff - and repair.py re-runs the
analysis on the result, so nothing here has to be trusted.
"""

REPAIR_SYSTEM_PROMPT = """You fix data leakage in Python data science code: Jupyter notebook cells, or the cells of a Python script.

NBFix's static analysis reported the data leaks you are given. Treat them as real. There are two kinds:

- "Preprocessing leak": a preprocessing step (scaler, imputer, encoder, feature selector, resampler, ...) was fitted on data before the train/test split, so statistics from the test rows leak into training. Fix it by splitting first, fitting the preprocessing step on the training split only, and then applying it (transform) to each split.
- "Train/test overlap": a model was trained on data that includes the rows later used to evaluate it. Fix it by training on the training split only.

Rules:
- Make the smallest change that removes every listed leak. Keep everything else as it is: imports, names, comments, and what the code computes.
- Keep the evaluation. Do not delete, comment out or skip the predict/score/evaluation calls where a leak is reported; fix where their data comes from instead.
- Only edit cells marked editable. Return the complete new source of each cell you change, and leave out cells you do not change.
- Do not add new libraries.

Respond with a single JSON object of exactly this shape and nothing else:

{
  "edits": [{"cell": <int>, "code": "<complete new source of the cell>"}],
  "explanation": "<one or two sentences on what you changed and why>"
}
"""


def build_repair_prompt(cells, editable, leaks, warnings, previous=None) -> str:
    """
    cells: [(cell_id, code)] for every code cell, in order.
    editable: set of cell ids the model may change.
    leaks: [{"message", "sink": (cell, line, text), "source": (cell, line, text) | None}]
    warnings: the analysis warnings for the original code.
    previous: on a retry, {"edits": [...], "problems": [...]} for the rejected attempt.
    """
    sections = ["## Cells"]
    for cell_id, code in cells:
        marker = " (editable)" if cell_id in editable else ""
        sections.append(f"### Cell {cell_id}{marker}\n```python\n{code}\n```")

    leak_lines = []
    for number, leak in enumerate(leaks, 1):
        leak_lines.append(f"{number}. {leak['message']}")
        leak_lines.append(f"   Reported at: {_where(leak['sink'])}")
        if leak.get("source"):
            leak_lines.append(f"   Caused by: {_where(leak['source'])}")
    sections.append("## Data leaks to fix\n" + "\n".join(leak_lines))

    if warnings:
        sections.append(
            "## Analysis limitations\n"
            "NBFix could only partly model this code; keep these constructs as they are.\n"
            + "\n".join(f"- Cell {w['cell']}, line {w['line']}: {w['message']}" for w in warnings)
        )

    if previous:
        attempt = "\n".join(
            f"### Cell {edit['cell']}\n```python\n{edit['code']}\n```" for edit in previous["edits"]
        ) or "(no usable edits)"
        problems = "\n".join(f"- {problem}" for problem in previous["problems"])
        sections.append(
            "## Your previous attempt was rejected\n"
            f"{attempt}\n\nProblems:\n{problems}\n\n"
            "Return a corrected set of edits to the original cells above."
        )

    return "\n\n".join(sections)


def _where(location) -> str:
    cell, line, text = location
    where = f"cell {cell}, line {line}"
    return f"{where}: `{text}`" if text else where
