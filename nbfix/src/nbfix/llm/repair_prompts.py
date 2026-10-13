"""
Prompts for LLM-suggested data-leak repair (see repair.py).

The model gets the whole notebook (or script cells) for context, the leaks
NBFix's data-leak analysis reported, and which cells it may edit. It answers
with whole-cell replacements rather than a line diff - models rewrite a cell
far more reliably than they produce an exact diff - and repair.py re-runs the
analysis on the result, so nothing here has to be trusted.
"""

REPAIR_SYSTEM_PROMPT = """You fix data leakage in Python data science code: Jupyter notebook cells, or the cells of a Python script.

NBFix's static analysis found the data leaks listed in the request. They are real. NBFix re-analyzes your fix and accepts it only when every listed leak is gone.

## The two kinds of leak

- "Preprocessing leak": a preprocessing step (scaler, imputer, encoder, feature selector, resampler, ...) was fitted on data before it was split into training and test data, so statistics from the test rows leak into training. Fix it by splitting first, fitting the preprocessing step on the training split only, and then transforming each split with that fitted step.
- "Train/test overlap": a model was trained on data that includes the rows later used to evaluate it. Fix it by training on the training split only.

A notebook can split its data more than once - for example, one split per model. Fix each leak against the split that produces the data the model is evaluated on; each leak in the request names that split.

## Cells

Cells run top to bottom, and a cell can only use names defined in itself or in an earlier cell. Fixing a preprocessing leak often means restructuring: when the split comes after the preprocessing step, move the split up, or move the preprocessing down into the split's cell (after the split) or a later cell. Rewriting the cells involved like this is expected.

## Rules

1. Remove every listed leak, and otherwise keep the code doing what it did:
   - Keep the evaluation. Do not delete, comment out or skip the predict/score calls where a leak is reported; fix where their data comes from instead.
   - Keep every preprocessing step. Do not remove it, stub it out, or replace or redefine a library class or function.
   - Fit each preprocessing step once, on the training split only, and reuse that fitted object to transform the test split. Never fit anything on test data.
   - Whatever preprocessing the training data gets before training, the data passed to the evaluation calls must get too.
   - Keep features and their labels from the same split together.
2. Beyond that, change as little as possible: keep imports, names and comments, and copy every line you don't need to change exactly, including blank lines and spacing. Update a comment if it describes the old, leaky code.
3. Only edit cells marked editable, and do not add new libraries.

## Answer

Return the complete new source of each cell you change, and leave out cells you don't change. Respond with a single JSON object of exactly this shape and nothing else:

{
  "edits": [{"cell": <int>, "code": "<complete new source of the cell>"}],
  "explanation": "<one or two sentences on what you changed and why>"
}
"""


def build_repair_prompt(cells, editable, leaks, warnings, previous=None) -> str:
    """
    cells: [(cell_id, code)] for every code cell, in order.
    editable: set of cell ids the model may change.
    leaks: [{"message", "sink": (cell, line, text), "source": (cell, line, text) | None,
             "split": (cell, text) | None}]
    warnings: the analysis warnings for the original code.
    previous: on a retry, {"attempt": n, "edits": [...], "problems": [...],
              "repeats": earlier attempt number | None} for the rejected attempt n.
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
        if leak.get("split"):
            cell, text = leak["split"]
            leak_lines.append(f"   The evaluated data comes from the split in cell {cell}: `{text}`")
    sections.append("## Data leaks to fix\n" + "\n".join(leak_lines))

    if warnings:
        sections.append(
            "## Analysis limitations\n"
            "NBFix could only partly model this code; keep these constructs as they are.\n"
            + "\n".join(f"- Cell {w['cell']}, line {w['line']}: {w['message']}" for w in warnings)
        )

    if previous:
        attempt = "\n".join(
            f"#### Cell {edit['cell']}\n```python\n{edit['code']}\n```" for edit in previous["edits"]
        ) or "(no usable edits)"
        problems = "\n".join(f"- {problem}" for problem in previous["problems"])
        repeated = (
            f"\n\nThis was the same fix as your attempt {previous['repeats']}, which was rejected too. "
            "Repeating it will not work - take a different approach."
            if previous.get("repeats") else ""
        )
        sections.append(
            f"## Your attempt {previous['attempt']} was rejected\n"
            "NBFix re-analyzed it and found the problems below. Its analysis is exact, so a fix it rejected "
            "will be rejected again: change the approach, not just the wording.\n\n"
            f"### Your edits\n{attempt}\n\n### Problems\n{problems}{repeated}\n\n"
            "Return a corrected set of edits to the original cells at the top, not to these edits."
        )

    return "\n\n".join(sections)


def _where(location) -> str:
    cell, line, text = location
    where = f"cell {cell}, line {line}"
    return f"{where}: `{text}`" if text else where
