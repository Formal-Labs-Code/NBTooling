"""
LLM-suggested repair of data leaks, verified by NBFix's own analysis.

fix() asks an LLM for whole-cell replacements that remove every data leak the
analysis reports, then re-runs the analysis on a patched copy. A patch is
"verified" only if, on the patched code:
  * the analysis reports no data leaks at all;
  * it raised no new analysis warnings - otherwise a leak could "disappear"
    just because the patch uses code the analysis can't model;
  * every call a leak was reported at (lr.predict, lr.score, ...) is still
    there - deleting the evaluation also makes the leak go away;
  * every step a leak was traced to (scaler.fit, smote.fit_resample, ...)
    is still called, and its result is still used - stubbing out the
    preprocessing, or computing it and ignoring it, does too;
  * it reads no name before an earlier cell defines it, fits nothing on test
    data, and never pairs features and labels from different splits (see
    code_checks).
A rejected attempt is retried with the reasons as feedback.

Nothing is ever written back: the result is a proposal (edits plus a unified
diff) for the user, or the agent calling the MCP tool, to apply.
"""
from __future__ import annotations

import copy
import difflib
import re
from collections import Counter
from typing import Any

from nbcore.resource_utils.script_loading import cell_start_lines, is_script

from .code_checks import (
    first_binding, fits_on_test_data, misaligned_splits, parse_cell, split_origin, undefined_reads,
    unused_results,
)
from .findings import analyze_data_leaks
from .llm.repair_prompts import REPAIR_SYSTEM_PROMPT, build_repair_prompt

DEFAULT_MAX_ATTEMPTS = 3

# Lines IntermediateRepresentations.remove_magic drops before numbering lines.
_MAGIC = re.compile(r"^[!%]")
# The sink call in a data-leak message: "`X_test` passed to `lr.score` ...".
_SINK_CALL = re.compile(r"passed to `([^`]+)`")
# Methods a repair may swap for each other at a leak's source: fit_transform
# on all the data often becomes fit on the training split plus transform.
_FIT_METHODS = {"fit", "fit_transform"}
_FENCE = re.compile(r"^```[a-zA-Z]*\n(?P<code>.*?)\n?```\s*$", re.DOTALL)


def fix(notebook: dict, *, client=None, max_attempts: int = DEFAULT_MAX_ATTEMPTS,
        source: str | None = None, filename: str | None = None) -> dict[str, Any]:
    """Suggests a verified repair for every data leak in a notebook dict (as
    read_input/script_to_notebook produce). For a script, pass its `source`
    text too, so the diff is a unified diff of the file itself.

    Returns {"status", "attempts", "findings", "warnings", ...}, status being
    one of:
      "no_findings" - nothing to fix;
      "verified"    - adds "edits", "diff", "explanation";
      "unverified"  - the last patch that could be analyzed but failed a check:
                      adds "edits", "diff", "explanation", "problems",
                      "remaining_findings";
      "failed"      - no attempt produced analyzable code: adds "problems".
    Raises RuntimeError if the LLM can't be reached or isn't installed."""
    llm_client_error, default_client = _llm()
    original = analyze_data_leaks(copy.deepcopy(notebook))
    result = {
        "status": "no_findings",
        "attempts": 0,
        "findings": original["findings"],
        "warnings": original["warnings"],
    }
    if not original["findings"]:
        return result

    client = client or default_client()
    cells = _code_cells(notebook)
    editable = _editable_cells(original["findings"], cells)
    leaks = [_describe(finding, cells) for finding in original["findings"]]

    previous, last_analyzed, tried = None, None, []
    for attempt in range(1, max_attempts + 1):
        prompt = build_repair_prompt(list(cells.items()), editable, leaks, original["warnings"], previous)
        try:
            response = client.chat_json(REPAIR_SYSTEM_PROMPT, prompt)
        except llm_client_error as exc:
            raise RuntimeError(str(exc)) from exc

        outcome = _check(response, notebook, cells, editable, original)
        result["attempts"] = attempt
        if not outcome["problems"]:
            return {**result, "status": "verified", **_proposal(outcome, notebook, cells, source, filename)}
        if "remaining_findings" in outcome:
            last_analyzed = outcome
        repeats = next((n for n, edits in enumerate(tried, 1) if edits and edits == outcome["edits"]), None)
        tried.append(outcome["edits"])
        previous = {"attempt": attempt, "edits": outcome["edits"], "problems": outcome["problems"], "repeats": repeats}

    if last_analyzed is not None:
        return {
            **result,
            "status": "unverified",
            **_proposal(last_analyzed, notebook, cells, source, filename),
            "problems": last_analyzed["problems"],
            "remaining_findings": last_analyzed["remaining_findings"],
        }
    return {**result, "status": "failed", "problems": previous["problems"]}


def _llm():
    try:
        from nbcore.llm.client import LLMClientError
        from nbcore.llm.config import default_client
    except ImportError as exc:
        raise RuntimeError("LLM repair is not installed - pip install nbfix[llm]") from exc
    return LLMClientError, default_client


def _check(response, notebook, cells, editable, original) -> dict[str, Any]:
    """Validates one LLM response and, if it is usable, analyzes the patched
    notebook. Returns {"edits", "explanation", "problems"}, plus
    "remaining_findings" once the patch could be analyzed."""
    edits, problems = _parse_edits(response, cells, editable)
    explanation = response.get("explanation", "") if isinstance(response, dict) else ""
    outcome = {"edits": edits, "explanation": explanation if isinstance(explanation, str) else "", "problems": problems}
    if problems:
        return outcome

    patched = _apply(notebook, edits)
    try:
        after = analyze_data_leaks(copy.deepcopy(patched))
    except Exception as exc:  # the analysis rejecting the code, not a bug in fix()
        detail = str(exc).strip().splitlines()[0] if str(exc).strip() else ""
        problems.append(f"NBFix could not analyze the patched code ({type(exc).__name__}: {detail}).")
        return outcome

    outcome["remaining_findings"] = after["findings"]
    patched_cells = _code_cells(patched)
    for finding in after["findings"]:
        split = split_origin(patched_cells, finding["sink"]) if finding.get("sink") else None
        where = f" (`{finding['sink']}` comes from the split in cell {split[0]}: `{split[1]}`)" if split else ""
        problems.append(f"NBFix still reports: {finding['message']}{where}")

    new_warnings = Counter(w["message"] for w in after["warnings"]) - Counter(w["message"] for w in original["warnings"])
    for warning in after["warnings"]:
        if new_warnings[warning["message"]] > 0:
            new_warnings[warning["message"]] -= 1
            problems.append(
                f"Cell {warning['cell']}, line {warning['line']} uses code NBFix cannot fully analyze "
                f"({warning['message']}); write it without that construct."
            )

    patched_code = "\n".join(patched_cells.values())
    for call in sorted({m.group(1) for f in original["findings"] if (m := _SINK_CALL.search(f["message"]))}):
        if f"{call}(" not in patched_code:
            problems.append(f"The `{call}` call where a leak was reported is gone; keep it and fix its input data instead.")

    source_methods = set()
    for call in sorted({f["source"]["call"] for f in original["findings"] if (f.get("source") or {}).get("call")}):
        # The receiver may legitimately change (MinMaxScaler().fit_transform ->
        # scaler.fit_transform), so only the method is required.
        method = call.rsplit(".", 1)[-1]
        methods = _FIT_METHODS if method in _FIT_METHODS else {method}
        source_methods |= methods
        if not any(f".{m}(" in patched_code for m in methods):
            problems.append(
                f"The `{call}` step a leak was traced to is no longer called; keep that step "
                "and fit it on the training split instead of removing or replacing it."
            )

    already_unused = unused_results(cells, source_methods)
    for name in sorted(unused_results(patched_cells, source_methods) - already_unused):
        problems.append(
            f"`{name}` is computed by the preprocessing step but never used afterwards, so the step "
            "has no effect; use its result (for example, train on it)."
        )

    for call, variable in sorted(fits_on_test_data(patched_cells) - fits_on_test_data(cells)):
        problems.append(
            f"`{call}` fits on test data (`{variable}` comes from the test split); fit only on the "
            "training split, and reuse that fitted object to transform the test data."
        )

    for call, features, labels in sorted(misaligned_splits(patched_cells) - misaligned_splits(cells)):
        problems.append(
            f"`{call}({features}, {labels}, ...)` pairs features and labels from different "
            "train_test_split calls, so the rows no longer match their labels; split features and "
            "labels together in one call."
        )

    already_undefined = undefined_reads(cells)
    for name, cell in sorted(undefined_reads(patched_cells).items(), key=lambda item: (item[1], item[0])):
        if name not in already_undefined:
            defined_in = first_binding(patched_cells, name, after=cell)
            where = (f"it is only defined later, in cell {defined_in}: move this code into cell {defined_in} "
                     "after that definition (or later), or move the definition up into an earlier cell"
                     if defined_in is not None else "nothing defines it")
            problems.append(
                f"Cell {cell} reads `{name}` before any earlier code defines it, so the patched code "
                f"would fail with a NameError - {where}."
            )

    return outcome


def _parse_edits(response, cells, editable) -> tuple[list[dict[str, Any]], list[str]]:
    raw = response.get("edits") if isinstance(response, dict) else None
    if not isinstance(raw, list):
        return [], ['The response has no "edits" list.']

    edits: dict[int, str] = {}
    problems = []
    for edit in raw:
        cell = edit.get("cell") if isinstance(edit, dict) else None
        code = edit.get("code") if isinstance(edit, dict) else None
        if not isinstance(cell, int) or isinstance(cell, bool) or not isinstance(code, str):
            problems.append(f"Malformed edit {edit!r}: expected {{\"cell\": <int>, \"code\": <str>}}.")
            continue
        if cell not in cells:
            problems.append(f"Cell {cell} is not a code cell.")
            continue
        if cell not in editable:
            problems.append(f"Cell {cell} is not editable.")
            continue
        code = _keep_original_formatting(cells[cell], _strip_fence(code))
        if code.rstrip() == cells[cell].rstrip():
            continue
        try:
            parse_cell(code)
        except SyntaxError as exc:
            problems.append(f"Cell {cell} has a syntax error on line {exc.lineno}: {exc.msg}.")
            continue
        edits[cell] = code

    if not edits and not problems:
        problems.append("The response changes no cells.")
    return [{"cell": cell, "code": code} for cell, code in sorted(edits.items())], problems


def _strip_fence(code: str) -> str:
    match = _FENCE.match(code.strip())
    return match.group("code") if match else code


def _keep_original_formatting(original: str, code: str) -> str:
    """The model's cell with formatting-only changes undone. Lines are matched
    on their content ignoring whitespace inside them and quote style
    (indentation counts - it is Python syntax); a matched line keeps its original text and the blank
    lines before it, and only changed or new lines come from the model.
    Models rewriting a whole cell (a script without cell markers is one)
    otherwise reformat lines they had no reason to touch."""
    old, old_gaps, old_tail = _split_blank_lines(original)
    new, new_gaps, _ = _split_blank_lines(code)
    matcher = difflib.SequenceMatcher(a=[_line_key(line) for line in old], b=[_line_key(line) for line in new],
                                      autojunk=False)
    kept = []
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                kept += [""] * old_gaps[i] + [old[i]]
        elif tag in ("replace", "insert"):
            for j in range(j1, j2):
                # A replacement keeps the spacing before the code it replaces.
                gap = old_gaps[i1] if tag == "replace" and j == j1 else new_gaps[j]
                kept += [""] * gap + [new[j]]
        # "delete": the lines go, with the blank lines before them.
    kept += [""] * old_tail
    return "\n".join(kept) + ("\n" if code.endswith("\n") else "")


def _split_blank_lines(text: str):
    """(non-blank lines, blank lines before each, trailing blank lines)."""
    lines, gaps, gap = [], [], 0
    for line in text.splitlines():
        if line.strip():
            lines.append(line)
            gaps.append(gap)
            gap = 0
        else:
            gap += 1
    return lines, gaps, gap


def _line_key(line: str) -> str:
    """Indentation plus content without whitespace, quote style ignored."""
    indent = len(line) - len(line.lstrip())
    content = re.sub(r"\s+", "", line).replace("'", '"')
    return f"{indent}:{content}"


def _code_cells(notebook: dict) -> dict[int, str]:
    """Cell index -> source, for the cells load_notebook analyzes."""
    cells = {}
    for index, cell in enumerate(notebook.get("cells", [])):
        source = cell.get("source")
        source = "".join(source) if isinstance(source, list) else source
        if cell.get("cell_type") == "code" and source:
            cells[index] = source
    return cells


def _editable_cells(findings, cells) -> set[int]:
    """The cells on each leak's path, plus its sink and source cells."""
    editable = set()
    for finding in findings:
        editable.update(finding.get("path") or [])
        editable.update(location["cell"] for location in finding.get("locations", []))
    return editable & set(cells)


def _describe(finding, cells) -> dict[str, Any]:
    locations = {location["role"]: location for location in finding.get("locations", [])}

    def where(role):
        location = locations.get(role)
        if location is None:
            return None
        cell, line = location["cell"], location["line"]
        return cell, line, _line_text(cells.get(cell, ""), line)

    sink = finding.get("sink")
    split = split_origin(cells, sink) if sink else None
    return {"message": finding["message"], "sink": where("sink"), "source": where("source"), "split": split}


def _line_text(code: str, line) -> str | None:
    """The source of a cell-relative line, numbered as the analysis numbers
    it: after dropping magics and leading blank lines (see remove_magic)."""
    kept = [text for text in code.splitlines() if not _MAGIC.match(text)]
    while kept and kept[0] == "":
        kept.pop(0)
    if isinstance(line, int) and 1 <= line <= len(kept):
        return kept[line - 1].strip()
    return None


def _apply(notebook: dict, edits) -> dict:
    patched = copy.deepcopy(notebook)
    for edit in edits:
        cell = patched["cells"][edit["cell"]]
        cell["source"] = edit["code"].splitlines(keepends=True) if isinstance(cell["source"], list) else edit["code"]
    return patched


def _proposal(outcome, notebook, cells, source, filename) -> dict[str, Any]:
    return {
        "explanation": outcome["explanation"],
        "edits": outcome["edits"],
        "diff": _diff(notebook, cells, outcome["edits"], source, filename),
    }


def _diff(notebook, cells, edits, source, filename) -> str:
    """A unified diff of the script file when its source is known (applies
    with `git apply`/`patch`), else one diff per edited cell."""
    if source is not None and is_script(notebook):
        name = filename or "script.py"
        new_source = _patch_script(source, notebook, cells, edits)
        return "".join(difflib.unified_diff(
            source.splitlines(keepends=True), new_source.splitlines(keepends=True),
            fromfile=f"a/{name}", tofile=f"b/{name}",
        ))

    name = filename or "notebook"
    return "".join(
        "".join(difflib.unified_diff(
            _lines(cells[edit["cell"]]), _lines(edit["code"]),
            fromfile=f"a/{name} (cell {edit['cell']})", tofile=f"b/{name} (cell {edit['cell']})",
        ))
        for edit in edits
    )


def _lines(text: str) -> list[str]:
    return [line + "\n" for line in text.splitlines()]


def _patch_script(source: str, notebook: dict, cells, edits) -> str:
    """The script with each edited cell's lines replaced. A cell's source is
    exactly its lines in the file (see script_to_notebook); its trailing blank
    lines are kept, so the spacing before the next cell marker survives."""
    lines = source.splitlines()
    starts = cell_start_lines(notebook)
    for edit in sorted(edits, key=lambda e: e["cell"], reverse=True):
        old = cells[edit["cell"]].splitlines()
        while old and not old[-1].strip():
            old.pop()
        start = starts[edit["cell"]]
        lines[start - 1:start - 1 + len(old)] = edit["code"].rstrip("\n").splitlines()
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")
