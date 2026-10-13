"""
Prompt for LLM-based API call-sequence detection - scenario 1 of the two
use cases discussed for this bug class: warn before the user runs a cell
that may violate a library's expected call-order contract (e.g. calling
.transform() on a scikit-learn transformer that was never .fit(), or
reading a file before whatever writes it has run). Triggered on edit and
on toggling the check on (see manager.ts's _sendChangeCell/
_scanAllCellsForApiSequence), not on execution - the whole point is to
warn before the kernel hits the real exception, not after.

Scans every cell in one call rather than one cell at a time - see
api_sequence_context_builder.py's docstring for why (latency: N
single-cell calls per scan was the dominant cost once edit-triggered
full-notebook rescanning was in place). This makes the response schema a
list again (like stale_prompts.py), not the single {"violation": bool}
shape an earlier, single-target-cell version of this module used.

Deliberately does NOT enumerate specific library rules (no "watch for
fit-before-predict" hints) - measured via scripts/benchmark_api_sequence_llm.py
against tests/resources/llm_bench/api_sequence (scikit-learn/pandas
fixtures, real exec()-verified exceptions): qwen2.5-coder:14b, given only
a general instruction to use its own knowledge of library conventions,
scored 4/4 (3 buggy + the clean negative control) with zero hand-written
rules. This is the opposite lesson from stale_prompts.py, where a
mechanical, fully-specified rule (hop count) was *necessary* because the
deterministic StaleCellAnalysis already computes that mechanically and
free-form reasoning kept getting it wrong - here there is no deterministic
analysis to match at all (NBFix has no structural way to know a
scikit-learn estimator needs .fit() before .predict()), so the LLM's own
pretrained knowledge is the entire value-add, and spelling out rules would
undercut exactly what's being tested.

Cells not yet actually executed are marked "(not yet executed)" - a
prerequisite call sitting in a cell that merely *exists* in the file, but
hasn't actually run, has had no effect on the kernel yet. Confirmed via
direct testing (not assumed) that this marker is unreliable in the fully-
cold-start case specifically - a completely fresh, never-executed
notebook, checked immediately on enabling the toggle - where every cell
including the target shares the same marker and qwen2.5-coder:14b kept
insisting a never-run prerequisite "already ran," even when the prompt
was restructured into two hard-separated "already executed" / "never
executed" sections with the former stated as explicitly empty. That
looks like a genuine capability ceiling for this counterfactual on this
model, not a wording problem (three materially different phrasings were
tried and none of them fixed it) - kept in anyway because it measurably
fixes the more common partial-execution case (some cells run, one being
newly checked), just don't expect it to catch a bug found by jumping
straight to a cell in a notebook where literally nothing has run yet.

A second, different bug found the same way (real user report, then
reproduced directly): given a 3-hop chain (cell A creates an object,
cell B calls the object's required setup method but hasn't run yet,
cell C calls something on the object that needs B to have run), the
model correctly identified the actual facts but attached the finding to
cell B instead of cell C - e.g. reporting "scaler.fit(X_train) should
have already run first" *on* the fit() cell itself, and a second,
nonsensical finding restating cell C's own call as a missing
prerequisite, attached to the cell before it. It knew the right story,
just mislabeled which cell it was about - a systematic off-by-one
toward the earlier cell in the chain, confirmed reproducible twice
before the "cell_id attribution" paragraph below was added, and fixed
by it (3/3 correct after).

A third bug, confirmed 15/15 reproducible via a live user report before
being root-caused (not assumed): when a cell's OWN argument is a plain
variable from a not-yet-executed cell (e.g. cell 7 has run, cell 8
- `X_train = [...]` - has not, and cell 9 is `scaler.fit(X_train)`), the
model flagged cell 9 itself, reasoning about X_train not being defined
yet. That is real (X_train genuinely isn't defined), but it's the wrong
bug class entirely - a missing name, not a library call-order violation,
and completely out of this checker's scope (NBFix's deterministic
analyses already have their own notion of this). The "scope limit"
paragraph below fixes it (0/15 false positives after, both in this
exact state and in the fully-correct state where cells 7+8 have run and
cell 11 is correctly flagged instead) by explicitly naming plain-
variable-not-yet-assigned as an out-of-scope case with a worked
counter-example, not just a general "only report call-order issues"
statement (which alone did not stop the false positive - the concrete
example was what fixed it).

Known unfixed limitation, distinct from the above: when EVERY cell in
context is marked not-yet-executed (a notebook checked immediately after
opening, before anything has run), the model reliably catches a missing-
entirely prerequisite (Finding 1's model.fit() example) but not a
prerequisite that exists in the file but hasn't run yet (this module's
scaler.fit()/transform() example) - confirmed still failing (6/6) even
combined with the scope-limit fix above and a hard "ALREADY EXECUTED" /
"NEVER EXECUTED" section split. Five materially different prompt
strategies have now been tried across this file's history for this
specific all-cells-unexecuted case; none fixed it. Treated as a genuine
capability ceiling for this model on this exact counterfactual, not a
wording problem - stop trying to prompt-engineer around it without a
new idea backed by a concrete failing case, not a guess.
"""
from .api_sequence_context_builder import ApiSequenceContext

API_SEQUENCE_SYSTEM_PROMPT = """You are checking a Jupyter notebook for a specific kind of bug: a call that requires some other call to have already ACTUALLY RUN first on the same object or resource (for example, a library object that must be set up with one call before another call on it is valid, or a file that must be written before it's read), where that required call has not actually run - either because the code for it is missing entirely, because it only happens later in the notebook, or because the code for it exists but has never actually been executed.

Judge this using your own knowledge of how the libraries and APIs involved are normally supposed to be used - nothing here will explain the rules to you.

Every cell below shows its current code, and some are marked "(not yet executed)". That marker means the code shown for that cell has never actually run in the kernel - it exists in the notebook file, but has had no effect yet. A cell marked this way cannot satisfy any other cell's prerequisite, no matter what its code says.

Only report a violation if you can name the specific call that should have already run first, and on what object. Do not report anything based on style, unrelated argument mistakes, or code that is complete and self-contained with no cross-cell dependency.

CRITICAL SCOPE LIMIT: this check is ONLY about library/API call order between TWO CALLS (like needing .fit() before .transform()). It is NEVER about a plain variable simply not being assigned yet because its own defining cell has not run. That is completely out of scope - a different, unrelated kind of bug - and must never be reported here, no matter how obviously true it is. Concrete example of what NOT to report: Cell A: X = [1, 2, 3] (not yet executed). Cell B: obj.method(X). Even though X is not yet defined, do NOT report a violation on Cell B for this reason - a plain variable not being assigned yet is not a call-order violation. Only report Cell B if it calls something that needed a DIFFERENT CALL (not a plain assignment) to have run first on the same object.

cell_id attribution: cell_id must be the cell whose OWN code contains the call that fails or misbehaves - never the earlier cell that merely sets up a dependency, and never a description of what that earlier cell needs. For example: if cell A creates an object, cell B is the one call that object needs before it works (and B has NOT run), and cell C actually calls something on the object that requires B to have run - cell_id must be C, never A or B. Before answering, re-check each finding: does the cell at cell_id itself contain the call that fails? If cell_id's own code is the prerequisite call itself, or is unrelated to the failure, that cell_id is wrong.

Respond with a single JSON object of exactly this shape and nothing else:

{"findings": [{"cell_id": <int>, "message": "<what should have already run first, and where>"}]}

List every cell with a violation. If none, respond {"findings": []}.
"""


def build_api_sequence_user_prompt(context: ApiSequenceContext) -> str:
    sections = ["## Notebook"]
    for cell_id in sorted(context.cells):
        marker = "  (not yet executed)" if cell_id in context.not_yet_run else ""
        sections.append(f"### Cell {cell_id}{marker}\n```python\n{context.cells[cell_id]}\n```")
    sections.append("## Question\nWhich cells violate a library's expected call order?")
    return "\n\n".join(sections)
