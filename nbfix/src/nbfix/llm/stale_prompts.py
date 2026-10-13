"""
Prompt for LLM-based stale-cell detection. Kept separate from
prompts.py since the question and JSON contract are entirely different
(see stale_context_builder.py's docstring for why staleness needs its
own context shape in the first place).

Only the "operational" framing from scripts/benchmark_stale_llm.py's
research is used here - the alternative ("which cells currently show an
out-of-date value," a retrospective question) was tried and measured to
produce a specific, consistent wrong answer: both a local model and a
separate Claude session, with no shared context, independently included
the cell reading the edited variable *directly* as "stale," when running
it next would actually be correct (its fresh value is already in the
kernel, since the edited cell was just re-executed as part of triggering
this check). The operational framing - "would running this cell right
now give a wrong answer" - is the question that actually matches what
the real StaleCellAnalysis computes, and was confirmed (not assumed) to
fix the mismatch for a capable-enough model. See
benchmark_results/llm_bench/experiments.md finding 13 for the full
before/after comparison this prompt is built from.

Still not enough on its own, though: a live check against qwen2.5-coder:14b
produced the exact same wrong call again ("this cell reads x, which has
not been updated..." - flagging the direct 1-hop reader), despite the
operational framing above. The real StaleCellAnalysis (code_impact_abs_
state.py's CodeImpactAS.condition) never reasons in prose at all - it
propagates an integer "hop count" from the changed variable through the
def-use chain and only flags a cell once that count reaches K=2 (a direct,
1-hop reader is deliberately exempt - see stale_cell_analysis.py's
docstring for why level=2 was the empirically-tightest correct threshold).
The prompt below now states that same hop-count rule mechanically instead
of trusting free-form "would this be correct" reasoning to rediscover it,
and rephrases the final question as a safety call ("should NOT be
executed") rather than a temporal one ("cannot yet be re-executed") -
the latter phrasing is exactly what produced the "has not been updated"
answer, since it reads equally well as "which cells are behind."

Response schema changed from {cell_id, message} objects to a bare int
list - measured via scripts/benchmark_stale_llm.py (--prompt-variant
product vs hop_count, qwen2.5-coder:14b/7b against
tests/resources/llm_bench_stale): dropping the free-text explanation cut
completion tokens ~7x (77 -> 11 avg) and wall-clock latency ~3x (7.1s ->
2.3s at 14b alone; ~6x combined with also switching to 7b). Real
tradeoff, not free: the terse 14b variant reproducibly false-positived on
the negative-control fixture (clean1) where the message-requiring variant
got it right, suggesting having to articulate *why* a cell is stale acts
as a self-check the bare-int format skips. Chosen anyway - for live,
interactive use the latency was measured unusable, and staleness findings
here are advisory, not a hard gate.
"""
from .stale_context_builder import StaleDetectionContext

STALE_SYSTEM_PROMPT = """You are deciding which cells in a Jupyter notebook are SAFE to execute RIGHT NOW, given the kernel's actual current memory state.

Cells execute one at a time, each using whatever values currently sit in
the kernel's memory at that moment. Editing a cell's code does not, by
itself, change anything in memory - only actually re-running a cell does.

One cell was just edited AND re-executed - so whatever variable(s) it
assigns are NOW FRESH in the kernel, fully up to date. Never flag that
cell itself, and never flag a cell just because it directly reads a
variable the edited cell assigns - that value is already fresh, so it
is fine to run next.

Every OTHER cell has NOT been re-run since the edit. Each one's own
variables in the kernel still hold whatever was computed the last time
THAT cell ran, using whatever was in memory back then.

Reason in hops from the edited cell, the same way you'd trace a chain of
dominoes:
- Hop 1: a cell that directly reads a variable the edited cell just
  assigned, and has not itself been re-run since. Running it RIGHT NOW is
  correct - the value it needs is already fresh. A hop-1 cell is fine to
  run next - never flag it, even though it "hasn't been re-run since the
  edit" in a literal sense.
- Hop 2 or more: a cell that depends (directly, or through other
  never-rerun cells) on what a hop-1-or-later cell itself COMPUTED on its
  OWN last run, before the edit - not on the freshly-edited variable
  itself. Running it RIGHT NOW would silently use that stale intermediate
  value instead of a fresh one. A hop-2-or-later cell should NOT be
  executed next.
- A cell that HAS already been re-run since the edit is itself fully
  fresh, and resets the hop count back to 0 for anything depending on it
  from that point on.

Your task: list every cell that should NOT be executed next, because
running it RIGHT NOW would silently compute a wrong result using a value
that is itself still stale (hop 2 or more). Do not list hop-1 cells or
the edited cell itself.

If the "edited cell" section below shows identical original and current
code, nothing has actually changed - respond with no findings.

Respond with a single JSON object of exactly this shape and nothing else:

{"stale_cells": [<int>, ...]}

List the cell_id of every cell that should NOT be executed next. If none, respond {"stale_cells": []}.
"""


def build_stale_user_prompt(context: StaleDetectionContext) -> str:
    sections = [
        "## Edited cell\n"
        f"Cell {context.edited_cell} was last run with:\n```python\n{context.original_code}\n```\n"
        f"It has since been edited to:\n```python\n{context.current_code}\n```\n"
        f"Cell {context.edited_cell} has just been re-executed with this new code."
    ]
    sections.append("## Notebook (current state)")
    for cell_id in sorted(context.cells):
        sections.append(f"### Cell {cell_id}\n```python\n{context.cells[cell_id]}\n```")
    sections.append(
        "## Question\nWhich cells should NOT be executed next, because running "
        "them right now would silently produce a wrong result?"
    )
    return "\n\n".join(sections)
