# NBCompile

Notebook → script converter - the third tool in the NBTooling suite.

Uses `nbcore`'s paper-correct cell propagation dependency graph
(`nbcore.analyses.dependency_analysis`) to compute a notebook's true
execution order - not a naive cell-order dump the way `nbconvert`
does. NBFix would then consume that script; NBCompile itself never
repairs anything.

**Status: working v1, still being validated.** Cells with no
dependencies become the linear script's entry points; genuine cycles
(a notebook cell interactively re-run based on a later cell's output -
no closed-form translation to a straight-line script) are detected as
strongly connected components and wrapped in a `while True:` loop with
a `# TODO` for a real exit condition. Ordering among independent
branches ("many subchains") isn't tuned for readability yet - any
valid topological order is used.

## Usage

```bash
nbcompile -f notebook.ipynb              # prints the script to stdout
nbcompile -f notebook.ipynb -o out.py    # writes it to a file instead
```

See the top-level repo README for how this fits into the wider suite.
