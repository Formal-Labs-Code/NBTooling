# NBHarness MCP discovery notes

## Invocation and raw output

The real NBHarness entry point is the existing batch-analysis API in `nbharness.cli.nbharness()`. It loads a notebook dict into an `AnalysisSession`, adds the selected analyses, and then emits the raw JSON result from `Result.dumps(True)`.

For the known leaky fixture in `tests/resources/dataleak_true.ipynb`, the invocation is effectively:

```python
from nbcore.analyses.dataleak_analysis import DATA_LEAK
from nbcore.resource_utils.utils import read_json
from nbharness.cli import nbharness

notebook = read_json("tests/resources/dataleak_true.ipynb")
print(nbharness(None, notebook, [DATA_LEAK], 0, 10))
```

The raw output is:

```json
[{"cell_id":4,"errors":[{"line":2,"label":"X_selected_test","error_type":"ErrorType.TERMINAL","message":"Training model with data leak."}],"path":[0,1,2,3,4]}]
```

This is the source of truth the wrapper normalizes.

## Feasibility check A: path vs sink

NBHarness exposes the sink and the propagation path, but not a separate explicit source provenance field. The raw result includes:

- `cell_id`: the sink cell
- `path`: the propagation chain through the notebook
- `label`: the variable/value at the sink that is contaminated
- `line`: the source-side/definition line metadata for the error

What the wrapper can populate:

- `sink`: the raw `label` value
- `path`: the raw `path` list
- `locations`: `cell {cell_id}` and `line {line}`
- `message`: the raw `message`

What it cannot honestly synthesize:

- `source`: not explicitly exposed by the raw analyzer; the result gives the contaminated sink and path, not a separate origin node or provenance chain beyond the notebook path itself

This means the wrapper should keep `source` as `null` unless the underlying analyzer later exposes a more specific origin field.

## Feasibility check B: notebook execution / verify

Notebook execution is feasible in this repo using the standard Jupyter stack (`nbformat` + `nbclient`) for valid `.ipynb` files. The wrapper includes an optional `verify()` helper using that approach, so it can report whether the notebook executed end-to-end and then re-run the NBHarness analysis.

The main caveat is environmental: execution depends on the kernel and any data-science libraries in the notebook being available in the current environment. The wrapper treats notebook execution as optional and reports `runs` as `True`/`False`/`None` rather than assuming execution always succeeds.

For the NBHarness-first contract, the primary tool is still `analyze()`; `verify()` is a convenience sanity check, not the main analyzer surface.

## Relevance to later PoCs

The key fact for the later verify/repair loop is that this analysis is already path-aware enough to report the sink and the notebook-level propagation chain, but it does not provide a clean, separate `source` field. That means a later verifier can rely on the existing `path` and `message`, while a repair loop will still need to infer or reconstruct the exact earlier cell(s) responsible for the contamination, rather than receiving a canonical source field from the analyzer.

This wrapper is intentionally thin: it translates the raw analyzer output without inventing new analysis semantics.
