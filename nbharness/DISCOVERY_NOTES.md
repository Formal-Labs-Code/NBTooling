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
[{"cell_id":4,"errors":[{"line":2,"label":"X_selected_test","error_type":"ErrorType.TERMINAL","message":"Preprocessing leak: `X_selected_test` passed to `lr.predict` is contaminated because `min_max_scaler.fit_transform` (cell 1, line 3) ran on unsplit `X` before the train/test split, so test-set statistics leak into the model.","source":{"cell":1,"line":3,"call":"min_max_scaler.fit_transform","variable":"X"}}],"path":[0,1,2,3,4]}]
```

This is the source of truth the wrapper normalizes.

## Feasibility check A: path, sink and source

The raw result includes:

- `cell_id`: the sink cell
- `path`: the propagation chain through the notebook
- `label`: the contaminated variable at the sink
- `line`: the sink's line within `cell_id`
- `source` (preprocessing leaks only): the taint-introducing call - `cell`, `line`, `call` and the unsplit `variable` it ran on. The data-leak analysis records it when a taintKB call (`fit_transform`, `fit_resample`, ...) runs, or at the `fit()` behind a `transform()` on unsplit data, and carries it through assignments and joins.

What the wrapper populates:

- `sink`: the raw `label` value
- `source`: the raw `source` object, or `null` when absent
- `path`: the raw `path` list
- `locations`: `{"role": "sink", "cell", "line"}`, plus `{"role": "source", ...}` when a source is known
- `message`: the raw `message`

`source` stays `null` for train/test overlap leaks (the same rows used for training and evaluation): there is no single originating call, and the analyzer does not yet record where each `fit()`/`predict()` usage happened.

## Feasibility check B: notebook execution / verify

Notebook execution is feasible in this repo using the standard Jupyter stack (`nbformat` + `nbclient`) for valid `.ipynb` files. The wrapper includes an optional `verify()` helper using that approach, so it can report whether the notebook executed end-to-end and then re-run the NBHarness analysis.

The main caveat is environmental: execution depends on the kernel and any data-science libraries in the notebook being available in the current environment. The wrapper treats notebook execution as optional and reports `runs` as `True`/`False`/`None` rather than assuming execution always succeeds.

For the NBHarness-first contract, the primary tool is still `analyze()`; `verify()` is a convenience sanity check, not the main analyzer surface.

## Relevance to later PoCs

The key fact for the later verify/repair loop is that preprocessing leaks now come with a canonical `source` - the cell and line of the call to move after the split - so a repair loop can target it directly instead of reconstructing it from `path`. Overlap leaks still only give the sink and path.

This wrapper is intentionally thin: it translates the raw analyzer output without inventing new analysis semantics.
