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
- `source`: `cell`, `line`, `call` and `variable` of what caused the leak, carried through assignments and joins by the data-leak analysis:
  - preprocessing leaks: the taint-introducing call (`fit_transform`, `fit_resample`, ..., or the `fit()` behind a `transform()` on unsplit data) and the unsplit variable it ran on
  - train/test overlap leaks: the other half of the train/test pair - the `fit()` call and its training variable when the error fires at `predict()`/`score()`, or the evaluation call when it fires at `fit()`
  - when the call is inside a notebook-defined function that NBHarness inlined, names are the user's own (not the parser's internal `~inline_...` renames) and `source` also carries `function`, the function's name; the message says e.g. "`model.fit` (in `train_model`, cell 2, line 5)"

What the wrapper populates:

- `sink`: the raw `label` value
- `source`: the raw `source` object, or `null` when absent
- `path`: the raw `path` list
- `locations`: `{"role": "sink", "cell", "line"}`, plus `{"role": "source", ...}` when a source is known
- `message`: the raw `message`

## Feasibility check B: notebook execution / verify

Notebook execution is feasible in this repo using the standard Jupyter stack (`nbformat` + `nbclient`) for valid `.ipynb` files. The wrapper includes an optional `verify()` helper using that approach, so it can report whether the notebook executed end-to-end and then re-run the NBHarness analysis.

The main caveat is environmental: execution depends on the kernel and any data-science libraries in the notebook being available in the current environment. The wrapper treats notebook execution as optional and reports `runs` as `True`/`False`/`None` rather than assuming execution always succeeds.

For the NBHarness-first contract, the primary tool is still `analyze()`; `verify()` is a convenience sanity check, not the main analyzer surface.

## Relevance to later PoCs

The key fact for the later verify/repair loop is that preprocessing leaks now come with a canonical `source` - the cell and line of the call to move after the split - so a repair loop can target it directly instead of reconstructing it from `path`. For overlap leaks the source is the training (or evaluation) call whose data has to be narrowed to its own split.

This wrapper is intentionally thin: it translates the raw analyzer output without inventing new analysis semantics.
