# NBFix `fix` evaluation

`scripts/eval_fix.py` runs `nbfix.repair.fix` on the 12 leaky fixtures
(`tests/resources/dataleak*_true.*`) with a local `qwen2.5-coder:14b` (Ollama),
3 attempts each. Each `runN_*.json` holds every result, including diffs,
explanations and rejection reasons.

"Verified" means NBFix's checks accepted the patch. Whether the patch is a
correct fix was judged by reading every diff.

| Run | Change | Verified | Verified but wrong |
| --- | --- | --- | --- |
| 1 | First version: no leaks left, no new warnings, sink call kept | 12 | 4 |
| 2 | Prompt: fit once on train and reuse; transform the evaluated data too; copy unchanged lines; keep preprocessing steps. Check: the source step is still called | 12 | 3 |
| 3 | Checks: no name read before an earlier cell defines it; the source step's result is used | 10 | 0 |
| 4 | Prompt: cells run top to bottom; move preprocessing after a later split | 11 | 0 |

Verified but wrong, runs 1-2 (none is a data leak, so the analysis alone can't
see them):

- **smote** (run 1): `SMOTE` replaced by a do-nothing class - the leak "fixed"
  by removing the resampling. Caught by the source-step check (run 2).
- **smote** (run 2): resamples `X_train` in a cell before the split defines it
  (NameError), and trains on the unresampled data. Caught by the run-3 checks.
- **rebind** (runs 1-2): scales `X_train` in a cell before the split defines it
  (NameError). Caught by the run-3 checks.
- **helpers** (runs 1-2): fits a second scaler on the test set. Fixed by the
  run-2 prompt in runs 3-4, but no check catches it.
- **readexcel** (run 1): trains on scaled data but scores on unscaled `X_test`.
  Fixed by the run-2 prompt, but no check catches it.

Run 4: 11 verified and all correct; smote unverified (the model shifted code
down a cell and lost the `lr.predict` call - rejected by the sink check).

Known gaps:

- Fitting a preprocessing step on test data, and train/test preprocessing
  mismatches, are not checked - only the prompt guards against them.
- Scripts without cell markers are one cell, so the model rewrites the whole
  file and sometimes changes blank lines or spacing on lines it didn't need to
  touch.
- Comments describing the old (leaky) code can be left in place.
