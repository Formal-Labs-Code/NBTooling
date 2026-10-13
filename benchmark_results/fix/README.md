# NBFix `fix` evaluation

`scripts/eval_fix.py` runs `nbfix.repair.fix` on every leaky fixture
(`tests/resources/dataleak*_true.*`) with a local model (Ollama), up to 3
attempts each. Each `runN_<model>.json` holds every result, including diffs,
explanations and the rejection reasons of the last attempt.

"Verified" means NBFix's checks accepted the patch. Whether a verified patch
is actually a correct fix was judged by reading every diff.

## Baseline (run 7, 16 fixtures)

| Model | Verified | Correct | Verified but wrong | Unverified | Time (16 fixtures) |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-coder:14b | 15 | 15 | 0 | 1 | ~7.5 min |
| qwen2.5-coder:7b | 10 | 9 | 1 | 6 | ~4 min |

- Every unverified result was a correct rejection: no fix was rejected that
  should have been accepted.
- **14b** needs retries where the split must move to an earlier cell than the
  preprocessing step (imputer: 3 attempts, smote: 2), and for the two-leak
  script, whose second leak only shows once the first is fixed (3). It can't
  fix the two-leak notebook: it re-splits the training data, which still leaks.
- **7b** mostly fails at moving code across cells (imputer, readexcel, rebind,
  smote, dataleak_true: reads a name before the cell that defines it). Its one
  wrong verified fix is helpers: it trains on scaled data but evaluates on the
  unscaled test set - a train/test preprocessing mismatch, which no check
  covers yet.
- A repair takes 10-25 s per attempt with 14b (the model writes out every edited
  cell), about half that with 7b. NBFix's own analysis and checks take
  milliseconds.

## Run history

| Run | Fixtures | Change | 14b verified (wrong) |
| --- | --- | --- | --- |
| 1 | 12 | First version: no leaks left, no new warnings, sink call kept | 12 (4) |
| 2 | 12 | Prompt: fit once on train and reuse; transform the evaluated data too; copy unchanged lines; keep preprocessing steps. Check: the source step is still called | 12 (3) |
| 3 | 12 | Checks: no name read before an earlier cell defines it; the source step's result is used | 10 (0) |
| 4 | 12 | Prompt: cells run top to bottom; move preprocessing after a later split | 11 (0) |
| 5 | 16 | Added imputer, feature-selection and two-leak fixtures. Formatting-only changes undone; check: nothing fitted on test data | 13 (1) |
| 6 | 16 | Check: features and labels from the same split. Feedback names the cell that defines a name read too early | 15 (0); 7b: 11 (2) |
| 7 | 16 | Check: names read inside functions; quote style ignored as formatting | 15 (0); 7b: 10 (1) |

Verified but wrong along the way (none is a data leak, so re-analysis alone
can't see them; each led to the check or prompt change in the next run):

- **smote** (run 1): `SMOTE` replaced by a do-nothing class - the leak "fixed"
  by removing the resampling.
- **smote** (run 2), **rebind** (runs 1-2): preprocessing moved before the
  split that defines its input (NameError); smote also trained on the
  unresampled data.
- **helpers** (runs 1-3, 14b): fits a second scaler on the test set. Prompt
  (run 2) and check (run 5).
- **readexcel** (run 1): trains on scaled data, scores on unscaled `X_test`.
  Prompt only.
- **two-leak script** (run 5, 14b): labels re-split on their own, so the second
  model trains on features and labels from different splits.
- **two-leak script** (run 6, 7b): uses `b_train` after removing its definition,
  inside `main()`.
- **helpers** (runs 6-7, 7b): trains on scaled data, evaluates on unscaled test
  data. Not caught yet.

## Known gaps

- Train/test preprocessing mismatches (training on transformed data, evaluating
  on untransformed data) are guarded only by the prompt.
- The analysis stops a path at its first leak, so a later leak on the same path
  only appears after the first is fixed, costing an attempt.
- A helper function defined in one notebook cell and called from another isn't
  inlined, so a leak inside it isn't found (and can't be fixed).
- Results only keep the last attempt's rejection reasons; a per-attempt history
  would show what drives retries.
- Comments describing the old (leaky) code can be left in place.
- Prompts contain the whole notebook; Ollama's default 4096-token context is
  enough for these fixtures (largest request ~1.4k tokens) but not for large
  notebooks - see RUNNING.md.
