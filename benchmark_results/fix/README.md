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

## Run 8: restructured prompt (16 fixtures)

The prompt was rewritten after tracing 14b's failures on the two-leak notebook
(it fitted model 2's scaler on model 1's split, then resubmitted the same
rejected fix twice): sections, correctness rules before "change as little as
possible", restructuring cells stated as expected, each leak names the split
that produces the evaluated data, and retries are numbered, say the analysis is
exact, and flag a repeated fix.

| Model | Verified | Correct | Verified but wrong | Attempts | Time |
| --- | --- | --- | --- | --- | --- |
| qwen2.5-coder:14b | 16 (was 15) | 15 (was 15) | 1 (was 0) | 21 (was 24) | 438 s (was 456 s) |
| qwen2.5-coder:7b | 10 (was 10) | 9 (was 9) | 1 (was 1) | 31 (was 28) | 305 s (was 236 s) |

14b now fixes the two-leak notebook on its first attempt; its one wrong fix
(two-leak script) unpacks a `train_test_split` into 2 names instead of 4, which
no check covers yet. 7b gained imputer (over-restructured, but correct) and the
two-leak script, and lost script_cells and the two-leak notebook.

### Why first attempts fail (run 8)

Traced attempt by attempt for the 12 fixtures where attempt 1 was rejected:

| What the model got wrong | 14b | 7b |
| --- | --- | --- |
| Fixed the leak in a cell that runs before the split (`X_train` isn't defined yet) | imputer, smote | readexcel, rebind, smote |
| Left later cells using names it removed (`X_filled`, `X_scaled`, `X_selected`, `b_train`) | | imputer, script_cells, dataleak_true, two_leaks.py |
| Fixed the preprocessing, but a helper refits on test data | helpers | |
| Re-split the already-split training data, still leaking | | two_leaks.ipynb |
| Fixed the one leak reported; the second surfaced after (analysis limitation) | two_leaks.py | |

9 of 12 are the same blind spot: the fix is a local edit where the leak is
reported, without following the data back to the split that creates the
training set or forward to the cells using the changed variable.

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
| 8 | 16 | Restructured prompt; leaks name their split; retries flag repeated fixes | 16 (1); 7b: 10 (1) |

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
