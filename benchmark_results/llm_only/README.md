# LLM alone vs NBFix `fix`

Baseline for NBFix's `fix`: the same local models (Ollama) asked to find and fix
data leakage on their own. `scripts/eval_llm_only.py` gives each fixture to the
model with a generic prompt ("review this code for data leakage between training
and test data, and fix any you find"), the cells numbered as in NBFix's prompt,
and asks for the leaks found plus whole-cell edits as JSON. One attempt,
temperature 0, no NBFix analysis in the prompt, no checks, no retries.

Fixtures: the 16 leaky `tests/resources/dataleak*_true.*` and the 7 leak-free
`dataleak*_false.*`. Each `qwen2.5-coder-*.json` holds every response, the diff,
whether the patched code parses, and NBFix's analysis of the original and the
patched code (supporting evidence only).

Correctness was judged by reading every result: **detection** is right if the
model reports the actual leak(s) for the right reason; a **fix** is correct if
the leak is gone and the code still runs and computes the same thing otherwise.
The NBFix `fix` columns are evaluation run 8 (`../fix/run8_*.json`), judged the
same way.

## Leaky fixtures

| Fixture | LLM-only 14b: detect / fix | LLM-only 7b: detect / fix | NBFix `fix` 14b | NBFix `fix` 7b |
| --- | --- | --- | --- | --- |
| fit_transform_split | ✅ / ✅ | ✅ / ✅ | ✅ | ✅ |
| imputer | ✅ / ✅ (fragile: relies on two identical splits) | ✅ / ❌ NameError | ✅ | ✅ |
| overlap | ✅ / ✅ | ✅ / ✅ | ✅ | ✅ |
| readexcel_score | ✅ / ❌ NameError | ✅ / ❌ NameError | ✅ | ❌ |
| rebind_before_split | ❌ missed | ❌ wrong reason, leak remains | ✅ | ❌ |
| script_cells | ❌ missed | ❌ missed | ✅ | ❌ |
| script_flat | ✅ / ✅ | ✅ / ✅ | ✅ | ✅ |
| script_helpers | ✅ / ❌ evaluates on unscaled data | ✅ / ❌ replaced the script with a fragment | ✅ | ⚠️ |
| script_main | ❌ missed | ❌ wrong reason; points `y` at a nonexistent file | ✅ | ✅ |
| script_nested_call | ❌ missed | ❌ missed | ✅ | ✅ |
| selectkbest | ✅ / ✅ | ✅ / ✅ | ✅ | ✅ |
| shadowed_param | ✅ / ✅ | ❌ wrong reason; deleted the training call | ✅ | ✅ |
| smote | ✅ / ❌ resampling no longer used | ✅ / ❌ NameError | ✅ | ❌ |
| dataleak_true | ❌ missed | ❌ missed | ✅ | ❌ |
| two_leaks.ipynb | ✅ both / ❌ row-count mismatch | ✅ both / ❌ NameError | ✅ | ❌ |
| two_leaks.py | ✅ both / ❌ unfitted imputer | ✅ both / ❌ second leak remains | ⚠️ | ✅ |
| **Leaks detected** | **11/16** | **10/16** | **16/16** | **16/16** |
| **Correct fixes** | **6/16** | **4/16** | **15/16** | **9/16** |

✅ correct · ❌ wrong or failed (for NBFix `fix`: no verified fix, every one a
correct rejection) · ⚠️ verified by NBFix but wrong (14b: unpacks a split into
2 names instead of 4; 7b: evaluates on unscaled data).

## Leak-free fixtures (7)

- **LLM-only 14b** flagged 2: `dataleak_false` for a nonsense reason ("fitting
  the scaler on the training data causes leakage" - though its edit happens to
  remove a refit on the test set), and `readexcel_score_false` for `y = X`
  (real target leakage, but a fixture artifact; the fix is a non-runnable
  placeholder column name).
- **LLM-only 7b** flagged 2: `overlap_false` ("y is random" - not a leak), and
  `dataleak_false` for `y = df` (fair: target leakage) and a refit on the test
  set (a bug, not a leak).
- **NBFix** flagged none.

## Findings

- **Fixing is where NBFix wins**: with the same model, 15 vs 6 correct fixes
  (14b) and 9 vs 4 (7b). Most LLM-only fixes fail the way NBFix's first attempts
  do (see `../fix/README.md`): the edit is made where the leak is, without
  following the data to the split that creates the training set or to the cells
  using the changed variable - but with no checks, the broken code ships.
- **The LLM alone misses about a third of the leaks**: typically where the leak
  spans cells (dataleak_true and script_cells - scale, then split in a later
  cell) or is indirect (rebinding `X`, a nested call, code inside `main()`).
- **The LLM alone finds both leaks of the two-leak script at once**; NBFix's
  analysis reports only the first until it's fixed.
- **One NBFix analysis false positive**: on 14b's (fragile but correct) imputer
  fix, the analysis reports 2 leaks - it can't tell that two splits are
  identical.
- **Caveat**: NBFix's detection numbers come from fixtures written alongside its
  analysis, so they favor it; the fix comparison is fairer, since both get the
  same leaks to fix.
