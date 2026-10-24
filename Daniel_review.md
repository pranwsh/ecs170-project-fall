# Daniel's Stage 2 Code Review

Reviewed on October 23, 2026 against [main at 281a7ed](https://github.com/pranwsh/ecs170-project/tree/281a7ed426cf7166d8f26a6dc62f661e5516c29b).

This document lists proposed fixes, not changes already applied. Links point to the reviewed commit so their line numbers remain stable.

## Review results

- All 27 automated tests passed during the review.
- All 35 saved runs passed checks comparing their accuracy with confusion-matrix counts, evaluation sample counts, history lengths, and applicable final validation-history values.
- No required change was found in Daniel's data loader, visualizations, or preprocessing preparation.
- Stage 1 and the professor's base classes are unchanged from the reviewed original-template baseline.
- These checks do not mean every experiment was retrained or every possible configuration was tested.

## Fix order

| Issue | Priority | Suggested owner |
|---|---|---|
| 1. Cached results ignore configuration changes | Fix before running more experiments | Persons 3/4 |
| 2. Summaries mix validation and test results | Fix before using summary comparisons | Persons 3/4 |
| 3. Report compares accuracy across different datasets | Fix before submission | Report contributors / Person 4 |
| 4. Discussion contradicts recorded training curves | Fix before submission | Daniel / experiment authors / Person 4 |
| 5. Incorrect MSE explanations | Correct before quoting in report | Person 3 |
| 6. Experiment instructions do not match code | Small documentation cleanup | Person 3 |

## 1. Cached results ignore newly requested settings

**Location:** [run_experiments.py:194-203](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/script/stage_2_script/run_experiments.py#L194).

**Issue:** The runner reuses a run whenever `metrics.json` exists and `--force` is absent. It does not check the saved `config.json` against the configuration requested now.

**Verified example:** Requesting `--groups preprocessing --epochs 1 --seed 999 --on-test` reused the saved seed-2, 20-epoch validation results and did not train. This was reproduced with directory creation and summary writing disabled, so the existing results were not changed.

**Why it matters:** Someone can request different training settings or final test evaluation and receive measurements from an older, different experiment.

**Proposed solution:**

1. Before reuse, compare the saved configuration with the requested configuration.
2. Compare all experiment-defining fields, including seed, normalization, architecture, activation, dropout, optimizer settings, loss, batch size, epochs, device, and evaluation dataset. Exclude output-only fields such as `run_seconds`.
3. Reuse matching results. If settings differ, stop with a clear explanation instead of returning the old scores. Handle missing or unreadable configuration as an unusable cache entry.
4. Keep `--force` as an explicit replacement operation. Document that it overwrites the previous run. Use distinct folders when both configurations need to be preserved, especially validation versus test runs.

**Verify the fix:** An identical configuration should reuse results. Changing seed, epochs, or evaluation dataset should not reuse old results. An explicit forced run should save the requested configuration and new measurements.

## 2. Summaries mix validation and test measurements

**Locations:** [Convergence_Plots.py:261-266](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/local_code/stage_2_code/Convergence_Plots.py#L261), [summarize_results.py:42-58](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/script/stage_2_script/summarize_results.py#L42), and [run_experiments.py:151](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/script/stage_2_script/run_experiments.py#L151).

**Issue:** `collect_results()` reads configuration files but drops `evaluated_on`. The summary then ranks validation and test scores together without an evaluation-set column. The group comparison title also always says "validation split," even with `--on-test`.

**Why it matters:** Validation experiments evaluate 12,000 examples after training on 48,000. Final test runs evaluate a different 10,000 examples after training on all 60,000. A single unlabeled ranking encourages invalid comparisons.

**Proposed solution:** Preserve `evaluated_on` in collected rows, include it in exported tables, and generate separate validation and test tables/figures. Update the table writer to support this text column; its current score columns are formatted as numbers. Build group titles from the actual evaluation dataset. The error-analysis helpers can similarly use "evaluation" instead of always calling the examples "test."

**Verify the fix:** Use one saved validation run and one saved test run. Both should retain their evaluation identity and appear in the correct separate table. A test-run comparison must not be titled as validation.

## 3. Reported improvement compares different datasets

**Location:** [ANALYSIS.md:188-195](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/result/stage_2_result/person3/ANALYSIS.md#L188).

**Issue:** The claimed 1.62-percentage-point gain subtracts baseline validation accuracy from the best final test accuracy. The claim that validation and test agree within 0.2 percentage points for every configuration is also false.

**Correct comparisons:**

| Comparison | Baseline | Combined configuration | Improvement |
|---|---:|---:|---:|
| Same test dataset | 98.07% | 98.65% | 0.58 percentage points |
| Same validation dataset | 97.0333% | 98.3833% | 1.35 percentage points |

Baseline validation and test scores differ by about 1.0367 percentage points. The combined configuration differs by about 0.2667 points. The final test models also use more training examples, so these are not the same trained weights evaluated twice.

**Proposed solution:** Replace the mixed-dataset improvement with a same-dataset comparison. Correct the agreement statement and label validation and test results explicitly. Do not infer the absence of test-set overfitting solely from similar scores.

**Verify the fix:** Trace each percentage to its saved `metrics.json` and confirm that both terms in every improvement calculation use the same evaluation dataset. No retraining is needed to correct this arithmetic.

## 4. Discussion contradicts the saved training curves

**Locations:** [ANALYSIS.md:89-92](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/result/stage_2_result/person3/ANALYSIS.md#L89) and [ANALYSIS.md:118-135](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/result/stage_2_result/person3/ANALYSIS.md#L118).

**Issue:** The discussion says there is no training/validation gap and describes normalized validation loss as falling. The saved baseline history shows:

- Final training accuracy: about 99.32%; final validation accuracy: about 97.03%.
- Final training loss: about 0.01937; final validation loss: about 0.14634.
- Normalized validation loss reaches its minimum around epoch 7, about 0.08790, then ends higher at 0.14634.
- Normalized versus raw final validation accuracy differs by 4 correct predictions out of 12,000, not 3.

**Proposed solution:** Describe the actual gap and the initial improvement followed by deterioration in validation loss. Report normalized inputs' lower final validation loss separately from their very similar final accuracy. Present explanations about Adam, confidence, capacity, or stability as hypotheses unless directly measured; a single seed does not establish a universal normalization benefit or statistical equivalence.

Also soften [the fixed significance cutoff at ANALYSIS.md:224-227](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/result/stage_2_result/person3/ANALYSIS.md#L224). A binomial standard-error estimate for one accuracy is not a significance test for paired model predictions and does not measure training-seed variability. Small differences can be described as inconclusive from the single seed/split used.

**Verify the fix:** Check statements against `baseline/baseline/history.json` and both preprocessing histories, using one-based epoch numbers for the figures. Check image-count differences using unrounded accuracies or confusion-matrix totals.

## 5. Incorrect MSE explanations

**Locations:** [Loss_Functions.py:15-20](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/local_code/stage_2_code/Loss_Functions.py#L15) and [ANALYSIS.md:74-75](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/result/stage_2_result/person3/ANALYSIS.md#L74).

**Issue:** MSE does not require probability targets. This wrapper chooses softmax probabilities and one-hot targets for a classification experiment. The analysis also incorrectly says its gradient contains the softmax derivative squared.

**Proposed solution:** Keep the implementation. Describe the wrapper as "MSE between softmax probabilities and one-hot labels." Remove the derivative-squared statement and avoid presenting slower convergence than cross-entropy as guaranteed. If gradient behavior is discussed, the chain rule applies the softmax Jacobian once to the MSE residual gradient.

**Verify the fix:** The description should match the existing softmax, one-hot conversion, and MSE calculation. Existing loss tests should continue passing because the numerical implementation need not change.

## 6. Experiment instructions do not match the code

**Locations:** [run_experiments.py:15](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/script/stage_2_script/run_experiments.py#L15), [preprocessing comment at lines 90-97](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/script/stage_2_script/run_experiments.py#L90), and [DEFAULT_GROUPS at line 117](https://github.com/pranwsh/ecs170-project/blob/281a7ed426cf7166d8f26a6dc62f661e5516c29b/ECS170_Fall_2026_Source_Code_Template/script/stage_2_script/run_experiments.py#L117).

**Issue:** The no-argument command is described as running all groups, but defaults to only baseline, optimizer, learning-rate, and loss. Another comment promises a third centered-input experiment, while only raw and normalized runs exist.

**Proposed solution:** Correct the default-command description, show explicit `--groups` examples for the other groups, and remove the stale centered-input description. Do not add another experiment just to match the comment. Make the same corrections wherever those instructions are repeated in the results README.

**Verify the fix:** Compare the instructions with `DEFAULT_GROUPS` and the existing `--dry-run` output. The documented preprocessing group should list exactly its two implemented variants.

## What should remain unchanged

The basic loader/model/evaluator integration, held-out validation strategy, and existing meaningful tests do not need a rewrite. Person 2's artificial-data test checks learnability; reusing that artificial data within that particular test is not a claim about unseen-digit accuracy and is not itself a defect. Small duplicated helpers currently agree and are optional cleanup, not submission blockers.