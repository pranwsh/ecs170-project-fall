# Stage 2 experiments - index of results (Person 3's area)

Everything here is produced by `script/stage_2_script/run_experiments.py` (which
calls the same pipeline as `script/stage_2_script/script_mlp.py`). The findings
themselves are written up in [`ANALYSIS.md`](ANALYSIS.md); this file is just the
map of the code and the result folders.

## Result folders

| folder | axis | person |
|---|---|---|
| `preprocessing/` | normalized [0,1] vs raw 0-255 | 1 |
| `architecture/` | hidden layers and widths | 2 |
| `optimizer/` | Adam / SGD / RMSprop | 3 |
| `lr/` | learning-rate sweep | 3 |
| `loss/` | CE / NLL / MSE / label smoothing | 3 |
| `regularization/` | dropout, weight decay | 3 |
| `activation/` | ReLU / tanh / sigmoid / leaky ReLU | 3 ("other settings") |
| `batch_size/` | 32 / 64 / 128 / 256 | 3 ("other settings") |
| `final/` | baseline + two single wins on **test.csv** | 4 |
| `best/` | combined best config on validation **and** test | 3 |

## Code

| File | Purpose |
|---|---|
| `local_code/stage_2_code/Evaluate_Metrics.py` | `Evaluate_Metrics` module (accuracy + weighted/macro/micro precision, recall, F1, balanced accuracy, Cohen's kappa) and the `compute_metrics` / `per_class_metrics` helpers |
| `local_code/stage_2_code/Convergence_Plots.py` | single-run convergence curves, multi-run overlays, metric bar charts, per-digit plots, markdown/CSV comparison tables, `collect_results` to re-read saved runs |
| `local_code/stage_2_code/Loss_Functions.py` | `ce`, `mse` (softmax + one-hot MSE), `nll` (log-softmax + NLL) and `ce_ls` (label smoothing) behind one `build_loss(name)` |
| `script/stage_2_script/script_mlp.py` | one run; new flags `--loss --activation --weight-decay --momentum --batch-size --group --quiet` |
| `script/stage_2_script/run_experiments.py` | runs whole experiment groups and writes the comparison tables/figures |
| `script/stage_2_script/summarize_results.py` | one sorted table across every saved run |
| `tests/test_stage2_evaluate.py` | unit tests for metrics, losses, plots and artifact collection |

## Running

```bash
# from ECS170_Fall_2026_Source_Code_Template/
python -m unittest discover -s tests -v              # all Stage 2 tests
python -m script.stage_2_script.run_experiments --dry-run   # list the plan
python -m script.stage_2_script.run_experiments      # baseline, optimizer, lr, loss
python -m script.stage_2_script.run_experiments --groups architecture regularization
python -m script.stage_2_script.run_experiments --groups preprocessing activation batch_size
python -m script.stage_2_script.summarize_results    # one table across every run

# single run; omit --validate to evaluate on test.csv
python -m script.stage_2_script.script_mlp --validate --name tuned \
    --loss ce_ls --optimizer sgd --lr 0.05 --momentum 0.9
```

Groups: `baseline`, `optimizer`, `lr`, `loss`, `regularization`, `architecture`,
`preprocessing`, `activation`, `batch_size`. `DEFAULT_GROUPS` runs the middle four,
`preprocessing`, `activation`, `batch_size`. With no `--groups`,
`DEFAULT_GROUPS` runs just `baseline optimizer lr loss`, so pass `--groups`
explicitly to sweep anything else.
Every run writes `config.json`, `metrics.json`, `history.json`, `convergence.png`,
`confusion_matrix.{csv,png}`, `misclassified_examples.png`,
`per_class_metrics.{json,png}` and `error_analysis.txt`. Each group also gets
`comparison.{md,csv,json}` plus overlay plots, and `summary.{md,csv}` covers all
runs.

## Method notes (for the report)

* All experiment runs use a stratified 80/20 split of `train.csv` (seed 2) so
  `test.csv` stays untouched until the final runs (`final/`, `best/`).
* One setting changes per run, everything else stays at the shared baseline
  (784-256-128-10, ReLU, Adam 1e-3, batch 128, 20 epochs, normalized inputs).
* `micro` precision/recall/F1 equal accuracy for single-label multiclass data;
  they are reported for completeness and this should be stated in the report.
* `macro` weights every digit equally and is therefore the metric that reveals a
  model that gives up on a hard digit; `weighted` follows the class sizes.
* **Variance warning:** one standard error on the 12,000-image validation split is
  about 0.0016, so accuracy differences below ~0.003 (roughly 35 images) are
  noise. Only the label-smoothing, SGD, architecture and combined-config results
  clear that bar.
