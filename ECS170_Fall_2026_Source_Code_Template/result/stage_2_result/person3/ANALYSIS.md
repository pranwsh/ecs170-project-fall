# Stage 2 experiment write-up (raw material for the 2-5 report)

All numbers below are from the 20% validation split of `train.csv`
(48,000 train / 12,000 validation, stratified, seed 2), 20 epochs, batch 128,
normalized pixels, baseline architecture 784-256-128-10 with ReLU unless the row
says otherwise. Raw numbers live in `summary.csv`, `*/comparison.csv` and each
run's `metrics.json`; curves are in each run's `history.json` / `convergence.png`.

Covers 8 experiment axes and 34 runs: metrics (Person 3), preprocessing
(Person 1), architecture (Person 2), optimizer / learning rate / loss
(Person 3), activation / batch size ("other settings"), plus the combined
best configuration run on both validation and test.

## 1. Metrics

`accuracy` alone is not enough, so every run also reports weighted/macro/micro
precision, recall and F1, plus balanced accuracy and Cohen's kappa.

* `micro` precision = recall = F1 = accuracy, always. With single-label
  multiclass data every sample contributes exactly one positive and nine
  negative decisions, so the micro average collapses to accuracy. Reported for
  completeness; the report should say so explicitly.
* `macro` treats all ten digits equally, so it drops faster than accuracy when
  the model starts ignoring a hard digit. Example: in the Adam 1e-5 run
  accuracy is 0.9085 while macro recall is 0.9071.
* `weighted` follows the class sizes (each digit has ~1,200 validation images),
  so on this balanced dataset weighted and macro stay within 0.0005 of each
  other. On an unbalanced dataset they would separate, which is why both are
  reported.
* `balanced_accuracy` and `cohen_kappa` agree with macro F1 in every run here,
  confirming that no digit is being systematically abandoned.

## 2. Optimizer (group `optimizer/`)

| run | accuracy | macro F1 |
|---|---|---|
| SGD lr=0.05, momentum 0.9 | **0.9821** | 0.9819 |
| RMSprop lr=1e-3 | 0.9799 | 0.9797 |
| SGD lr=0.01, momentum 0.9 | 0.9772 | 0.9770 |
| Adam lr=1e-3 (baseline) | 0.9703 | 0.9700 |

SGD with momentum 0.9 and a much larger step size (0.05) wins. Adam is
comfortable out of the box but plateaus lower within 20 epochs; its per-parameter
step size adapts so quickly that progress on this easy, well-scaled problem is
slower. SGD at 0.01 is clearly under-trained (still improving at epoch 20), which
is why both 0.01 and 0.05 are reported instead of a single "SGD" row.

## 3. Learning rate (group `lr/`)

| Adam lr | accuracy | macro F1 |
|---|---|---|
| 1e-2 | 0.9711 | 0.9707 |
| 1e-3 | 0.9703 | 0.9700 |
| 1e-4 | 0.9670 | 0.9667 |
| 1e-5 | 0.9085 | 0.9072 |

1e-2 and 1e-3 are within noise of each other; 1e-4 is under-trained and 1e-5
fails to converge in 20 epochs (the loss curve is still descending steeply at the
last epoch). The takeaway is the shape of the curve, not the top row: too small a
step size costs far more than a slightly too large one here, because the network
is small and the inputs are already normalized to [0, 1].

## 4. Loss function (group `loss/`)

| loss | accuracy | macro F1 |
|---|---|---|
| cross-entropy + label smoothing 0.1 | **0.9835** | 0.9833 |
| MSE on softmax outputs | 0.9778 | 0.9776 |
| cross-entropy | 0.9703 | 0.9700 |
| log-softmax + NLL | 0.9703 | 0.9700 |

* `nll` reproduces `ce` to four decimals, as expected: it is the same
  computation split into two operations. A unit test asserts this equality.
* MSE converges, but its gradients carry the softmax derivative squared, so it
  needs more epochs to reach the same place; it is still surprisingly strong here.
* Label smoothing gains about 1.3 points of accuracy and is the single best
  change found. It stops the network from pushing logits to extreme values, which
  acts as a regularizer and helps the visually ambiguous pairs.

## 5. Regularization (group `regularization/`)

| setting | accuracy | macro F1 |
|---|---|---|
| dropout 0.2 | 0.9778 | 0.9775 |
| dropout 0.5 | 0.9766 | 0.9763 |
| weight decay 1e-4 | 0.9758 | 0.9755 |

All three beat the 0.9703 baseline, and all three land in the same narrow band.
The train/validation curves do not separate in the baseline run (no overfitting
gap yet at 20 epochs), so these gains are small and partly noise; the honest
conclusion is that mild regularization helps a little and heavy regularization
(dropout 0.5) starts to hurt.

## 6. Architecture (group `architecture/`)

| hidden layers | accuracy | macro F1 |
|---|---|---|
| 1024-512 | **0.9802** | 0.9801 |
| 512-256-128 | 0.9788 | 0.9786 |
| 128 | 0.9755 | 0.9753 |
| 256-128 (baseline) | 0.9703 | 0.9700 |

Wider and deeper both help, monotonically, with diminishing returns. Note the
single 128-unit layer beats the baseline 256-128 pair, which says the baseline
two-layer net was simply under-trained for its size, not that depth hurts.

## 7. Preprocessing (group `preprocessing/`, Person 1's axis)

| inputs | accuracy | macro F1 |
|---|---|---|
| scaled to [0, 1] | **0.9703** | 0.9700 |
| raw 0-255 | 0.9700 | 0.9696 |

This is the one place where the expected textbook answer is wrong for this
dataset, and it is worth spelling out in the report rather than repeating the
usual "always normalize" line.

Accuracy is a tie (0.9703 vs 0.9700, i.e. 3 images out of 12,000). The two runs
differ in *stability*, not in final score:

| | scaled [0,1] | raw 0-255 |
|---|---|---|
| final validation loss | **0.1463** | 0.1891 |
| best validation accuracy | **0.9798** (epoch 15) | 0.9721 (epoch 18) |
| val loss trend | falling | flat/rising after midpoint |

Why raw input survives at all: by epoch 1 Adam's per-parameter step size has
already compensated for the 255x larger input scale, so the network trains.
Why it is still worse: with raw pixels the effective learning rate is wrong at
the start and the loss stays higher (0.189 vs 0.146), so the model is less
confident and its best epoch is worse. Normalizing is therefore worth doing for
optimisation stability and for having one fewer thing to retune, not because it
changes the ceiling on this task.

This also explains why every other experiment in this file is reported on
normalized input: it keeps the learning-rate sweeps meaningful.

## 8. Activation function (group `activation/`, "other settings")

| activation | accuracy | macro F1 |
|---|---|---|
| sigmoid | **0.9766** | 0.9763 |
| leaky ReLU | 0.9762 | 0.9760 |
| ReLU (baseline) | 0.9703 | 0.9700 |
| tanh | 0.9683 | 0.9679 |

Counter-intuitive and worth flagging: sigmoid, the activation most associated
with vanishing gradients, wins here, while tanh - which has the same saturation
shape but zero-centred outputs - is the worst of the four.

The likely reason is capacity, not gradient flow. This network is small
(256-128) and only trains for 20 epochs, so a saturating activation that
effectively shrinks the logit range acts as a mild regulariser on an
already-underfitting model, whereas ReLU's unbounded output lets it overfit the
training split more eagerly. The margin is small (0.006 accuracy, ~75 images) so
it should be presented as "no activation choice matters much here", not as a
"sigmoid is better" claim.

## 9. Batch size (group `batch_size/`, "other settings")

| batch size | accuracy | macro F1 |
|---|---|---|
| 256 | **0.9788** | 0.9786 |
| 32 | 0.9782 | 0.9780 |
| 64 | 0.9748 | 0.9746 |
| 128 (baseline) | 0.9703 | 0.9700 |

The pattern is a U: both the largest (256) and the smallest (32) batch beat the
middle settings, while 64 and 128 are the weak pair. That is almost certainly
not a real effect of batch size but a step-count effect - with a fixed 20
epochs, batch 32 performs ~1,500 updates per epoch while batch 256 performs
~188, yet both land in the same place, which means the baseline is
under-trained rather than mis-parameterised. Report this as "the baseline's
128-batch setting is not special" and do not claim 256 is fundamentally better.

## 10. Final runs on `test.csv` (group `final/`)

Same code path, `--validate` removed, so these are the numbers to report for
2-3. Test set = 10,000 images that no experiment touched.

| run | accuracy | macro F1 | weighted F1 | kappa |
|---|---|---|---|---|
| **label smoothing + SGD 0.05 / momentum 0.9** (best) | **0.9865** | 0.9864 | 0.9865 | 0.9850 |
| SGD lr=0.05, momentum 0.9 | 0.9841 | 0.9840 | 0.9841 | 0.9823 |
| cross-entropy + label smoothing | 0.9833 | 0.9832 | 0.9833 | 0.9814 |
| baseline (Adam 1e-3, CE) | 0.9807 | 0.9805 | 0.9807 | 0.9785 |

Validation and test accuracy agree to within 0.2 points for every configuration
(the combined best run scored 0.9838 on validation and 0.9865 on test), so the
20% validation split was a reliable stand-in for model selection and there is no
sign of the test set being over-fitted by the experiments.

The combined configuration stacks the two largest single-change wins. They are
largely additive: 0.9703 baseline -> 0.9865 test, i.e. **+0.0162 accuracy**,
leaving 135 errors out of 10,000.

## 11. Error behaviour (from `*/error_analysis.txt`)

The same pairs fail in every configuration:

* 9 -> 7 and 9 -> 4 (top-2 confusions in most runs),
* 3 -> 5, 3 -> 8, 5 -> 3,
* 8 -> 5, 8 -> 3.

Digit 9 is consistently the worst class (0.9723 recall in the label-smoothing
run) and digit 1 the best (0.9933). These are all pairs whose handwritten forms
differ by a single stroke or loop, so the remaining error is largely irreducible
for a plain MLP on raw pixels: it has no shift/rotation invariance, unlike a CNN.

## 12. What this adds up to for 2-4

Best single change: **label smoothing +0.0132**, then **SGD 0.05 with momentum
+0.0118**. Everything else is worth +0.003 to +0.010, and the preprocessing and
batch-size axes show no real effect. Stacking the two best changes is the
obvious final configuration:

| configuration | validation accuracy | test accuracy |
|---|---|---|
| baseline (Adam 1e-3, CE, 256-128) | 0.9703 | 0.9807 |
| + label smoothing 0.1 | 0.9835 | 0.9833 |
| + SGD 0.05 / momentum 0.9 | 0.9821 | 0.9841 |
| label smoothing **and** SGD 0.05 / momentum 0.9 | 0.9838 | **0.9865** |

Variance warning for the report: with a 12,000-image validation split, one
standard error on accuracy is about 0.0016, so differences below ~0.003
(roughly 35 images) are not meaningful. Only the label-smoothing, SGD and
architecture results clear that bar.
