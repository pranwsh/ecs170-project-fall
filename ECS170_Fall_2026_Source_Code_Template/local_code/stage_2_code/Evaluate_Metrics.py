"""Multiclass evaluation metrics for Stage 2 (accuracy, precision, recall, F1).

Stage 2 is a 10-class digit classification task, so the binary versions of
precision / recall / F1 cannot be used. We report the 'weighted', 'macro' and
'micro' averaged versions instead:

  weighted - average over classes weighted by how many samples each class has.
             Closest to "overall performance" when classes are unbalanced.
  macro    - plain average over classes, so every digit counts equally no
             matter how often it appears. Punishes bad minority-class scores.
  micro    - average over all (sample, class) decisions. For single-label
             multiclass problems micro precision = micro recall = micro F1 =
             accuracy, so those three numbers will always match; they are kept
             because the assignment asks for all of them and the report should
             explain why they coincide.

The evaluate() method returns a flat dict of float -> float so the Stage 2
setting can print it, dump it to metrics.json, and build comparison tables.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

from local_code.base_class.evaluate import evaluate

# averages we report for precision / recall / F1
AVERAGES = ('weighted', 'macro', 'micro')


def to_numpy(values):
    """Accept a torch tensor, a list or an array and return a NumPy array.

    Method_MLP returns predictions as a CPU torch tensor, while the true labels
    stay a NumPy array, so every metric function normalises its inputs first.
    """
    if hasattr(values, 'detach'):
        values = values.detach().cpu().numpy()
    return np.asarray(values)


def compute_metrics(true_y, pred_y, averages=AVERAGES, labels=None):
    """Return accuracy plus weighted/macro/micro precision, recall and F1.

    labels defaults to every class that appears in either array, sorted, so the
    metric definitions stay stable when a run happens to miss a digit.
    """
    true_y = to_numpy(true_y).ravel()
    pred_y = to_numpy(pred_y).ravel()

    if true_y.shape != pred_y.shape:
        raise ValueError(f'true_y {true_y.shape} and pred_y {pred_y.shape} must have the same shape')
    if len(true_y) == 0:
        raise ValueError('cannot evaluate an empty prediction set')

    if labels is None:
        labels = np.unique(np.concatenate([true_y, pred_y]))

    scores = {
        'accuracy': float(accuracy_score(true_y, pred_y)),
        # accuracy averaged over classes instead of samples; a low value here
        # while accuracy is high means some digit is being ignored
        'balanced_accuracy': float(balanced_accuracy_score(true_y, pred_y)),
    }

    for average in averages:
        scores[f'precision_{average}'] = float(precision_score(
            true_y, pred_y, labels=labels, average=average, zero_division=0))
        scores[f'recall_{average}'] = float(recall_score(
            true_y, pred_y, labels=labels, average=average, zero_division=0))
        scores[f'f1_{average}'] = float(f1_score(
            true_y, pred_y, labels=labels, average=average, zero_division=0))

    # agreement corrected for chance; useful in the report because it does not
    # reward a model that just guesses the most common digit
    scores['cohen_kappa'] = float(cohen_kappa_score(true_y, pred_y, labels=labels))
    return scores


def per_class_metrics(true_y, pred_y, labels=None):
    """Per-digit precision / recall / F1 / support, for the report tables."""
    true_y = to_numpy(true_y).ravel()
    pred_y = to_numpy(pred_y).ravel()

    if labels is None:
        labels = np.unique(np.concatenate([true_y, pred_y]))

    precision = precision_score(true_y, pred_y, labels=labels, average=None, zero_division=0)
    recall = recall_score(true_y, pred_y, labels=labels, average=None, zero_division=0)
    f1 = f1_score(true_y, pred_y, labels=labels, average=None, zero_division=0)
    support = np.array([(true_y == label).sum() for label in labels])

    return {
        'labels': [int(label) for label in labels],
        'precision': [float(v) for v in precision],
        'recall': [float(v) for v in recall],
        'f1': [float(v) for v in f1],
        'support': [int(v) for v in support],
    }


# the metrics we care about most, in the order they should be reported
HEADLINE_METRICS = ('accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro')


def format_metrics(scores, metrics=HEADLINE_METRICS):
    """One-line summary used in console output and report tables."""
    parts = [f'{name}: {scores[name]:.4f}' for name in metrics if name in scores]
    return ', '.join(parts)


class Evaluate_Metrics(evaluate):
    """Stage 2 evaluation module: accuracy + multiclass precision/recall/F1."""

    data = None
    averages = AVERAGES

    def __init__(self, eName=None, eDescription=None, averages=AVERAGES):
        super().__init__(eName, eDescription)
        self.averages = tuple(averages)

    def evaluate(self):
        """self.data must hold {'true_y': ..., 'pred_y': ...}; returns a dict."""
        print('evaluating performance (accuracy, precision, recall, f1)...')
        return compute_metrics(self.data['true_y'], self.data['pred_y'], averages=self.averages)
