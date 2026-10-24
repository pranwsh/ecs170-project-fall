"""Confusion matrix and misclassification analysis for the Stage 2 test predictions."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


DIGITS = list(range(10))


def compute_confusion_matrix(true_y, pred_y):
    """Rows are the true digit, columns are the predicted digit."""
    return confusion_matrix(true_y, pred_y, labels=DIGITS)


def per_class_accuracy(cm):
    """Fraction of each digit that was classified correctly (recall per class)."""
    totals = cm.sum(axis=1)
    return np.divide(np.diag(cm), totals, out=np.zeros(len(totals)), where=totals > 0)


def top_confusions(cm, k=10):
    """Most frequent (true digit, predicted digit, count) mistakes, largest first."""
    mistakes = []
    for t in DIGITS:
        for p in DIGITS:
            if t != p and cm[t, p] > 0:
                mistakes.append((t, p, int(cm[t, p])))

    mistakes.sort(key=lambda m: m[2], reverse=True)
    return mistakes[:k]


def save_confusion_matrix(cm, path, title='Confusion matrix (test set)'):
    # color by row percentage so mistakes on smaller classes still show up,
    # but print the raw counts in each cell
    row_pct = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(row_pct, cmap='Blues', vmin=0, vmax=1)

    for t in DIGITS:
        for p in DIGITS:
            if row_pct[t, p] > 0.5:
                color = 'white'
            else:
                color = 'black'
            ax.text(p, t, str(cm[t, p]), ha='center', va='center', fontsize=8, color=color)

    ax.set_xticks(DIGITS)
    ax.set_yticks(DIGITS)
    ax.set_xlabel('Predicted digit')
    ax.set_ylabel('True digit')
    ax.set_title(title)
    fig.colorbar(image, ax=ax, label='Fraction of true digit')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def save_misclassified_examples(X, true_y, pred_y, path, count=25, seed=2):
    """Show a sample of test images the model got wrong."""
    wrong = np.flatnonzero(true_y != pred_y)
    if len(wrong) == 0:
        return 0
    rng = np.random.default_rng(seed)
    chosen = rng.choice(wrong, size=min(count, len(wrong)), replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(8, 8.5))
    for ax in axes.flatten():
        ax.axis('off')

    for ax, index in zip(axes.flatten(), chosen):
        ax.imshow(np.asarray(X[index]).reshape(28, 28), cmap='gray')
        ax.set_title(f'true {true_y[index]}, pred {pred_y[index]}', fontsize=9)

    fig.suptitle(f'Misclassified test images ({len(wrong)} total errors)')
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return len(wrong)


def write_report(cm, path, k=10):
    """Plain-text summary used for the error analysis section of the report."""
    accuracy = per_class_accuracy(cm)
    total_errors = int(cm.sum() - np.trace(cm))

    lines = [
        f'Total test samples: {int(cm.sum())}',
        f'Total errors: {total_errors}',
        '',
        'Per-digit accuracy (worst first):'
    ]

    for digit in np.argsort(accuracy):
        wrong_count = int(cm[digit].sum() - cm[digit, digit])
        total_count = int(cm[digit].sum())
        lines.append(f'  {digit}: {accuracy[digit]:.4f}  ({wrong_count} wrong of {total_count})')

    lines += ['', f'Top {k} confusions (true -> predicted):']

    for t, p, n in top_confusions(cm, k):
        if total_errors > 0:
            share = n / total_errors
        else:
            share = 0
        lines.append(f'  {t} -> {p}: {n}  ({share:.1%} of all errors)')

    Path(path).write_text('\n'.join(lines) + '\n')
    return '\n'.join(lines)
