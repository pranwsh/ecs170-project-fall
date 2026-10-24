"""Convergence curves and comparison tables/figures for the Stage 2 experiments.

Two kinds of plots are needed for the report:

1. Single-run convergence - train (and validation) loss and accuracy per epoch,
   to show that training actually converged and to spot overfitting.
2. Multi-run comparison - several experiments on the same axes, plus bar charts
   and a table of the final metrics, so optimizer / loss / learning-rate
   choices can be compared at a glance.

Method_MLP fills `model.history` with per-epoch lists, so every function here
only needs that dict plus the metrics dict produced by Evaluate_Metrics.
"""

import csv
import json
import pickle
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

# consistent colours/markers so figures from different runs look like one set
SERIES_STYLE = [
    {'color': '#1f77b4', 'marker': 'o'},
    {'color': '#d62728', 'marker': 's'},
    {'color': '#2ca02c', 'marker': '^'},
    {'color': '#9467bd', 'marker': 'D'},
    {'color': '#ff7f0e', 'marker': 'v'},
    {'color': '#17becf', 'marker': 'P'},
    {'color': '#8c564b', 'marker': 'X'},
    {'color': '#e377c2', 'marker': '*'},
]


def _style(index):
    return SERIES_STYLE[index % len(SERIES_STYLE)]


def _epochs(history, key='train_loss'):
    return np.arange(1, len(history[key]) + 1)


def save_convergence_plots(history, path, title='', dpi=150):
    """Loss and accuracy per epoch; validation curves are included when present.

    history keys used: train_loss, train_acc, val_loss, val_acc (the val_* keys
    are empty lists unless the setting passed a 'val' split to Method_MLP).
    """
    if not history.get('train_loss'):
        raise ValueError('history has no train_loss; was the model trained?')

    epochs = _epochs(history)
    fig, (loss_ax, acc_ax) = plt.subplots(1, 2, figsize=(11, 4.2))

    loss_ax.plot(epochs, history['train_loss'], marker='o', markersize=3, label='train')
    if history.get('val_loss'):
        loss_ax.plot(epochs, history['val_loss'], marker='o', markersize=3, label='validation')
    loss_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    loss_ax.set_xlabel('Epoch')
    loss_ax.set_ylabel('Loss')
    loss_ax.set_title('Loss per epoch')
    loss_ax.grid(alpha=0.3)
    loss_ax.legend()

    if history.get('train_acc'):
        acc_ax.plot(epochs, history['train_acc'], marker='o', markersize=3, label='train')
    if history.get('val_acc'):
        acc_ax.plot(epochs, history['val_acc'], marker='o', markersize=3, label='validation')
    acc_ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    acc_ax.set_xlabel('Epoch')
    acc_ax.set_ylabel('Accuracy')
    acc_ax.set_title('Accuracy per epoch')
    acc_ax.grid(alpha=0.3)
    acc_ax.legend()

    if title:
        fig.suptitle(title)

    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def save_multi_run_convergence(runs, path, metric='val_acc', title=None, dpi=150):
    """Overlay one metric (e.g. 'val_acc' or 'train_loss') for several runs.

    runs: iterable of (label, history) pairs. Runs whose history lacks the
    metric are skipped, which keeps this usable when some runs had no
    validation split.
    """
    runs = [(label, history) for label, history in runs if history.get(metric)]
    if not runs:
        raise ValueError(f'no run recorded {metric!r}')

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    for index, (label, history) in enumerate(runs):
        style = _style(index)
        ax.plot(_epochs(history, metric), history[metric],
                markersize=3, linewidth=1.6, label=label, **style)

    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel('Epoch')
    ax.set_ylabel({'val_acc': 'Validation accuracy', 'train_acc': 'Training accuracy',
                   'val_loss': 'Validation loss', 'train_loss': 'Training loss'
                   }.get(metric, metric))
    ax.set_title(title or f'{metric} per epoch')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def save_metric_bars(results, path, metrics=('accuracy', 'f1_macro', 'f1_weighted'),
                     title='Final metrics by run', dpi=150):
    """Grouped bar chart comparing the final metrics of several runs.

    results: iterable of (label, scores dict) pairs from Evaluate_Metrics.
    """
    results = list(results)
    if not results:
        raise ValueError('no results to plot')

    labels = [label for label, _ in results]
    positions = np.arange(len(labels))
    width = 0.8 / len(metrics)

    fig, ax = plt.subplots(figsize=(max(7.0, 1.1 * len(labels)), 4.6))
    for index, metric in enumerate(metrics):
        values = [scores.get(metric, np.nan) for _, scores in results]
        offset = (index - (len(metrics) - 1) / 2) * width
        ax.bar(positions + offset, values, width, label=metric,
               color=_style(index)['color'])

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=30, ha='right', fontsize=8)
    # zoom in on the interesting range; accuracy differences are small
    finite = [v for _, scores in results for v in
              (scores.get(m) for m in metrics) if v is not None and np.isfinite(v)]
    if finite and min(finite) > 0.5:
        ax.set_ylim(max(0.0, min(finite) - 0.05), 1.0)
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def save_per_class_plot(per_class, path, title='Per-digit metrics', dpi=150):
    """Bar chart of per-digit precision / recall / F1 from per_class_metrics()."""
    labels = per_class['labels']
    positions = np.arange(len(labels))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for index, key in enumerate(('precision', 'recall', 'f1')):
        ax.bar(positions + (index - 1) * width, per_class[key], width, label=key,
               color=_style(index)['color'])

    ax.set_xticks(positions)
    ax.set_xticklabels([str(label) for label in labels])
    ax.set_xlabel('Digit')
    ax.set_ylabel('Score')
    ax.set_ylim(max(0.0, min(min(per_class[k]) for k in ('precision', 'recall', 'f1')) - 0.05), 1.0)
    ax.set_title(title)
    ax.grid(alpha=0.3, axis='y')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
    return Path(path)


def write_comparison_table(results, path, columns=None, title='Experiment comparison'):
    """Write a markdown + CSV table of final metrics, one row per run.

    results: iterable of dicts with at least 'run' and the metric keys, i.e.
    {'run': 'adam lr=1e-3', 'accuracy': 0.97, ...}. Returns the markdown text.
    """
    results = list(results)
    if not results:
        raise ValueError('no results to tabulate')

    if columns is None:
        columns = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro']
        columns = [c for c in columns if any(c in row for row in results)]

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # CSV next to the markdown, so the numbers can be pasted into a report or
    # re-plotted later without re-running the experiments
    with path.with_suffix('.csv').open('w', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['run'] + columns)
        for row in results:
            writer.writerow([row['run']] + [f'{row[c]:.6f}' if c in row else ''
                                            for c in columns])

    header = '| run | ' + ' | '.join(columns) + ' |'
    divider = '|---' * (len(columns) + 1) + '|'
    lines = [f'### {title}', '', header, divider]
    for row in results:
        cells = [f'{row[c]:.4f}' if c in row else '-' for c in columns]
        lines.append(f'| {row["run"]} | ' + ' | '.join(cells) + ' |')

    text = '\n'.join(lines) + '\n'
    path.write_text(text)
    return text


def load_run_history(run_dir):
    """Read a saved run's config/metrics and per-epoch history."""

    run_dir = Path(run_dir)
    config = json.loads((run_dir / 'config.json').read_text())
    metrics = json.loads((run_dir / 'metrics.json').read_text())

    history = {}
    # history.json is written next to every run, so plots can be rebuilt without
    # the (large, git-ignored) pickled predictions
    history_path = run_dir / 'history.json'
    if history_path.is_file():
        history = json.loads(history_path.read_text())
    else:
        for candidate in sorted(run_dir.glob('prediction_result*')):
            with candidate.open('rb') as handle:
                payload = pickle.load(handle)
            if isinstance(payload, dict) and payload.get('history'):
                history = payload['history']
                break

    return {'config': config, 'metrics': metrics, 'history': history}


def collect_results(result_root, group=None):
    """Load every saved run under result_root (optionally one group folder).

    Returns a list of dicts ready for write_comparison_table / save_metric_bars.
    """
    result_root = Path(result_root)
    root = result_root / group if group else result_root

    rows = []
    for config_path in sorted(root.glob('**/config.json')):
        run_dir = config_path.parent
        try:
            loaded = load_run_history(run_dir)
        except (OSError, ValueError, pickle.UnpicklingError):
            continue

        row = {'run': str(run_dir.relative_to(result_root)), **loaded['metrics']}
        row['hidden_dims'] = ','.join(str(h) for h in loaded['config'].get('hidden_dims', []))
        for key in ('optimizer', 'learning_rate', 'batch_size', 'max_epoch', 'dropout',
                    'loss', 'activation', 'normalize'):
            if key in loaded['config']:
                row[key] = loaded['config'][key]
        rows.append(row)

    return rows


