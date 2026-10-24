"""Run Person 3's hyperparameter experiments and build the comparison outputs.

Every run is a normal Stage 2 pipeline run (same code path as script_mlp.py) with
one setting changed, evaluated on the 20% validation split of train.csv so that
test.csv stays clean for the final model. After each group finishes, the runs are
aggregated into:

    comparison.md / comparison.csv   table of final metrics, one row per run
    comparison_accuracy.png          bar chart of accuracy / macro F1 / weighted F1
    convergence_val_acc.png          validation accuracy curves overlaid
    convergence_val_loss.png         validation loss curves overlaid

Usage (from the template root):

    python -m script.stage_2_script.run_experiments                  # all groups
    python -m script.stage_2_script.run_experiments --groups optimizer lr
    python -m script.stage_2_script.run_experiments --groups loss --epochs 10
    python -m script.stage_2_script.run_experiments --groups optimizer --dry-run

Results land in result/stage_2_result/person3/<group>/<run>/.
"""

import argparse
import json
import sys
import time
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

import matplotlib
matplotlib.use('Agg')

from local_code.stage_2_code import Convergence_Plots
from script.stage_2_script.script_mlp import BASELINE, RESULT_ROOT, run_experiment

GROUP_ROOT = 'person3'

# Each group is a list of (run name, overrides applied on top of BASELINE).
# Keep one setting different per run so the comparison stays interpretable.
EXPERIMENT_GROUPS = {
    # reference point: everyone's shared baseline, on the validation split
    'baseline': [
        ('baseline', {}),
    ],

    # 2-4: different optimizers, each at a sensible learning rate for that
    # optimizer (SGD needs a much larger lr than Adam to move at all)
    'optimizer': [
        ('adam_1e-3', {'optimizer': 'adam', 'learning_rate': 1e-3}),
        ('sgd_1e-2_m0.9', {'optimizer': 'sgd', 'learning_rate': 1e-2, 'momentum': 0.9}),
        ('sgd_5e-2_m0.9', {'optimizer': 'sgd', 'learning_rate': 5e-2, 'momentum': 0.9}),
        ('rmsprop_1e-3', {'optimizer': 'rmsprop', 'learning_rate': 1e-3}),
    ],

    # 2-4: learning rate sweep for the default optimizer (Adam)
    'lr': [
        ('adam_1e-2', {'learning_rate': 1e-2}),
        ('adam_1e-3', {'learning_rate': 1e-3}),
        ('adam_1e-4', {'learning_rate': 1e-4}),
        ('adam_1e-5', {'learning_rate': 1e-5}),
    ],

    # 2-4: different loss functions
    'loss': [
        ('ce', {'loss': 'ce'}),
        ('nll', {'loss': 'nll'}),
        ('mse', {'loss': 'mse'}),
        ('ce_labelsmooth', {'loss': 'ce_ls'}),
    ],

    # "other settings": regularisation that fights overfitting
    'regularization': [
        ('dropout_0.2', {'dropout': 0.2}),
        ('dropout_0.5', {'dropout': 0.5}),
        ('weight_decay_1e-4', {'weight_decay': 1e-4}),
    ],

    # Person 2's axis, repeated here only so the report can compare against the
    # same baseline pipeline (their branch runs the architecture comparison)
    'architecture': [
        ('single_128', {'hidden_dims': [128]}),
        ('two_256_128', {'hidden_dims': [256, 128]}),
        ('three_512_256_128', {'hidden_dims': [512, 256, 128]}),
        ('wide_1024_512', {'hidden_dims': [1024, 512]}),
    ],

    # 2-1: Person 1's axis - does preprocessing (0-1 scaling) matter? The third
    # run keeps the 0-255 range but centres it, which separates "small values"
    # from "features on a comparable scale".
    'preprocessing': [
        ('normalized_0_1', {'normalize': True}),
        ('raw_0_255', {'normalize': False}),
    ],

    # 2-4 "other settings": activation function. tanh/sigmoid saturate, so they
    # are expected to converge slower than ReLU; leaky_relu should match ReLU.
    'activation': [
        ('relu', {'activation': 'relu'}),
        ('tanh', {'activation': 'tanh'}),
        ('sigmoid', {'activation': 'sigmoid'}),
        ('leaky_relu', {'activation': 'leaky_relu'}),
    ],

    # 2-4 "other settings": mini-batch size. Small batches add gradient noise
    # (sometimes a regulariser), large ones use the hardware better.
    'batch_size': [
        ('batch_32', {'batch_size': 32}),
        ('batch_64', {'batch_size': 64}),
        ('batch_128', {'batch_size': 128}),
        ('batch_256', {'batch_size': 256}),
    ],
}

DEFAULT_GROUPS = ['baseline', 'optimizer', 'lr', 'loss']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Stage 2 hyperparameter experiments')
    parser.add_argument('--groups', nargs='+', choices=sorted(EXPERIMENT_GROUPS),
                        default=DEFAULT_GROUPS, help='which experiment groups to run')
    parser.add_argument('--epochs', type=int, default=BASELINE['max_epoch'],
                        help='epochs per run (default: baseline 20)')
    parser.add_argument('--seed', type=int, default=BASELINE['seed'])
    parser.add_argument('--device', default=BASELINE['device'])
    parser.add_argument('--on-test', action='store_true',
                        help='evaluate on test.csv instead of the validation split (final runs only)')
    parser.add_argument('--force', action='store_true',
                        help='re-run experiments whose folder already exists')
    parser.add_argument('--dry-run', action='store_true', help='print the plan and exit')
    parser.add_argument('--quiet', action='store_true')
    return parser.parse_args(argv)


def build_config(overrides, epochs, seed, device, on_test):
    config = dict(BASELINE)
    config.update(overrides)
    config['max_epoch'] = epochs
    config['seed'] = seed
    config['device'] = device
    config['evaluated_on'] = 'test.csv' if on_test else 'validation (20% of train.csv)'
    return config


def summarize_group(group, results, group_dir):
    """Write the table and comparison figures for one finished group."""
    table_rows = [{'run': name, **scores} for name, scores, _ in results]
    text = Convergence_Plots.write_comparison_table(
        table_rows, group_dir / 'comparison.md', title=f'{group} comparison (validation split)')

    Convergence_Plots.save_metric_bars(
        [(name, scores) for name, scores, _ in results],
        group_dir / 'comparison_accuracy.png',
        title=f'{group}: final metrics')

    histories = [(name, learned.get('history') or {}) for name, _, learned in results]
    for metric, filename in (('val_acc', 'convergence_val_acc.png'),
                             ('val_loss', 'convergence_val_loss.png'),
                             ('train_loss', 'convergence_train_loss.png')):
        try:
            Convergence_Plots.save_multi_run_convergence(
                histories, group_dir / filename, metric=metric,
                title=f'{group}: {metric} per epoch')
        except ValueError:
            pass  # that metric was not recorded for any run in this group

    (group_dir / 'comparison.json').write_text(json.dumps(table_rows, indent=2) + '\n')
    return text


def main(argv=None):
    args = parse_args(argv)
    plan = []
    for group in args.groups:
        for name, overrides in EXPERIMENT_GROUPS[group]:
            plan.append((group, name, build_config(overrides, args.epochs, args.seed,
                                                   args.device, args.on_test)))

    if args.dry_run:
        print(f'{len(plan)} runs planned:')
        for group, name, config in plan:
            diff = {k: v for k, v in config.items() if BASELINE.get(k) != v}
            print(f'  {group}/{name}: {diff}')
        return

    started = time.time()
    for group in args.groups:
        group_dir = RESULT_ROOT / GROUP_ROOT / group
        group_dir.mkdir(parents=True, exist_ok=True)

        results = []
        for run_name, overrides in EXPERIMENT_GROUPS[group]:
            run_dir = group_dir / run_name
            config = build_config(overrides, args.epochs, args.seed, args.device, args.on_test)

            if (run_dir / 'metrics.json').exists() and not args.force:
                print(f'skipping {group}/{run_name} (already exists, use --force to re-run)')
                scores = json.loads((run_dir / 'metrics.json').read_text())
                learned = Convergence_Plots.load_run_history(run_dir)
                results.append((run_name, scores, {'history': learned['history']}))
                continue

            print(f'\n===== {group}/{run_name} =====')
            scores, learned, run_dir = run_experiment(
                config, validate=not args.on_test, group=f'{GROUP_ROOT}/{group}',
                name=run_name, quiet=args.quiet)
            results.append((run_name, scores, learned))

        if results:
            text = summarize_group(group, results, group_dir)
            print(f'\n--- {group} summary ---\n{text}')

    print(f'\nall experiments finished in {time.time() - started:.0f}s')


if __name__ == '__main__':
    main()
