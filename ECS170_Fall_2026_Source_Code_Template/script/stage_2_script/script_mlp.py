"""Stage 2 MLP pipeline: load data, train, evaluate, and save everything for one run.

Run from the template root:

    python -m script.stage_2_script.script_mlp                  # final run on test.csv
    python -m script.stage_2_script.script_mlp --validate       # 80/20 split of train.csv
    python -m script.stage_2_script.script_mlp --validate --name lr_1e-2 --lr 1e-2
    python -m script.stage_2_script.script_mlp --validate --name deep --hidden 512,256,128
    python -m script.stage_2_script.script_mlp --validate --name sgd --optimizer sgd --lr 0.05
    python -m script.stage_2_script.script_mlp --validate --name mse --loss mse

Each run gets its own folder in result/stage_2_result/<group>/<name>/ holding the
config, metrics, saved predictions, convergence plots, confusion matrix and the
error analysis text, so every experiment is reproducible from its config.json.

Use --validate for every experiment and keep test.csv for the final run only.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

# let the script run from any working directory
TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
from torch import nn

from local_code.stage_2_code import Convergence_Plots, Error_Analysis
from local_code.stage_2_code.Dataset_Loader import Dataset_Loader
from local_code.stage_2_code.Evaluate_Metrics import Evaluate_Metrics, per_class_metrics
from local_code.stage_2_code.Loss_Functions import LOSSES, LOSS_DESCRIPTIONS, build_loss
from local_code.stage_2_code.Method_MLP import Method_MLP
from local_code.stage_2_code.Result_Saver import Result_Saver
from local_code.stage_2_code.Setting_Stage2 import Setting_Stage2

RESULT_ROOT = TEMPLATE_ROOT / 'result' / 'stage_2_result'

OPTIMIZERS = {'adam': torch.optim.Adam, 'sgd': torch.optim.SGD, 'rmsprop': torch.optim.RMSprop}
ACTIVATIONS = {'relu': nn.ReLU, 'tanh': nn.Tanh, 'sigmoid': nn.Sigmoid, 'leaky_relu': nn.LeakyReLU}

# Baseline config everyone uses unless their experiment changes one of these.
# Defaults match Method_MLP's own defaults.
BASELINE = {
    'seed': 2,
    'normalize': True,
    'hidden_dims': [256, 128],
    'activation': 'relu',
    'dropout': 0.0,
    'optimizer': 'adam',
    'learning_rate': 1e-3,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'loss': 'ce',
    'batch_size': 128,
    'max_epoch': 20,
    'device': 'cpu',
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Stage 2 MLP pipeline')
    parser.add_argument('--name', default=None, help='run folder name (default: baseline / baseline_validation)')
    parser.add_argument('--group', default=None,
                        help='sub-folder inside result/stage_2_result (e.g. person3/optimizers)')
    parser.add_argument('--validate', action='store_true', help='evaluate on 20%% of train.csv instead of test.csv')
    parser.add_argument('--seed', type=int, default=BASELINE['seed'])
    parser.add_argument('--no-normalize', action='store_true', help='keep raw 0-255 pixels')
    parser.add_argument('--hidden', default=','.join(map(str, BASELINE['hidden_dims'])),
                        help='hidden layer sizes, e.g. 512,256,128')
    parser.add_argument('--activation', choices=ACTIVATIONS, default=BASELINE['activation'])
    parser.add_argument('--dropout', type=float, default=BASELINE['dropout'])
    parser.add_argument('--optimizer', choices=OPTIMIZERS, default=BASELINE['optimizer'])
    parser.add_argument('--lr', type=float, default=BASELINE['learning_rate'])
    parser.add_argument('--weight-decay', type=float, default=BASELINE['weight_decay'])
    parser.add_argument('--momentum', type=float, default=BASELINE['momentum'],
                        help='only used by sgd')
    parser.add_argument('--loss', choices=sorted(LOSSES), default=BASELINE['loss'],
                        help=' | '.join(f'{k}: {v}' for k, v in LOSS_DESCRIPTIONS.items()))
    parser.add_argument('--batch-size', type=int, default=BASELINE['batch_size'])
    parser.add_argument('--epochs', type=int, default=BASELINE['max_epoch'])
    parser.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default=BASELINE['device'])
    parser.add_argument('--quiet', action='store_true', help='hide per-epoch training lines')
    return parser.parse_args(argv)


def build_config(args):
    return {
        'seed': args.seed,
        'normalize': not args.no_normalize,
        'hidden_dims': [int(h) for h in args.hidden.split(',') if h.strip()],
        'activation': args.activation,
        'dropout': args.dropout,
        'optimizer': args.optimizer,
        'learning_rate': args.lr,
        'weight_decay': args.weight_decay,
        'momentum': args.momentum if args.optimizer == 'sgd' else 0.0,
        'loss': args.loss,
        'batch_size': args.batch_size,
        'max_epoch': args.epochs,
        'device': args.device,
        'evaluated_on': 'validation (20% of train.csv)' if args.validate else 'test.csv',
    }


def optimizer_kwargs_for(config):
    """Extra optimizer arguments that Method_MLP should pass to the optimizer."""
    kwargs = {}
    if config['weight_decay']:
        kwargs['weight_decay'] = config['weight_decay']
    if config['optimizer'] == 'sgd' and config['momentum']:
        kwargs['momentum'] = config['momentum']
    return kwargs


def build_objects(config, validate, group=None, name=None, quiet=False):
    """Create the dataset / method / result / setting / evaluate objects."""
    set_seed(config['seed'])

    train_obj = Dataset_Loader('digits (train)', 'train.csv', split='train', normalize=config['normalize'])
    test_obj = Dataset_Loader('digits (test)', 'test.csv', split='test', normalize=config['normalize'])

    method_obj = Method_MLP(
        'multi-layer perceptron', '',
        hidden_dims=tuple(config['hidden_dims']),
        activation=ACTIVATIONS[config['activation']],
        dropout=config['dropout'],
        loss_function=build_loss(config['loss']),
    )
    method_obj.optimizer_class = OPTIMIZERS[config['optimizer']]
    method_obj.optimizer_kwargs = optimizer_kwargs_for(config)
    method_obj.learning_rate = config['learning_rate']
    method_obj.batch_size = config['batch_size']
    method_obj.max_epoch = config['max_epoch']
    method_obj.device = config['device']
    method_obj.verbose = not quiet

    run_name = name or ('baseline_validation' if validate else 'baseline')
    run_dir = RESULT_ROOT / group / run_name if group else RESULT_ROOT / run_name

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = run_dir
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Stage2('provided train/test split', '')
    setting_obj.use_validation = validate
    setting_obj.seed = config['seed']

    evaluate_obj = Evaluate_Metrics('accuracy, precision, recall, f1', '')
    return setting_obj, (train_obj, test_obj), method_obj, result_obj, evaluate_obj, run_dir


def save_run_artifacts(run_dir, config, scores, learned, eval_data, per_class=None):
    """Write config, metrics, convergence plots, confusion matrix and error text."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    (run_dir / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (run_dir / 'metrics.json').write_text(json.dumps(scores, indent=2) + '\n')

    history = learned.get('history') or {}
    # per-epoch curves as JSON too, so the comparison plots can be rebuilt from
    # a clone without the (large, git-ignored) pickled predictions
    (run_dir / 'history.json').write_text(json.dumps(history) + '\n')
    if history.get('train_loss'):
        Convergence_Plots.save_convergence_plots(
            history, run_dir / 'convergence.png',
            title=f"{config.get('optimizer')} lr={config.get('learning_rate')} "
                  f"hidden={config.get('hidden_dims')} loss={config.get('loss')}")
    else:
        print('note: Method_MLP did not record a training history, skipping convergence plots')

    true_y, pred_y = learned['true_y'], learned['pred_y']
    cm = Error_Analysis.compute_confusion_matrix(true_y, pred_y)
    np.savetxt(run_dir / 'confusion_matrix.csv', cm, fmt='%d', delimiter=',')
    Error_Analysis.save_confusion_matrix(
        cm, run_dir / 'confusion_matrix.png',
        title=f'Confusion matrix ({config.get("evaluated_on")})')
    Error_Analysis.save_misclassified_examples(
        eval_data['X'], true_y, pred_y, run_dir / 'misclassified_examples.png')
    report = Error_Analysis.write_report(cm, run_dir / 'error_analysis.txt')

    if per_class is None:
        per_class = per_class_metrics(true_y, pred_y)
    (run_dir / 'per_class_metrics.json').write_text(json.dumps(per_class, indent=2) + '\n')
    Convergence_Plots.save_per_class_plot(
        per_class, run_dir / 'per_class_metrics.png',
        title=f'Per-digit precision / recall / F1 ({config.get("evaluated_on")})')

    return report


def run_experiment(config, validate=False, group=None, name=None, quiet=False):
    """Run one full experiment and return (scores, learned_result, run_dir)."""
    setting_obj, (train_obj, test_obj), method_obj, result_obj, evaluate_obj, run_dir = build_objects(
        config, validate, group=group, name=name, quiet=quiet)

    print('************ Start ************')
    print('config:', config)
    setting_obj.prepare(train_obj, method_obj, result_obj, evaluate_obj, test_obj)
    setting_obj.print_setup_summary()

    start = time.time()
    scores, _ = setting_obj.load_run_save_evaluate()
    config['run_seconds'] = round(time.time() - start, 1)

    if not isinstance(scores, dict):
        scores = {'accuracy': scores}
    scores = {key: float(value) for key, value in scores.items()}

    print('************ Overall Performance ************')
    for key, value in scores.items():
        print(f'{key}: {value:.4f}')

    learned = setting_obj.learned_result
    report = save_run_artifacts(run_dir, config, scores, learned, setting_obj.eval_data)
    print(report)
    print(f'saved run to {run_dir}')
    print('************ Finish ************')
    return scores, learned, run_dir


def main(argv=None):
    args = parse_args(argv)
    config = build_config(args)
    run_experiment(config, validate=args.validate, group=args.group, name=args.name,
                   quiet=args.quiet)


if __name__ == '__main__':
    main()
