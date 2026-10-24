"""Aggregate every saved Stage 2 run into one comparison table and figure.

Reads the config.json / metrics.json / prediction_result files that
script_mlp.py and run_experiments.py wrote, so it never re-trains anything.

    python -m script.stage_2_script.summarize_results
    python -m script.stage_2_script.summarize_results --group person3
    python -m script.stage_2_script.summarize_results --group person3 --out summary

Writes <out>.md, <out>.csv and <out>_accuracy.png next to the result root
(default: result/stage_2_result/summary).
"""

import argparse
import sys
from pathlib import Path

TEMPLATE_ROOT = Path(__file__).resolve().parents[2]
if str(TEMPLATE_ROOT) not in sys.path:
    sys.path.insert(0, str(TEMPLATE_ROOT))

import matplotlib
matplotlib.use('Agg')

from local_code.stage_2_code import Convergence_Plots
from script.stage_2_script.script_mlp import RESULT_ROOT

COLUMNS = ['accuracy', 'f1_macro', 'f1_weighted', 'precision_macro', 'recall_macro',
           'balanced_accuracy', 'cohen_kappa']


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description='Summarize saved Stage 2 runs')
    parser.add_argument('--group', default=None, help='sub-folder to summarize (default: everything)')
    parser.add_argument('--out', default=None, help='output basename (default: <root>/summary)')
    parser.add_argument('--sort', default='accuracy', help='metric to sort by (default: accuracy)')
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = Convergence_Plots.collect_results(RESULT_ROOT, group=args.group)
    if not rows:
        print(f'no saved runs found under {RESULT_ROOT / (args.group or "")}')
        return

    rows.sort(key=lambda row: row.get(args.sort, 0.0), reverse=True)
    columns = [c for c in COLUMNS if any(c in row for row in rows)]

    out = Path(args.out) if args.out else (
        RESULT_ROOT / args.group / 'summary' if args.group else RESULT_ROOT / 'summary')
    out.parent.mkdir(parents=True, exist_ok=True)

    text = Convergence_Plots.write_comparison_table(
        rows, out.with_suffix('.md'), columns=columns,
        title=f'All Stage 2 runs (sorted by {args.sort})')
    Convergence_Plots.save_metric_bars(
        [(row['run'], row) for row in rows], out.parent / f'{out.name}_accuracy.png',
        title=f'Stage 2 runs sorted by {args.sort}')
    print(text)
    print(f'wrote {out.with_suffix(".md")}, {out.with_suffix(".csv")} and '
          f'{out.parent / (out.name + "_accuracy.png")}')


if __name__ == '__main__':
    main()
