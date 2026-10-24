"""Person 3 tests: evaluation metrics, losses and plotting.

Run from the template root:
    python -m unittest discover -s tests -v
"""

import json
import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from local_code.stage_2_code import Convergence_Plots, Error_Analysis
from local_code.stage_2_code.Evaluate_Metrics import (
    Evaluate_Metrics,
    compute_metrics,
    per_class_metrics,
)
from local_code.stage_2_code.Loss_Functions import build_loss


class MetricTests(unittest.TestCase):
    def setUp(self):
        # 10 samples, exactly one mistake: the second '2' is predicted as a 3
        self.true_y = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        self.pred_y = np.array([0, 0, 1, 1, 2, 3, 3, 3, 4, 4])

    def test_perfect_predictions_score_one(self):
        scores = compute_metrics(self.true_y, self.true_y)
        self.assertAlmostEqual(scores['accuracy'], 1.0)
        for average in ('weighted', 'macro', 'micro'):
            self.assertAlmostEqual(scores[f'f1_{average}'], 1.0)
            self.assertAlmostEqual(scores[f'precision_{average}'], 1.0)
            self.assertAlmostEqual(scores[f'recall_{average}'], 1.0)
        self.assertAlmostEqual(scores['cohen_kappa'], 1.0)

    def test_known_accuracy_and_micro_equivalence(self):
        scores = compute_metrics(self.true_y, self.pred_y)
        self.assertAlmostEqual(scores['accuracy'], 0.9)
        # for single-label multiclass, micro P/R/F1 all equal accuracy
        self.assertAlmostEqual(scores['precision_micro'], scores['accuracy'])
        self.assertAlmostEqual(scores['recall_micro'], scores['accuracy'])
        self.assertAlmostEqual(scores['f1_micro'], scores['accuracy'])
        # macro F1 is a different number: digit 2 got half its samples wrong
        # (f1 0.667) and digit 3 has precision 2/3 (f1 0.8), so the plain
        # average over the five digits is (1 + 1 + 0.667 + 0.8 + 1) / 5
        self.assertAlmostEqual(scores['f1_macro'], (1 + 1 + 2 / 3 + 0.8 + 1) / 5, places=6)
        self.assertNotAlmostEqual(scores['f1_macro'], scores['accuracy'])

    def test_macro_differs_from_weighted_on_imbalance(self):
        # class 0 has 8 samples, class 1 has 2; the model misses both class 1s
        true_y = np.array([0] * 8 + [1] * 2)
        pred_y = np.array([0] * 10)
        scores = compute_metrics(true_y, pred_y)
        self.assertAlmostEqual(scores['accuracy'], 0.8)
        self.assertAlmostEqual(scores['recall_macro'], 0.5)      # (1.0 + 0.0) / 2
        self.assertAlmostEqual(scores['recall_weighted'], 0.8)   # 8/10 of the samples
        self.assertLess(scores['balanced_accuracy'], scores['accuracy'])

    def test_accepts_torch_tensor_predictions(self):
        # Method_MLP returns a CPU tensor, so this must not crash
        pred = torch.as_tensor(self.pred_y)
        scores = compute_metrics(self.true_y, pred)
        self.assertAlmostEqual(scores['accuracy'], 0.9)

    def test_shape_mismatch_and_empty_input_raise(self):
        with self.assertRaises(ValueError):
            compute_metrics(self.true_y, self.pred_y[:-1])
        with self.assertRaises(ValueError):
            compute_metrics(np.array([]), np.array([]))

    def test_evaluate_class_returns_dict_of_floats(self):
        module = Evaluate_Metrics('metrics', '')
        module.data = {'true_y': self.true_y, 'pred_y': self.pred_y}
        scores = module.evaluate()
        self.assertIsInstance(scores, dict)
        self.assertIn('accuracy', scores)
        self.assertTrue(all(isinstance(value, float) for value in scores.values()))

    def test_per_class_metrics_cover_every_digit(self):
        table = per_class_metrics(self.true_y, self.pred_y)
        self.assertEqual(table['labels'], [0, 1, 2, 3, 4])
        self.assertEqual(table['support'], [2, 2, 2, 2, 2])
        self.assertAlmostEqual(table['recall'][0], 1.0)
        self.assertAlmostEqual(table['recall'][2], 0.5)


class LossTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.logits = torch.randn(6, 10)
        self.targets = torch.tensor([0, 1, 2, 3, 4, 5])

    def test_every_loss_returns_a_scalar(self):
        for name in ('ce', 'mse', 'nll', 'ce_ls'):
            loss = build_loss(name)(self.logits, self.targets)
            self.assertEqual(loss.dim(), 0, msg=name)
            self.assertGreater(loss.item(), 0.0, msg=name)

    def test_nll_matches_cross_entropy(self):
        ce = build_loss('ce')(self.logits, self.targets)
        nll = build_loss('nll')(self.logits, self.targets)
        self.assertAlmostEqual(ce.item(), nll.item(), places=5)

    def test_unknown_loss_raises(self):
        with self.assertRaises(ValueError):
            build_loss('nope')


class PlotTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.dir = Path(self.temp.name)
        self.history = {
            'train_loss': [1.0, 0.5, 0.3],
            'train_acc': [0.6, 0.85, 0.92],
            'val_loss': [1.1, 0.6, 0.4],
            'val_acc': [0.58, 0.83, 0.9],
        }

    def test_convergence_plot_written(self):
        path = Convergence_Plots.save_convergence_plots(self.history, self.dir / 'c.png')
        self.assertTrue(path.is_file())

    def test_convergence_plot_works_without_validation(self):
        history = {'train_loss': [1.0, 0.5], 'train_acc': [0.6, 0.9], 'val_loss': [], 'val_acc': []}
        path = Convergence_Plots.save_convergence_plots(history, self.dir / 'c2.png')
        self.assertTrue(path.is_file())

    def test_multi_run_plot_and_bar_chart(self):
        runs = [('adam', self.history), ('sgd', self.history)]
        self.assertTrue(Convergence_Plots.save_multi_run_convergence(
            runs, self.dir / 'm.png', metric='val_acc').is_file())
        self.assertTrue(Convergence_Plots.save_metric_bars(
            [('adam', {'accuracy': 0.97, 'f1_macro': 0.96}),
             ('sgd', {'accuracy': 0.9, 'f1_macro': 0.89})], self.dir / 'b.png').is_file())

    def test_multi_run_plot_without_metric_raises(self):
        with self.assertRaises(ValueError):
            Convergence_Plots.save_multi_run_convergence(
                [('adam', {'train_loss': [1.0]})], self.dir / 'm2.png', metric='val_acc')

    def test_comparison_table_writes_markdown_and_csv(self):
        rows = [{'run': 'adam', 'accuracy': 0.97, 'f1_macro': 0.96},
                {'run': 'sgd', 'accuracy': 0.9, 'f1_macro': 0.89}]
        text = Convergence_Plots.write_comparison_table(rows, self.dir / 't.md')
        self.assertIn('| adam |', text)
        self.assertTrue((self.dir / 't.md').is_file())
        self.assertTrue((self.dir / 't.csv').is_file())

    def test_confusion_matrix_helpers(self):
        true_y = np.array([0, 1, 1, 2])
        pred_y = np.array([0, 1, 2, 2])
        cm = Error_Analysis.compute_confusion_matrix(true_y, pred_y)
        self.assertEqual(int(cm.sum()), 4)
        self.assertEqual(int(np.trace(cm)), 3)
        self.assertEqual(Error_Analysis.top_confusions(cm, 1)[0], (1, 2, 1))


class ArtifactTests(unittest.TestCase):
    def test_saved_run_can_be_collected(self):
        """collect_results should read back what save_run_artifacts wrote."""
        from script.stage_2_script.script_mlp import save_run_artifacts

        with tempfile.TemporaryDirectory() as temp:
            run_dir = Path(temp) / 'group' / 'run'
            run_dir.mkdir(parents=True)

            history = {'train_loss': [1.0, 0.4], 'train_acc': [0.5, 0.9],
                       'val_loss': [1.2, 0.5], 'val_acc': [0.5, 0.88]}
            learned = {'pred_y': np.array([0, 1, 2]), 'true_y': np.array([0, 1, 1]),
                       'history': history}
            with (run_dir / 'prediction_result').open('wb') as handle:
                pickle.dump(learned, handle)

            config = {'optimizer': 'adam', 'learning_rate': 1e-3, 'hidden_dims': [256, 128],
                      'batch_size': 128, 'max_epoch': 2, 'dropout': 0.0, 'loss': 'ce',
                      'activation': 'relu', 'normalize': True, 'evaluated_on': 'validation'}
            scores = compute_metrics(learned['true_y'], learned['pred_y'])
            save_run_artifacts(run_dir, config, scores, learned, {'X': np.zeros((3, 784))})

            for name in ('config.json', 'metrics.json', 'convergence.png',
                         'confusion_matrix.csv', 'confusion_matrix.png',
                         'per_class_metrics.json', 'per_class_metrics.png',
                         'error_analysis.txt'):
                self.assertTrue((run_dir / name).is_file(), msg=name)

            rows = Convergence_Plots.collect_results(Path(temp))
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]['optimizer'], 'adam')
            self.assertEqual(rows[0]['hidden_dims'], '256,128')
            self.assertIn('accuracy', rows[0])
            self.assertAlmostEqual(
                json.loads((run_dir / 'metrics.json').read_text())['accuracy'],
                rows[0]['accuracy'])


if __name__ == '__main__':
    unittest.main()
