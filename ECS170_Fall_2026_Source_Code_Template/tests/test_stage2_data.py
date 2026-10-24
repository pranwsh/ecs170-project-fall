"""Daniel's data tests: python -m unittest discover -s tests -v."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

from local_code.base_class.dataset import dataset
from local_code.stage_2_code.Dataset_Loader import DATA_DIRECTORY, Dataset_Loader
from local_code.stage_2_code.Preprocessing_Experiments import prepare_comparison, set_seed
from local_code.stage_2_code.Visualize_Data import save_visualizations


class LoaderTests(unittest.TestCase):
    def setUp(self):
        # Give each test its own small CSV so it cannot change the real dataset.
        # Include ten examples per digit, with a unique ID in the first pixel
        # to identify each row when checking the training/validation split.
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.rows = np.zeros((100, 785))
        self.rows[:, 0] = np.repeat(np.arange(10), 10)
        self.rows[:, 1] = np.arange(100)
        self.rows[:, -1] = 255
        self.write(self.rows)

    def write(self, rows):
        # Save test data in the same label-first, headerless format as train.csv.
        np.savetxt(self.directory / 'train.csv', rows, delimiter=',', fmt='%g')

    def loader(self, **kwargs):
        # Point the loader at this test's temporary folder.
        return Dataset_Loader('digits', 'Stage 2', data_dir=self.directory, **kwargs)

    def test_template_and_normalization(self):
        # Check compatibility with the professor's dataset class and output
        # format, including shapes, types, unchanged labels, and pixels / 255.
        loader = self.loader()
        self.assertIsInstance(loader, dataset)
        result = loader.load()
        self.assertIs(loader.data, result)
        self.assertEqual(result['X'].shape, (100, 784))
        self.assertEqual(result['y'].shape, (100,))
        self.assertEqual(result['X'].dtype, np.float32)
        self.assertEqual(result['y'].dtype, np.int64)
        np.testing.assert_array_equal(result['y'], self.rows[:, 0])
        np.testing.assert_array_equal(result['X'], self.rows[:, 1:].astype(np.float32) / 255)

    def test_raw_legacy_paths(self):
        # Setting the template's folder/file fields should still work, and
        # disabling normalization should preserve every original pixel value.
        loader = Dataset_Loader(normalize=False)
        loader.dataset_source_folder_path = str(self.directory)  # no trailing slash
        loader.dataset_source_file_name = 'train.csv'
        np.testing.assert_array_equal(loader.load()['X'], self.rows[:, 1:])

    def test_missing_file_and_options(self):
        # Missing files and unsupported settings should raise clear errors.
        # A string such as 'False' should not be accepted as a boolean setting.
        with self.assertRaises(FileNotFoundError):
            self.loader(split='test').load()
        with self.assertRaises(ValueError):
            self.loader(split='validation')
        with self.assertRaises(TypeError):
            self.loader(normalize='False')

    def test_malformed_data(self):
        # Deliberately create wrong column counts, invalid digit labels, and
        # out-of-range or nonfinite pixels. The loader must reject every case.
        cases = [self.rows[:, :-1], np.column_stack((self.rows, np.zeros(100)))]
        for column, value in [(0, -1), (0, 10), (0, 1.5), (1, -1), (1, 256), (1, np.nan), (1, np.inf)]:
            bad = self.rows.copy()
            bad[0, column] = value
            cases.append(bad)
        for index, bad in enumerate(cases):
            with self.subTest(case=index):
                self.write(bad)
                with self.assertRaises(ValueError):
                    self.loader().load()

    def test_text_empty_ragged_lfs_and_blank_rows(self):
        # Reject empty files, nonnumeric text, inconsistent row lengths,
        # Git LFS pointers, and blank rows instead of accepting incomplete data.
        for content in ['', 'label,pixel\n', '1,2\n1,2,3\n',
                        'version https://git-lfs.github.com/spec/v1\n',
                        ','.join(['0'] * 785) + '\n\n']:
            with self.subTest(content=content[:30]):
                (self.directory / 'train.csv').write_text(content)
                with self.assertRaises(ValueError):
                    self.loader().load()

    def test_paired_split_and_reproducibility(self):
        # Prepare twice with the same default seed. Raw and normalized inputs
        # must use matching examples, and repeated preparation must match too.
        # With this 100-row dataset, the default split should be 80/20.
        first = prepare_comparison(data_dir=self.directory)
        again = prepare_comparison(data_dir=self.directory)
        raw, normalized = first['raw'], first['normalized']
        # Keep validation separate from the final testing dataset.
        self.assertEqual(set(raw), {'train', 'val'})
        self.assertEqual(set(normalized), {'train', 'val'})
        for split, count in [('train', 80), ('val', 20)]:
            self.assertEqual(len(raw[split]['y']), count)
            np.testing.assert_array_equal(raw[split]['y'], normalized[split]['y'])
            np.testing.assert_array_equal(raw[split]['X'] / 255, normalized[split]['X'])
            np.testing.assert_array_equal(raw[split]['X'], again['raw'][split]['X'])
            self.assertEqual(len(np.unique(raw[split]['y'])), 10)
        # Use the unique first-pixel IDs from setUp to check that the groups
        # do not overlap and together contain all 100 original examples.
        train_ids = set(raw['train']['X'][:, 0])
        validation_ids = set(raw['val']['X'][:, 0])
        self.assertFalse(train_ids & validation_ids)
        self.assertEqual(len(train_ids | validation_ids), 100)

    def test_seed_and_separate_experiment_arrays(self):
        # Reset to the SAME seed before each draw. Python, NumPy, and PyTorch
        # should repeat their random values, as needed for matching model runs.
        import random
        import torch
        set_seed(2)
        first = (random.random(), np.random.random(), torch.rand(1).item())
        set_seed(2)
        second = (random.random(), np.random.random(), torch.rand(1).item())
        self.assertEqual(first, second)

        # Both versions should own separate arrays, so changing one's pixels
        # or labels cannot accidentally change the other experiment's inputs.
        comparison = prepare_comparison(data_dir=self.directory)
        for split in ('train', 'val'):
            for key in ('X', 'y'):
                self.assertFalse(np.shares_memory(comparison['raw'][split][key],
                                                  comparison['normalized'][split][key]))
        # Reserving 100% for validation would leave no data for training.
        with self.assertRaises(ValueError):
            prepare_comparison(data_dir=self.directory, validation_size=1)

    def test_figures(self):
        # Check that both plots are saved as nonempty PNG files. This
        # verifies file creation; it does not judge their visual appearance.
        destination = self.directory / 'figures'
        save_visualizations(self.loader(normalize=False).load(), destination)
        self.assertEqual(
            {p.name for p in destination.glob('*.png')},
            {'sample_digits.png', 'label_distribution.png'})
        self.assertTrue(all(p.stat().st_size > 1000 for p in destination.glob('*.png')))


class ProvidedDatasetTests(unittest.TestCase):
    def test_full_training_and_testing_files(self):
        # Verify both real CSVs have the expected sizes, numerical types, digit
        # labels, and raw/normalized ranges. No model is trained or evaluated.
        for split, count in [('train', 60000), ('test', 10000)]:
            with self.subTest(split=split):
                raw = Dataset_Loader(split=split, normalize=False).load()
                normalized = Dataset_Loader(split=split).load()
                self.assertEqual(raw['X'].shape, (count, 784))
                self.assertEqual(raw['y'].shape, (count,))
                self.assertEqual(raw['X'].dtype, np.float32)
                self.assertEqual(raw['y'].dtype, np.int64)
                self.assertEqual(set(raw['y']), set(range(10)))
                self.assertEqual(float(raw['X'].min()), 0)
                self.assertEqual(float(raw['X'].max()), 255)
                self.assertEqual(normalized['X'].dtype, np.float32)
                self.assertEqual(float(normalized['X'].min()), 0)
                self.assertEqual(float(normalized['X'].max()), 1)
                np.testing.assert_array_equal(raw['y'], normalized['y'])
                # Check every normalized pixel equals its raw value / 255.
                # Compare 1,000 rows at a time to avoid another full-size copy.
                for start in range(0, count, 1000):
                    np.testing.assert_array_equal(raw['X'][start:start+1000] / 255,
                                                  normalized['X'][start:start+1000])


if __name__ == '__main__':
    unittest.main()
