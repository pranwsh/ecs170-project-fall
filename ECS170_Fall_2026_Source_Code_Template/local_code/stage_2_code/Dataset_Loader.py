"""Load handwritten digit images and labels for Stage 2."""

from pathlib import Path

import numpy as np
import pandas as pd

from local_code.base_class.dataset import dataset


# Locate the supplied datasets independently of PyCharm's working directory.
DATA_DIRECTORY = Path(__file__).resolve().parents[2] / 'data' / 'stage_2_data'


class Dataset_Loader(dataset):
    """Load train.csv or test.csv, with optional pixel normalization."""

    def __init__(self, dName=None, dDescription=None, *, split='train',
                 normalize=True, data_dir=None):
        super().__init__(dName, dDescription)
        if split not in ('train', 'test'):
            raise ValueError("split must be 'train' or 'test'")
        if not isinstance(normalize, bool):
            raise TypeError('normalize must be a bool')
        self.normalize = normalize

        # Preserve the source-path fields used by the professor's scripts.
        if data_dir is None:
            self.dataset_source_folder_path = DATA_DIRECTORY
        else:
            self.dataset_source_folder_path = Path(data_dir)
        self.dataset_source_file_name = f'{split}.csv'

    def load(self):
        """Return X with shape (N, 784) and y with shape (N,).

        The supplied files contain 60,000 training or 10,000 testing images.
        Smaller datasets are also allowed for testing the loader.
        """
        path = Path(self.dataset_source_folder_path) / self.dataset_source_file_name
        if not path.is_file():
            raise FileNotFoundError(f'Stage 2 CSV not found: {path}')

        # This repository uses Git LFS; the actual CSV must be downloaded first.
        with path.open('r', encoding='utf-8') as source:
            if source.readline().startswith('version https://git-lfs.github.com/spec/'):
                raise ValueError(f'{path} is a Git LFS pointer; fetch the dataset with git lfs pull')
        # The CSV has no header. Keep blank rows so missing examples are caught.
        try:
            table = pd.read_csv(path, header=None, skip_blank_lines=False)
            values = table.to_numpy(dtype=np.float64)
        except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
            raise ValueError(f'Invalid Stage 2 CSV {path}: {exc}') from exc
        # Each row must contain one label and all 784 pixels of a 28x28 image.
        if values.shape[0] == 0 or values.shape[1] != 785:
            raise ValueError(f'{path}: expected nonempty rows with 785 columns (label + 784 pixels); got {values.shape}')
        if not np.isfinite(values).all():
            raise ValueError(f'{path}: labels and pixels must be finite, with no missing values')
        # Labels and images must stay in matching row order.
        labels = values[:, 0]
        pixels = values[:, 1:]

        # Validate before converting types, so a label like 3.5 is not rounded
        # down to 3 and accidentally accepted as a valid digit.
        if np.any(labels < 0) or np.any(labels > 9) or np.any(labels != np.floor(labels)):
            raise ValueError(f'{path}: labels must be integers from 0 through 9')
        if np.any(pixels < 0) or np.any(pixels > 255):
            raise ValueError(f'{path}: pixel values must be between 0 and 255')
        # Match the input and label types expected by the PyTorch model.
        X = pixels.astype(np.float32)
        y = labels.astype(np.int64)

        # Compare original 0-255 pixels with 0-1 inputs in our experiments.
        if self.normalize:
            X /= 255.0

        # Keep the same output format as the Stage 1 loader.
        self.data = {'X': X, 'y': y}
        return self.data
