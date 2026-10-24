"""Prepare raw and normalized training data for Daniel's comparison."""

import random

import numpy as np
from sklearn.model_selection import train_test_split

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader


def prepare_comparison(*, data_dir=None, validation_size=0.2, seed=2):
    """Return raw and normalized versions of the same train/validation split.

    The 'val' key contains validation data for the shared Stage 2 model.
    This function only opens train.csv. The supplied test.csv remains untouched.
    """
    if not 0 < validation_size < 1:
        raise ValueError('validation_size must be between 0 and 1')

    # Load only train.csv. Keep test.csv separate for the team's final evaluation.
    loader = Dataset_Loader(split='train', normalize=False, data_dir=data_dir)
    data = loader.load()

    # Split once so both experiments use exactly the same images and labels.
    # The default 80/20 split gives 48,000 training and 12,000 validation images.
    # Stratification keeps similar digit proportions in both groups.
    X_train, X_validation, y_train, y_validation = train_test_split(
        data['X'], data['y'], test_size=validation_size,
        random_state=seed, stratify=data['y'])

    # Preserve the original 0-255 inputs as the baseline experiment.
    # Use 'val' for the held-out training examples so the shared model can
    # record validation loss and accuracy after each epoch.
    raw = {
        'train': {'X': X_train, 'y': y_train},
        'val': {'X': X_validation, 'y': y_validation}
    }

    # Create 0-1 inputs from the same split. Copy labels so changes to one
    # experiment's data do not accidentally affect the other experiment.
    normalized = {
        'train': {'X': X_train / 255.0, 'y': y_train.copy()},
        'val': {'X': X_validation / 255.0, 'y': y_validation.copy()}
    }

    return {'raw': raw, 'normalized': normalized}


def set_seed(seed=2):
    """Call before creating each fresh model to repeat its random starting point."""
    # Preparing the data above does not require importing PyTorch.
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
