"""Run from the template root: python -m local_code.stage_2_code.Visualize_Data."""

import argparse
from pathlib import Path

import matplotlib
# Save figures without requiring a separate plotting window.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from local_code.stage_2_code.Dataset_Loader import Dataset_Loader


# Keep figures together under the template's result folder.
RESULT_DIRECTORY = Path(__file__).resolve().parents[2] / 'result' / 'stage_2_result' / 'daniel'


def save_visualizations(data, output_dir=RESULT_DIRECTORY, seed=2):
    """Save sample digit images and a chart of training label counts.

    Pass training data loaded with normalize=False to display pixels
    using their original 0-255 brightness range.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    X = data['X']
    y = data['y']

    # 1. Select up to 25 different images. Reuse the seed so repeated runs
    # show the same examples, making the figures easier to compare.
    sample_count = min(25, len(y))
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(y), size=sample_count, replace=False)

    fig, axes = plt.subplots(5, 5, figsize=(8, 8))
    axes = axes.flatten()

    # Hide plot axes and leave unused spaces blank for smaller datasets.
    for ax in axes:
        ax.axis('off')

    # Restore each flattened image to 28x28 and show its correct digit label.
    for position in range(sample_count):
        index = indices[position]
        image = X[index].reshape(28, 28)
        axes[position].imshow(image, cmap='gray', vmin=0, vmax=255)
        axes[position].set_title(f'Label: {y[index]}')

    fig.suptitle('Random training examples')
    fig.tight_layout()
    fig.savefig(output_dir / 'sample_digits.png', dpi=150)
    plt.close(fig)

    # 2. Count every training label to check the number of examples per digit.
    # Include all ten digits even if a smaller dataset is missing a class.
    counts = np.bincount(y, minlength=10)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(np.arange(10), counts)
    ax.set_xticks(np.arange(10))
    ax.set_xlabel('Digit label')
    ax.set_ylabel('Number of examples')
    ax.set_title('Training label distribution')
    fig.tight_layout()
    fig.savefig(output_dir / 'label_distribution.png', dpi=150)
    plt.close(fig)

    return counts


def main():
    # Allow custom folders and a sample seed while keeping project defaults.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path)
    parser.add_argument('--output-dir', type=Path, default=RESULT_DIRECTORY)
    parser.add_argument('--seed', type=int, default=2)
    args = parser.parse_args()
    # Load training pixels on their original scale; test.csv is not needed.
    loader = Dataset_Loader(split='train', normalize=False, data_dir=args.data_dir)
    data = loader.load()
    counts = save_visualizations(data, args.output_dir, args.seed)
    print(f'Saved two figures to {args.output_dir}; label counts: {counts.tolist()}')


if __name__ == '__main__':
    main()
