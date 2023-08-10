"""Tests for the baseline module."""


from pathlib import Path

import numpy as np

from two4two import baseline


def test_baseline(tmp_path: Path):
    """Test the baseline function."""
    images = np.random.normal(size=(1000, 3, 128, 128))
    logits = np.random.normal(size=(1000,))

    results = baseline.baseline(
        images, logits, figure_path=str(tmp_path / "baseline.png"),
    )
    print(results.fig_path)
