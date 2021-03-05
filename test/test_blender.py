"""Tests for blender.py."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

import two4two


def render(tmp_path: Path, **kwargs: Dict[str, Any]):
    """Renders 3 images and checks #objects, shapes, and if parameters are returned correctly."""
    sampler = two4two.Sampler()
    sampled_params = [sampler.sample() for _ in range(3)]
    i = 0
    for (img, mask, param) in two4two.render(
        sampled_params,
        **kwargs
    ):
        # should be 9 unique objects including background
        assert (np.unique(mask) == np.arange(9)).all()
        assert param == sampled_params[i]
        i += 1
    # ensures the for loop is executed
    assert i == len(sampled_params)


def test_blender_rending(tmp_path: Path):
    """Tests the rendering using the local blender version."""
    print("test temp dir: ", tmp_path)
    np.random.seed(242)
    render(
        tmp_path,
        n_processes=1,
        chunk_size=1,
        output_dir=str(tmp_path),
        download_blender=True
    )


def test_blender_rending_tmp_dir(tmp_path: Path):
    """Tests the rendering using a temporary directory."""
    np.random.seed(241)
    render(
        tmp_path,
        n_processes=1,
        chunk_size=1,
        output_dir=None,
    )


@pytest.mark.slow
def test_blender_download(tmp_path: Path):
    """Tests downloading of blender."""
    np.random.seed(242)
    print(tmp_path)
    render(
        tmp_path,
        n_processes=1,
        chunk_size=1,
        output_dir=str(tmp_path),
        download_blender=True,
        blender_dir=str(tmp_path),
    )
