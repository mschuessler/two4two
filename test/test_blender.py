"""Tests for blender.py."""
from pathlib import Path

import numpy as np
import pytest

from two4two import blender
from two4two import scene_parameters


def test_blender_rending(tmp_path: Path):
    """Tests the rendering using the local blender version."""
    np.random.seed(242)
    sampler = scene_parameters.SampleSceneParameters()
    sampled_params = [sampler.sample() for _ in range(3)]
    print("test temp dir: ", tmp_path)
    i = 0
    for (img, param) in blender.render(
        sampled_params,
        n_processes=1,
        chunk_size=1,
        output_dir=str(tmp_path),
        download_blender=True
    ):
        assert param == sampled_params[i]
        i += 1
    # ensures the for loop is executed
    assert i == len(sampled_params)


def test_blender_rending_tmp_dir(tmp_path: Path):
    """Tests the rendering using a temporary directory."""
    np.random.seed(242)
    sampler = scene_parameters.SampleSceneParameters()
    sampled_params = [sampler.sample() for _ in range(3)]
    i = 0
    for (img, param) in blender.render(
        sampled_params,
        n_processes=1,
        chunk_size=1,
        output_dir=None,
    ):
        assert param == sampled_params[i]
        assert img.shape == (128, 128, 4)
        i += 1

    # ensures the for loop is executed
    assert i == len(sampled_params)


@pytest.mark.slow
def test_blender_download(tmp_path: Path):
    """Tests downloading of blender."""
    np.random.seed(242)
    sampler = scene_parameters.SampleSceneParameters()
    sampled_param = sampler.sample()
    print(tmp_path)
    for img, param in blender.render(
        [sampled_param],
        n_processes=1,
        chunk_size=1,
        output_dir=str(tmp_path),
        download_blender=True,
        blender_dir=str(tmp_path),
    ):
        pass

    assert param == sampled_param
