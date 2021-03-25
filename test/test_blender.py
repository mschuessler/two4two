"""Tests for blender.py."""

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

import two4two


def render(tmp_path: Path, **kwargs: Dict[str, Any]):
    """Renders 3 images and checks #objects, shapes, and if parameters are returned correctly."""
    sticky = two4two.SceneParameters.default_sticky()

    # setting the id ensures a determinisitic filename
    sticky.id = 'sticky'

    sticky_bended = sticky.clone()
    sticky_bended.id = 'sticky_bended'
    sticky_bended.bending = 0.25

    sampled_params = [
        sticky,
        sticky_bended,
    ]

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


def test_blender_rendering(tmp_path: Path):
    """Tests the rendering using the local blender version."""
    print("test temp dir: ", tmp_path)
    np.random.seed(242)
    render(
        tmp_path,
        n_processes=1,
        chunk_size=1,
        output_dir=str(tmp_path),
        download_blender=True,
        save_blender_file=True,
        print_output=False,
        print_cmd=True,
    )
    assert tmp_path.glob("*.png")
    assert tmp_path.glob("*.blender")

    two4two.blender.render_single(two4two.SceneParameters())


def test_blender_rending_tmp_dir(tmp_path: Path):
    """Tests the rendering using a temporary directory."""
    np.random.seed(241)
    render(
        tmp_path,
        n_processes=1,
        chunk_size=1,
        output_dir=None,
    )


def test_blender_fliplr(tmp_path: Path):
    """Tests if fliplr produces the exact same image but only flipped."""
    np.random.seed(200002)
    sampler = two4two.Sampler()
    param_original = sampler.sample()
    param_flip = param_original.clone()
    param_flip.fliplr = True

    print(tmp_path)
    original_path = tmp_path / 'original'
    fliplr_path = tmp_path / 'fliplr'
    original_path.mkdir()
    fliplr_path.mkdir()

    for (img_original, mask_original, _) in two4two.render(
        [param_original],
        output_dir=str(original_path),
    ):
        pass

    for (img_fliplr, mask_fliplr, _) in two4two.render(
        [param_flip],
        output_dir=str(fliplr_path),
    ):
        pass

    assert (img_fliplr == img_original[:, ::-1]).all()
    assert (mask_fliplr == mask_original[:, ::-1]).all()


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
        print_output=True,
        print_cmd=True
    )
