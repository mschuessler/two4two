"""Tests for ``two4two/pytorch.py``."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import two4two
import two4two.pytorch


def test_pytorch_dataloader(tmp_path: Path):
    """Tests if dataset can load the rendered images."""
    print("test temp dir: ", tmp_path)
    np.random.seed(242)

    sampler = two4two.Sampler()
    sampled_params = [sampler.sample() for _ in range(2)]

    (tmp_path / 'train').mkdir()

    for _ in two4two.render(
        sampled_params,
        n_processes=1,
        output_dir=str(tmp_path / 'train'),
    ):
        pass

    dataset = two4two.pytorch.Two4Two(str(tmp_path), split='train')

    df = dataset.get_dataframe()
    assert df.obj_name[0] == sampled_params[0].obj_name
    assert df.obj_name[1] == sampled_params[1].obj_name
    assert "resolution" not in set(df.keys())

    df = dataset.get_dataframe(to_dict=two4two.pytorch.all_attributes)
    assert df.attribute_status_obj_name[0] == "sampled"
    assert df.attribute_status_obj_name[1] == "sampled"

    # check dataset shapes
    assert len(dataset) == 2
    img, mask, labels = dataset[0]
    assert img.shape == (3, 128, 128)
    assert mask.shape == (1, 128, 128)
    assert labels.shape == (1,)

    assert type(img) == torch.Tensor
    assert type(mask) == torch.Tensor
    assert type(labels) == torch.Tensor

    # check dataset loader
    dataloader = DataLoader(dataset, batch_size=2)
    imgs, masks, labels = next(iter(dataloader))

    assert type(imgs) == torch.Tensor
    assert type(masks) == torch.Tensor
    assert type(labels) == torch.Tensor

    assert imgs.shape == (2, 3, 128, 128)
    assert masks.shape == (2, 1, 128, 128)
    assert labels.shape == (2, 1,)

    dataset.set_return_attributes([
        'obj_name', 'bending', 'bg_color', 'spherical'])

    label_names = dataset.get_label_names()
    expected_label_names = [
        'obj_name',
        'bending',
        'bg_color',
        'spherical',
    ]
    assert label_names == expected_label_names

    img, mask, labels = dataset[0]
    assert labels.shape == (len(expected_label_names),)
