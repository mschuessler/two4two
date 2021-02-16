import numpy as np

from two4two import scene_parameters
from two4two import blender
from pathlib import Path
import json


def test_blender_rending(tmp_path: Path):
    np.random.seed(242)
    sampler = scene_parameters.SampleSceneParameters()
    sampled_param = sampler.sample()
    print(tmp_path)
    for img, param in blender.render(
        [sampled_param],
        n_processes=1,
        chunk_size=1,
        output_dir=str(tmp_path),
    ):
        assert param == sampled_param

