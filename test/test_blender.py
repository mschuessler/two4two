import numpy as np

from two4two import scene_parameters
from two4two import blender
from pathlib import Path
import json


def test_blender_rending(tmp_path: Path):
    np.random.seed(242)
    sampler = scene_parameters.SampleSceneParameters()
    params = sampler.sample()
    params_file = tmp_path.joinpath('params.json')
    with params_file.open('w') as f:
        f.writelines(json.dumps(params.state_dict()) + '\n')
    print(tmp_path)
    blender.Blender(
        str(params_file),
        str(tmp_path),
        n_processes=1,
        chunk_size=1,
    )
