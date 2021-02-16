"""Tests for ``scene_parameters.py``."""
import json

from two4two import scene_parameters


def test_scene_parameters_loading():
    """Tests if SceneParameters are still equal after loading from json."""
    sampler = scene_parameters.SampleSceneParameters()
    sampled_param = sampler.sample()
    json_buf = json.dumps(sampled_param.state_dict())
    loaded_param = scene_parameters.SceneParameters(**json.loads(json_buf))
    assert sampled_param == loaded_param
