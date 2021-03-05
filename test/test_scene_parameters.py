"""Tests for ``scene_parameters.py``."""

import dataclasses
import json

from two4two import bias
from two4two import scene_parameters


@dataclasses.dataclass
class MyParameters(scene_parameters.SceneParameters):
    """Class to test subclass of SceneParameters."""
    my_field: str = "my very unique value"


def test_subclass_scene_parameters():
    """Tests if load selects the right subclass."""
    param = MyParameters()
    json_buf = json.dumps(param.state_dict())
    loaded_param = scene_parameters.SceneParameters.load(json.loads(json_buf))
    assert param == loaded_param
    assert type(loaded_param) == MyParameters
    assert loaded_param.my_field == "my very unique value"


def test_scene_parameters_loading():
    """Tests if SceneParameters are still equal after loading from json."""
    sampler = bias.Sampler()
    sampled_param = sampler.sample()
    json_buf = json.dumps(sampled_param.state_dict())
    loaded_param = scene_parameters.SceneParameters.load(json.loads(json_buf))
    assert sampled_param == loaded_param


def test_sample_scene_parameters():
    """Test sampling of SceneParameters."""
    sampler = bias.Sampler()
    for i in range(1000):
        param = sampler.sample()
        param.check_values()
