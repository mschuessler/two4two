"""Tests for ``scene_parameters.py``."""

import dataclasses
import json
import random
import numbers

import pytest

from two4two import scene_parameters
from two4two import utils



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
    sampler = scene_parameters.SampleSceneParameters()
    sampled_param = sampler.sample()
    json_buf = json.dumps(sampled_param.state_dict())
    loaded_param = scene_parameters.SceneParameters.load(json.loads(json_buf))
    assert sampled_param == loaded_param


def test_generic_sampler():
    """Tests if generic sample can handle all its intended types."""
    sampler = scene_parameters.SampleSceneParameters()
    dist1 = utils.truncated_normal(0, 0.5, 0, 1)
    dist2 = random.random
    dict = {'sticky': dist1, 'stretchy': dist2, 'ignore': None}

    assert isinstance(sampler._sample('sticky', dist1), numbers.Number)
    assert isinstance(sampler._sample('sticky', dict), numbers.Number)
    assert isinstance(sampler._sample('stretchy', dict), numbers.Number)
    assert isinstance(sampler._sample('stretchy', dict, size=5), list)

    with pytest.raises(KeyError):
        scene_parameters.SampleSceneParameters._sample('ronny', dict)

    colorBiasedSample = scene_parameters.ColorBiasedSceneParameterSampler()
    colorBiasedSample.sample()


def test_sample_scene_parameters():
    """Test sampling of SceneParameters."""
    sampler = scene_parameters.SampleSceneParameters()
    for i in range(1000):
        param = sampler.sample()
        param.check_values()
