"""Tests for ``bias.py``."""

import numbers
import random

import numpy as np
import pytest

import two4two
from two4two import utils


def test_generic_sampler():
    """Tests if generic sample can handle all its intended types."""
    sampler = two4two.Sampler()
    scipy_trunc_normal = utils.truncated_normal(0, 0.5, 0, 1)
    py_uniform = random.random
    test_dict = {'peaky': scipy_trunc_normal, 'stretchy': py_uniform, 'ignore': None}

    assert isinstance(sampler._sample('peaky', scipy_trunc_normal), numbers.Number)
    assert isinstance(sampler._sample('peaky', test_dict), numbers.Number)
    assert isinstance(sampler._sample('stretchy', test_dict), numbers.Number)
    assert isinstance(sampler._sample('stretchy', test_dict, size=5), tuple)

    with pytest.raises(KeyError):
        two4two.Sampler._sample('ronny', test_dict)


def test_samplers_valid():
    """Test if the custom samplers run."""

    samplers = [
        two4two.Sampler(),
        two4two.ColorBiasedSampler(),
        two4two.HighVariationSampler(),
        two4two.HighVariationColorBiasedSampler()
    ]

    for sampler in samplers:
        for _ in range(40):
            sampler.sample()


def test_resampeling():
    """Tests if sampler is keeping track of which attributes are sampled again."""
    sampler = two4two.ColorBiasedSampler()
    param1 = two4two.SceneParameters()
    param2 = sampler.sample()
    assert all(value == 'default' for value in param1._attributes_status.values())
    assert all(value == 'sampled' for value in param2._attributes_status.values())
    sampler.sample_obj_rotation_roll(param1)
    sampler.sample_spherical(param2)
    param3 = param2.clone()
    sampler.sample_arm_position(param3)
    assert param1.get_status('obj_rotation_roll') == 'sampled'
    assert param2.get_status('spherical') == 'resampled'
    assert param3.get_status('spherical') == 'resampled'
    assert param3.get_status('arm_position') == 'resampled'
    assert param3.original_id == param2.id


def test_distribution_is_used():
    """Sets the distributions to a narrow interval and tests if sample fall into it."""

    sampler = two4two.ColorBiasedSampler()
    param = two4two.SceneParameters()

    # have a thight bound of values
    lower = 0.5
    upper = 0.5 + 0.00001

    continouos_attributes = [
        'labeling_error',
        'spherical',
        'bending',
        'arm_position',
        'obj_rotation_roll',
        'obj_rotation_yaw',
        'obj_rotation_pitch',
        'position_x',
        'position_y',
        'obj_color',
        'bg_color',
    ]

    for attr in continouos_attributes:
        setattr(sampler, attr, lambda: np.random.uniform(lower, upper))
        meth = getattr(sampler, 'sample_' + attr)
        meth(param)
        value = getattr(param, attr)
        assert lower <= value <= upper, f"failed for {attr}, value: {value}."
