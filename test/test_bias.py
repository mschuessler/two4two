"""Tests for ``bias.py``."""

import numbers
import random

import pytest

import two4two
from two4two import utils


def test_generic_sampler():
    """Tests if generic sample can handle all its intended types."""
    sampler = two4two.Sampler()
    scipy_trunc_normal = utils.truncated_normal(0, 0.5, 0, 1)
    py_uniform = random.random
    test_dict = {'sticky': scipy_trunc_normal, 'stretchy': py_uniform, 'ignore': None}

    assert isinstance(sampler._sample('sticky', scipy_trunc_normal), numbers.Number)
    assert isinstance(sampler._sample('sticky', test_dict), numbers.Number)
    assert isinstance(sampler._sample('stretchy', test_dict), numbers.Number)
    assert isinstance(sampler._sample('stretchy', test_dict, size=5), list)

    with pytest.raises(KeyError):
        two4two.Sampler._sample('ronny', test_dict)

    colorBiasedSample = two4two.ColorBiasedSampler()
    colorBiasedSample.sample()
