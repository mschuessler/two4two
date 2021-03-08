"""This module contains the code to sample ``two4two.SceneParameters``."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Union

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats

from two4two import utils
from two4two.scene_parameters import SceneParameters


_Continouos = Union[scipy.stats.rv_continuous, Callable[[], float], float]
Continouos = Union[_Continouos, Dict[str, _Continouos]]

_Discrete = Union[scipy.stats.rv_discrete, Callable[[], float], Callable[[], str], float, str]
Discrete = Union[_Discrete, Dict[str, _Discrete]]

Distribution = Union[Discrete, Continouos]


@dataclasses.dataclass()
class Sampler:
    """Samples the parameters of the ``SceneParameters`` objects.

    Attributes describe how the sampling is done.
    Concretely they provide the color maps for the object and the background and
    the distributors from which the value for the scene parameters are drawn.

    Distribution can be:
    * scipy-distribution from ``scipy.stats``
    * callable functions returning a single value
    * a single (default) value.
    * a dictionary of all before-mentioned types containing the keys ``sticky``and ``stretchy``.

    These dictionaries are the easiest way to implement a bias.
    See ``ColorBiasedSceneParameterSampler`` as an example.

    To implement more complex biases, you can inherit this class and modify how individual
    attributes are sample, e.g., by introducing additional dependencies.

    For the valid values ranges, see ``SceneParameters.VALID_VALUES``.

    Attrs:
        bg_color_map: used color map for the background.
        obj_color_map: used color map for the object.
        spherical: distribution of ``SceneParameters.spherical``.
        bone_bend: distribution of ``SceneParameters.bone_bend``.
        bone_rotation: distribution of ``SceneParameters.bone_rotation``.
        obj_name: distribution of ``SceneParameters.obj_name``.
        arm_position: distribution of ``SceneParameters.arm_position``.
        labeling_error: distribution of ``SceneParameters.labeling_error``.
        obj_incline: distribution of ``SceneParameters.obj_incline``.
        obj_rotation:distribution of ``SceneParameters.obj_rotation``.
        fliplr: distribution of ``SceneParameters.fliplr``.
        position: distribution of ``SceneParameters.position``.
        obj_color_scalar: distribution of ``SceneParameters.obj_color_scalar``.
        bg_color: distribution of ``SceneParameters.bg_color``.
    """

    obj_name: Discrete = utils.discrete({'sticky': 0.5, 'stretchy': 0.5})
    spherical: Continouos = scipy.stats.beta(0.3, 0.3)
    bone_bend: Continouos = utils.truncated_normal(0, 0.1 * np.pi / 4, *utils.HALF_CIRCLE)
    bone_rotation: Continouos = utils.truncated_normal(0, 0.1 * np.pi / 4, *utils.HALF_CIRCLE)

    arm_position: Continouos = dataclasses.field(
        default_factory=lambda: {
            'sticky': utils.truncated_normal(mean=0, std=0.40, lower=0, upper=0.65),
            'stretchy': utils.truncated_normal(mean=1, std=0.40, lower=0, upper=0.65)
        })

    labeling_error: Discrete = utils.discrete({True: 0.05, False: 0.95})
    obj_incline: Continouos = utils.truncated_normal(0, 0.03 * np.pi / 4, *utils.HALF_CIRCLE)
    obj_rotation: Continouos = utils.truncated_normal(0, 0.3 * np.pi / 4, *utils.HALF_CIRCLE)
    fliplr: Discrete = utils.discrete({True: 0., False: 1.})
    position: Continouos = scipy.stats.uniform(-0.5, 0.5)
    obj_color_scalar: Continouos = scipy.stats.uniform(0., 1.)
    bg_color: Continouos = scipy.stats.uniform(0.05, 0.80)
    bg_color_map: str = 'binary'
    obj_color_map: str = 'seismic'

    def sample(self) -> SceneParameters:
        """Returns a new SceneParameters with random values.

        If you create your own biased sampled dataset by inheriting from this class,
        you might want to change the order of how attributes are set.
        For example, if you want that ``obj_rotation`` should depend on the
        ``arm_position``then you should also sample the ``arm_position`` first.
        However, it is highly recommended to sample the object name first, as
        the sampling of the attribute might be dependent on the label
        (see the explanation of distributions in class description)
        """
        params = SceneParameters()
        self.sample_obj_name(params)
        self.sample_spherical(params)
        self.sample_bone_bend(params)
        self.sample_bone_rotation(params)
        self.sample_obj_incline(params)
        self.sample_obj_rotation(params)
        self.sample_fliplr(params)
        self.sample_position(params)
        self.sample_arm_position(params)
        self.sample_obj_color(params)
        self.sample_bg_color(params)
        params.check_values()
        return params

    @staticmethod
    def _sample(obj_name: str, dist: Distribution, size: int = 1) -> Any:
        """Samples values from the distributon according to its type.

        The default number of values sampled is one, which can be changed with flag size.

        Distribution can be:
        * scipy-distribution from ``scipy.stats``
        * callable functions returning a single value
        * a single (default) value.
        * a dictionary of all before-mentioned types containing the keys ``sticky``and ``stretchy``.

        Will unpack np.ndarray, list, or tuple with a single element returned by distribution.

        """

        if size > 1:
            return tuple([Sampler._sample(obj_name, dist) for i in range(0, size)])

        if isinstance(dist, dict):
            dist = dist[obj_name]

        if hasattr(dist, 'rvs'):
            value = dist.rvs()
        elif callable(dist):
            value = dist()
        else:
            value = dist

        # Unpacking float values contained in numpyarrays and list
        if type(value) in (list, tuple):
            if len(value) != 1:
                raise ValueError(f"Expected a single element. \
                 Got {type(value)} of size {len(value)}!")
            else:
                value = value[0]

        if isinstance(value, np.ndarray):
            value = utils.to_python_scalar(value)

        return value

    def sample_obj_name(self, params: SceneParameters):
        """Samples the ``obj_name``."""
        params.obj_name = self._sample(None, self.obj_name)

    def sample_labeling_error(self, params: SceneParameters):
        """Samples the ``labeling_error``."""
        params.labeling_error = self._sample(params.obj_name, self.labeling_error)

    def sample_spherical(self, params: SceneParameters):
        """Samples the ``spherical``."""
        params.spherical = self._sample(params.obj_name, self.spherical)

    def sample_bone_bend(self, params: SceneParameters):
        """Samples the ``bone_bend``."""
        params.bone_bend = self._sample(params.obj_name, self.bone_bend, size=7)

    def sample_bone_rotation(self, params: SceneParameters):
        """Samples the ``bone_rotation``."""
        params.bone_rotation = self._sample(params.obj_name, self.bone_rotation, size=7)

    def sample_obj_incline(self, params: SceneParameters):
        """Samples the ``obj_incline``."""
        params.obj_incline = self._sample(params.obj_name, self.obj_incline)

    def sample_obj_rotation(self, params: SceneParameters):
        """Samples the ``obj_rotation``."""
        params.obj_rotation = self._sample(params.obj_name, self.obj_rotation)

    def sample_fliplr(self, params: SceneParameters):
        """Samples the ``fliplr``."""
        params.fliplr = self._sample(params.obj_name, self.fliplr)

    def sample_position(self, params: SceneParameters):
        """Samples the ``position``."""
        # params.position = self.position.rvs(2).tolist()
        params.position = self._sample(params.obj_name, self.position, size=2)

    def sample_arm_position(self, params: SceneParameters):
        """Samples the ``arm_position``."""
        params.arm_position = float(self._sample(params.obj_name, self.arm_position))

    def _object_cmap(self, params: SceneParameters) -> utils.ColorGenerator:
        return plt.get_cmap(self.obj_color_map)

    def sample_obj_color(self, params: SceneParameters):
        """Samples the ``obj_color_scalar`` and ``obj_color_scalar``."""
        params.obj_color_scalar = float(self._sample(params.obj_name, self.obj_color_scalar))
        params.obj_color = tuple(self._object_cmap(params)(params.obj_color_scalar))

    def _bg_cmap(self, params: SceneParameters) -> mpl.colors.Colormap:
        return plt.get_cmap(self.bg_color_map)

    def sample_bg_color(self, params: SceneParameters):
        """Samples the ``bg_color`` and ``bg_color_scalar``."""
        params.bg_color_scalar = float(self._sample(params.obj_name, self.bg_color))
        params.bg_color = tuple(self._bg_cmap(params)(params.bg_color_scalar))


class ColorBiasedSampler(Sampler):
    """An example implementation of a color-biased SceneParameterSample.

    The color is sampled from a conditional distribution that is dependent on the object type.
    """

    obj_color_scalar: Continouos = {
        'sticky': utils.truncated_normal(1, 0.5, 0, 1),
        'stretchy': utils.truncated_normal(0, 0.5, 0, 1),
    }
