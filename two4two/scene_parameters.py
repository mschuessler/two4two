"""This module contains classes to sample and describes individual scenes."""

from __future__ import annotations

import copy
import dataclasses
import pprint
from typing import Any, Callable, Dict, Optional, Sequence, Tuple, Union
import uuid

import numpy as np
import scipy.stats

from two4two import utils

RGBAColor = Tuple[float, float, float, float]

HALF_CIRCLE = (-np.pi / 4, np.pi / 4)


@dataclasses.dataclass()
class SceneParameters:
    """All parameters need to render a single image / scene.

    See the ``SceneParameters.VALID_VALUES`` for valid value ranges for the
    invididual attributes. Intervals are encoded as tuples and categoricals
    as sets.

    Attrs:
        obj_name: Object name (either ``"sticky"`` or ``"stretchy"``).
        labeling_error: If ``True``, the ``obj_name_with_label_error``, will
            return the flipped obj_name. The ``obj_name`` attribute itself will
            not change.
        spherical: For ``1``,  spherical objects. For ``0``, cubes.
            Can have values in-between.
        bone_bend: Bending of the individual bone segments.
        bone_rotation: Rotation of the individual bone segments.
        obj_incline: Incline of the object.
        obj_rotation: Rotation of the whole object.
        flip: Wheter the image should be flipped left to right.
        position: Position of the object.
        arm_position: Absolute arm positions.
        obj_color: Object color as RGBA
        obj_scalar: Object color in [0, 1]. This is before converting
            the scalar to a color map.
        bg_scalar: Background color in [0, 1]. This is before converting
            the scalar to a color map.
        bg_color: Background color as RGBA
        resolution: Resolution of the final image.
        filename: When rendering, save the image as this file.
    """

    obj_name: str = None
    labeling_error: bool = False
    spherical: float = None
    bone_bend: Sequence[float] = None
    bone_rotation: Sequence[float] = None
    obj_incline: int = None
    obj_rotation: float = None
    flip: bool = None
    position: float = None
    arm_position: float = None
    obj_color: RGBAColor = None
    obj_scalar: float = None
    bg_scalar: float = None
    bg_color: RGBAColor = None
    resolution: Tuple[int, int] = (128, 128)
    filename: Optional[str] = None

    VALID_VALUES = {
        'spherical': (0, 1),
        'bone_bend': HALF_CIRCLE,
        'bone_rotation': HALF_CIRCLE,
        'obj_name': set(['sticky', 'stretchy']),
        'labeling_error': set([False, True]),
        'obj_incline': HALF_CIRCLE,
        'obj_rotation': HALF_CIRCLE,
        'flip': set([True, False]),
        'position': (-0.5, 0.5),
        'obj_scalar': (0, 1),
        'bg_scalar': (0, 1),
    }

    @classmethod
    def _is_allowed_value(cls, value: Any, name: str) -> bool:
        """Checks if values are in the allowed value ranges."""
        valid = cls.VALID_VALUES[name]
        if type(valid) == tuple:
            vmin, vmax = valid

            return vmin <= value <= vmax
        elif type(valid) == set:
            return value in valid
        else:
            raise ValueError(f"Unknown valid value description: {valid}")

    def check_values(self):
        """Raises a ValueError if a value is not in its valid range."""
        def supports_iteration(value: Union[Any, Sequence[Any]]) -> bool:
            try:
                for _ in value:
                    return True
            except TypeError:
                return False

        for name, valid in self.VALID_VALUES.items():
            value = getattr(self, name)
            # value can be either a single value or multiple
            if type(value) == str or not supports_iteration(value):
                # test for a single value
                if not self._is_allowed_value(value, name):
                    raise ValueError(f"Attribute {name} has value {value} but "
                                     f"valid values would be in {valid}.")
            else:
                # test each value individually
                for i, item in enumerate(value):
                    if not self._is_allowed_value(item, name):
                        raise ValueError(
                            f"Attribute {name} has value {item}"" at position"
                            f" {i} but valid values would be in {valid}.")

    def __post_init__(self):
        # convert possible lists to tuples
        if type(self.bg_color) == list:
            self.bg_color = tuple(self.bg_color)
        if type(self.obj_color) == list:
            self.obj_color = tuple(self.obj_color)
        if type(self.resolution) == list:
            self.resolution = tuple(self.resolution)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the object as python dict.

        If ``self.filename`` is not set, a unique filename will be
        selected (using uuid4).
        """
        if self.filename is None:
            self.filename = str(uuid.uuid4()) + ".png"
        return dataclasses.asdict(self)

    def __str__(self) -> str:
        """Returns the object as a string."""
        pp = pprint.PrettyPrinter(indent=2)
        return self.__class__.__name__ + pp.pformat(self.__dict__)

    def clone(self, discard_filename: bool = True) -> SceneParameters:
        """Returns a deep copy.

        Args:
            discard_filename: Resets the filename of the copy.
        """
        clone = copy.deepcopy(self)
        if discard_filename and hasattr(clone, 'filename'):
            clone.filename = None
        return clone

    @property
    def obj_name_with_label_error(self) -> str:
        """Returns the object name taking into account the label error."""
        flip_obj_name = {
            'sticky': 'stretchy',
            'stretchy': 'sticky',
        }
        return {
            False: self.obj_name,
            True: flip_obj_name[self.obj_name]
        }[self.labeling_error]


Continouos = Union[scipy.stats.rv_continuous, Callable[[], float]]
Discrete = Union[scipy.stats.rv_discrete, Callable[[], float]]


@dataclasses.dataclass()
class SampleSceneParameters:
    """Samples the parameters of the ``SceneParameters`` objects.

    To implement biases, you can inhirent this class and modify how individual
    attributes are sample, e.g introducing addtional dependencies.

    For the valid values ranges, see ``SceneParameters.VALID_VALUES``.

    Attrs:
        bg_color_map: used color map for the background.
        spherical: distribution of ``SceneParameters.spherical``.
        bone_bend: distribution of ``SceneParameters.bone_bend``.
        bone_rotation: distribution of ``SceneParameters.bone_rotation``.
        obj_name: distribution of ``SceneParameters.obj_name``.
        arm_position: distribution of ``SceneParameters.arm_position``.
        labeling_error: distribution of ``SceneParameters.labeling_error``.
        obj_incline: distribution of ``SceneParameters.obj_incline``.
        obj_rotation:distribution of ``SceneParameters.obj_rotation``.
        flip: distribution of ``SceneParameters.flip``.
        position: distribution of ``SceneParameters.position``.
        obj_color: distribution of ``SceneParameters.obj_color``.
        bg_color: distribution of ``SceneParameters.bg_color``.
    """

    obj_name: Discrete = utils.discrete({'sticky': 0.5, 'stretchy': 0.5})
    spherical: Continouos = scipy.stats.beta(0.3, 0.3)
    bone_bend: Continouos = utils.truncated_normal(0, 0.1 * np.pi / 4, *HALF_CIRCLE)
    bone_rotation: Continouos = utils.truncated_normal(0, 0.1 * np.pi / 4, *HALF_CIRCLE)

    arm_position: Continouos = utils.truncated_normal(
        mean=0, std=0.40, lower=0, upper=0.65)
    labeling_error: Discrete = utils.discrete({True: 0.05, False: 0.95})
    obj_incline: Continouos = utils.truncated_normal(0, 0.03 * np.pi / 4, *HALF_CIRCLE)
    obj_rotation: Continouos = utils.truncated_normal(0, 0.3 * np.pi / 4, *HALF_CIRCLE)
    flip: Discrete = utils.discrete({True: 0., False: 1.})
    position: Continouos = scipy.stats.uniform(-0.5, 0.5)
    obj_color: Continouos = scipy.stats.uniform(0., 1.)
    bg_color: Continouos = scipy.stats.uniform(0.05, 0.80)
    bg_color_map: str = 'binary'

    def sample(self) -> SceneParameters:
        """Returns a new SceneParameters with random values.

        If create your own biased sampled dataset by inhirenting, from this
        class you might want to change the order the attributes are set.
        For example, if you want that ``obj_rotation`` should depends on the
        ``arm_position`` than you should also sample the ``arm_position`` first.
        """
        params = SceneParameters()
        self.sample_obj_name(params)
        self.sample_spherical(params)
        self.sample_bone_bend(params)
        self.sample_bone_rotation(params)
        self.sample_obj_incline(params)
        self.sample_obj_rotation(params)
        self.sample_flip(params)
        self.sample_position(params)
        self.sample_arm_position(params)
        self.sample_obj_color(params)
        self.sample_bg_color(params)
        return params

    def sample_obj_name(self, params: SceneParameters):
        """Samples the ``obj_name``."""
        params.obj_name = self.obj_name.rvs()

    def sample_labeling_error(self, params: SceneParameters):
        """Samples the ``labeling_error``."""
        params.labeling_error = float(self.labeling_error.rvs())

    def sample_spherical(self, params: SceneParameters):
        """Samples the ``spherical``."""
        params.spherical = float(self.spherical.rvs())

    def sample_bone_bend(self, params: SceneParameters):
        """Samples the ``bone_bend``."""
        params.bone_bend = self.bone_bend.rvs(size=7).tolist()

    def sample_bone_rotation(self, params: SceneParameters):
        """Samples the ``bone_rotation``."""
        params.bone_rotation = self.bone_rotation.rvs(size=7).tolist()

    def sample_obj_incline(self, params: SceneParameters):
        """Samples the ``obj_incline``."""
        params.obj_incline = self.obj_incline.rvs()

    def sample_obj_rotation(self, params: SceneParameters):
        """Samples the ``obj_rotation``."""
        params.obj_rotation = float(self.obj_rotation.rvs())

    def sample_flip(self, params: SceneParameters):
        """Samples the ``flip``."""
        params.flip = float(self.flip.rvs())

    def sample_position(self, params: SceneParameters):
        """Samples the ``position``."""
        params.position = self.position.rvs(2).tolist()

    def sample_arm_position(self, params: SceneParameters):
        """Samples the ``arm_position``."""
        arm_shift = float(self.arm_position.rvs())
        if params.obj_name == 'sticky':
            params.arm_position = arm_shift
        elif params.obj_name == 'stretchy':
            params.arm_position = 1 - arm_shift
        else:
            raise ValueError(f"Unknown `obj_name`: {params.obj_name}")

    def _object_cmap(self, params: SceneParameters) -> utils.ColorGenerator:
        return utils.ColorGenerator('seismic')

    def sample_obj_color(self, params: SceneParameters):
        """Samples the ``obj_color`` and ``obj_scalar``."""
        params.obj_scalar = float(self.obj_color.rvs())
        params.obj_color = tuple(self._object_cmap(params).get_color(params.obj_scalar))

    def _bg_cmap(self, params: SceneParameters) -> utils.ColorGenerator:
        return utils.ColorGenerator(self.bg_color_map)

    def sample_bg_color(self, params: SceneParameters):
        """Samples the ``bg_color`` and ``bg_scalar``."""
        params.bg_scalar = float(self.bg_color.rvs())
        params.bg_color = tuple(self._bg_cmap(params).get_color(params.bg_scalar))


class ColorBiasedSceneParameterSampler(SampleSceneParameters):
    """An example implementation of a color-biased SceneParameterSample.

    The color is sampled from a conditional distribution that is dependend on the object type.
    """

    def sample_obj_color(self, params: SceneParameters):
        """Samples the ``obj_color`` and ``obj_scalar`` with custom distributions."""
        if params.obj_name == 'sticky':
            color = utils.truncated_normal(1, 0.5, 0, 1).rvs()
        else:
            color = utils.truncated_normal(0, 0.5, 0, 1).rvs()
        params.obj_scalar = float(color)
        params.obj_color = tuple(self._object_cmap(params).get_color(color))
