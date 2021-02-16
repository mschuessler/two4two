"""This module contains classes to sample and describes individual scenes."""

import copy
import dataclasses
import pprint
from typing import Any, Dict, Optional, Self, Sequence, Tuple
import uuid
from two4two import utils
from scipy.stats import truncnorm

from two4two.color_generator import ColorGenerator

RGBAColor = Tuple[float, float, float, float]


@dataclasses.dataclass()
class SceneParameters:
    """All parameters need to render a single image / scene.

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

    def clone(self, discard_filename: bool = True) -> Self:
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


@dataclasses.dataclass()
class SampleSceneParameters:
    bg_color_map: str = 'gray'
    spherical: Tuple[float, float] = (0.3, 0.3)
    bone_bend: float = 0.3
    bone_rotation: float = 0.2
    obj_type: float = 0.5
    label_error_probability: float = 0.
    obj_incline: float = 0.1
    obj_rotation: float = 0.7
    flip: float = 0.
    position: Tuple[float, float] = (-0.5, 0.5)
    obj_color: Tuple[float, float] = (0, 1)
    bg_color: Tuple[float, float] = (0.05, 0.80)

    def sample(self) -> SceneParameters:
        params = SceneParameters()
        self.sample_obj_type(params)
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

    def sample_obj_type(self, params: SceneParameters):
        params.obj_name = ['sticky', 'stretchy'][int(np.random.uniform() > self.obj_type)]

    def sample_wrong_obj_name_label(self, params: SceneParameters):
        params.wrong_obj_name_label = bool(np.random.uniform() < self.label_error_probability)

    def sample_spherical(self, params: SceneParameters):
        params.spherical = float(np.random.beta(*self.spherical))

    def sample_bone_bend(self, params: SceneParameters):
        params.bone_bend = np.random.uniform(
            -self.bone_bend * np.pi / 4,
            self.bone_bend * np.pi / 4, size=7).tolist()

    def sample_bone_rotation(self, params: SceneParameters):
        params.bone_rotation = np.random.uniform(
            -self.bone_rotation * np.pi / 4,
            self.bone_rotation * np.pi / 4, size=7).tolist()

    def sample_obj_incline(self, params: SceneParameters):
        params.obj_incline = float(
            self.obj_incline * np.random.uniform(-np.pi / 4, np.pi / 4))

    def sample_obj_rotation(self, params: SceneParameters):
        params.obj_rotation = float(
            self.obj_rotation * np.random.uniform(-np.pi / 4, np.pi / 4))

    def sample_flip(self, params: SceneParameters):
        params.flip = bool(np.random.uniform() < self.flip)

    def sample_position(self, params: SceneParameters):
        params.position = np.random.uniform(-0.5, 0.5, size=2).tolist()

    def sample_arm_position(self, params: SceneParameters):
        def get_truncated_normal(mean: float = 0, sd: float = 1,
                                 low: float = 0, upp: float = 10):
            return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

        rv = get_truncated_normal(mean=0, sd=0.40, low=0, upp=0.65)
        arm_shift = float(rv.rvs())
        if params.real_obj_name == 'sticky':
            params.arm_position = arm_shift
        elif params.real_obj_name == 'stretchy':
            params.arm_position = 1 - arm_shift
        else:
            raise ValueError(f"Unknown `obj_name`: {params.obj_name}")

    def object_cmap(self, params: SceneParameters) -> utils.ColorGenerator:
        return utils.ColorGenerator('seismic')

    def sample_obj_color(self, params: SceneParameters):
        params.obj_scalar = float(np.random.uniform(*self.obj_color))
        params.obj_color = tuple(self.object_cmap(params).get_color(params.obj_scalar))

    def bg_cmap(self, params: SceneParameters) -> utils.ColorGenerator:
        return utils.ColorGenerator('binary')

    def sample_bg_color(self, params: SceneParameters):
        params.bg_scalar = float(np.random.uniform(*self.bg_color))
        params.bg_color = tuple(self.bg_cmap(params).get_color(params.bg_scalar))
