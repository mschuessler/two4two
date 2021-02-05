import dataclasses
from typing import Sequence, Tuple, Optional, TextIO
import numpy as np
import copy
import pprint
import json
import uuid
from two4two.color_generator import ColorGenerator
from scipy.stats import truncnorm


@dataclasses.dataclass()
class SceneParameters:
    obj_name: str = None
    spherical: float = None
    bone_bend: Sequence[float] = None
    bone_rotation: Sequence[float] = None
    obj_incline: int = None
    obj_rotation: float = None
    flip: bool = None
    position: float = None
    arm_position: float = None
    obj_color: float = None
    obj_scalar: Sequence[float] = None
    bg_scalar: float = None
    bg_color: Sequence[float] = None
    resolution: Tuple[int, int] = (128, 128)
    filename: Optional[str] = None

    def state_dict(self):
        if self.filename is None:
            self.filename = str(uuid.uuid4()) + ".png"
        return dataclasses.asdict(self)

    def __str__(self):
        pp = pprint.PrettyPrinter(indent=2)
        return self.__class__.__name__ + pp.pformat(self.__dict__)

    def clone(self, discard_filename=True):
        clone = copy.deepcopy(self)
        if discard_filename and hasattr(clone, 'filename'):
            del clone.__dict__['filename']
        return clone


@dataclasses.dataclass()
class SampleSceneParameters:
    bg_color_map: str = 'gray'
    spherical: Tuple[float, float] = (0.3, 0.3)
    bone_bend: float = 0.3
    bone_rotation: float = 0.2
    obj_type: float = 0.5
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

    def sample_spherical(self, params: SceneParameters):
        params.spherical = float(np.random.beta(*self.spherical))

    def sample_bone_bend(self, params: SceneParameters):
        params.bone_bend = np.random.uniform(
            -self.bone_bend * np.pi/4,
            self.bone_bend * np.pi/4, size=7).tolist()

    def sample_bone_rotation(self, params: SceneParameters):
        params.bone_rotation = np.random.uniform(
            -self.bone_rotation * np.pi/4,
            self.bone_rotation * np.pi/4, size=7).tolist()

    def sample_obj_incline(self, params: SceneParameters):
        params.obj_incline = float(
            self.obj_incline * np.random.uniform(-np.pi/4, np.pi/4))

    def sample_obj_rotation(self, params: SceneParameters):
        params.obj_rotation = float(
            self.obj_rotation * np.random.uniform(-np.pi/4, np.pi/4))

    def sample_flip(self, params: SceneParameters):
        params.flip = bool(np.random.uniform() < self.flip)

    def sample_position(self, params: SceneParameters):
        params.position = np.random.uniform(-0.5, 0.5, size=2).tolist()

    def sample_arm_position(self, params: SceneParameters):
        def get_truncated_normal(mean=0, sd=1, low=0, upp=10):
            return truncnorm((low - mean) / sd, (upp - mean) / sd, loc=mean, scale=sd)

        rv = get_truncated_normal(mean=0, sd=0.40, low=0, upp=0.65)
        arm_shift = float(rv.rvs())
        if params.obj_name == 'sticky':
            params.arm_position = arm_shift
        elif params.obj_name == 'stretchy':
            params.arm_position = 1 - arm_shift
        else:
            raise ValueError(f"Unknown `obj_name`: {params.obj_name}")

    def object_cmap(self, params: SceneParameters) -> ColorGenerator:
        return ColorGenerator('seismic')

    def sample_obj_color(self, params: SceneParameters):
        params.obj_scalar = float(np.random.uniform(*self.obj_color))
        params.obj_color = self.object_cmap(params).get_color(params.obj_scalar)

    def bg_cmap(self, params: SceneParameters) -> ColorGenerator:
        return ColorGenerator('binary')

    def sample_bg_color(self, params: SceneParameters):
        params.bg_scalar = float(np.random.uniform(*self.bg_color))
        params.bg_color = self.bg_cmap(params).get_color(params.bg_scalar)
