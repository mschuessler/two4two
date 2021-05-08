"""This module contains classes to sample and describes individual scenes."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import json
import math
import pprint
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union
import uuid

from two4two import utils


OBJ_NAME_TO_INT = {
    'peaky': 0,
    'stretchy': 1,
}
"""Mapping from SceneParameters.obj_name to an integer. Please use this
encoding convention if you train a binary classifier."""


@dataclasses.dataclass()
class SceneParameters:
    """All parameters need to render a single image / scene.

    See the ``SceneParameters.VALID_VALUES`` for valid value ranges for the
    invididual attributes. Intervals are encoded as tuples and categoricals
    as sets.

    Subclasses should also be a ``dataclasses.dataclass``. Any added attributes
    will be saved and exposed through the dataloaders. When saving the
    parameters with ``state_dict``, your subclasses will also be saved. The
    ``SceneParameters.load`` method will also load and instantiate your
    subclass.

    Attrs:
        obj_name: Object name (either ``"peaky"`` or ``"stretchy"``).
        labeling_error: If ``True``, the ``obj_name_with_label_error``, will
            return the flipped obj_name. The ``obj_name`` attribute itself will
            not change.
        spherical: For ``1``,  spherical objects. For ``0``, cubes.
            Can have values in-between.
        bending: Rotation of bone segments causing object to bend.
        obj_rotation_roll: Rotation of the object around the Y axis.
        obj_rotation_pitch: Rotation of the object around the Z axis.
        fliplr: Wheter the image should be flipped left to right.
        position_x: Position of the object on x-axis.
        position_y: Position of the object on y-axis.
        arm_position: Absolute arm positions.
        obj_color_rgba: Object color as RGBA
        obj_color: Object color in [0, 1]. This is before converting
            the scalar to a color map.
        bg_color: Background color in [0, 1]. This is before converting
            the scalar to a color map.
        bg_color_rgba: Background color as RGBA
        resolution: Resolution of the final image.
        id: UUID used for saving rendered image and mask as image file.
        original_id: The id of the original SceneParameters before cloning

    """
    # TODO: once #38 is done. describe the coordinate system in full detail.
    obj_name: str = 'peaky'
    labeling_error: bool = False
    spherical: float = 0.5
    bending: float = 0.0
    obj_rotation_roll: float = 0.0
    obj_rotation_pitch: float = 0.0
    obj_rotation_yaw: float = 0.0
    fliplr: bool = False
    position_x: float = 0.0
    position_y: float = 0.0
    arm_position: float = 0.0
    obj_color: float = 0.5
    # When passing 0.5 to the cmap 'seismic' the following color is obtained
    obj_color_rgba: utils.RGBAColor = (1.0, 0.9921568627450981, 0.9921568627450981, 1.0)
    bg_color: float = 0.45
    # When passing 0.45 to the cmap 'binary' the following color is obtained
    bg_color_rgba: utils.RGBAColor = (
        0.5490196078431373, 0.5490196078431373, 0.5490196078431373, 1.0)
    resolution: Tuple[int, int] = (128, 128)
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    original_id: Optional[str] = None
    _attributes_status: Dict[str, str] = dataclasses.field(
        repr=False,
        default_factory=lambda: {
            'obj_name': 'default',
            'labeling_error': 'default',
            'spherical': 'default',
            'bending': 'default',
            'obj_rotation_roll': 'default',
            'obj_rotation_pitch': 'default',
            'obj_rotation_yaw': 'default',
            'fliplr': 'default',
            'position_x': 'default',
            'position_y': 'default',
            'arm_position': 'default',
            'bg_color': 'default',
            'obj_color': 'default'
        })

    VALID_VALUES: ClassVar[dict[
        str, Union[
            tuple[float, float],
            set[bool],
            set[str],
        ]]] = {
        'spherical': (0., 1.),
        'arm_position': (0., 1.),
        'bending': (- math.pi / 8, math.pi / 8),
        'obj_name': set(['peaky', 'stretchy']),
        'labeling_error': set([False, True]),
        'obj_rotation_roll': (- math.pi / 3, math.pi / 3),
        'obj_rotation_pitch': (- math.pi / 3, math.pi / 3),
        'obj_rotation_yaw': utils.FULL_CIRCLE,
        'fliplr': set([True, False]),
        'position_x': (-3.0, 3.0),
        'position_y': (-3.0, 3.0),
        'obj_color': (0., 1.),
        'bg_color': (0., 1.),
    }

    @classmethod
    def default_peaky(cls) -> SceneParameters:
        """Creates SceneParameters with default values for peaky."""
        return cls()

    @classmethod
    def default_stretchy(cls) -> SceneParameters:
        """Creates SceneParameters with default values for stretchy."""
        params = cls()
        params.obj_name = 'stretchy'
        params.arm_position = 1
        return params

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
        for name, valid in self.VALID_VALUES.items():
            value = getattr(self, name)
            # value can be either a single value or multiple
            if type(value) == str or not utils.supports_iteration(value):
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
        if type(self.bg_color_rgba) == list:
            self.bg_color_rgba = tuple(self.bg_color_rgba)
        if type(self.obj_color_rgba) == list:
            self.obj_color_rgba = tuple(self.obj_color_rgba)
        if type(self.resolution) == list:
            self.resolution = tuple(self.resolution)

    @staticmethod
    def load(state: Dict[str, Any]) -> SceneParameters:
        """Load parameter class from saved state.

        If the ``state`` was saved from a subclass, it will load and initialize
        the correct subclass.
        """
        state = copy.copy(state)
        module = state.pop('__module__')
        cls_name = state.pop('__name__')
        state.pop('label', None)

        module = importlib.import_module(module)
        cls = getattr(module, cls_name)
        return cls(**state)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the object as python dict."""
        state = dataclasses.asdict(self)
        state['label'] = self.obj_name_with_label_error
        state['__module__'] = type(self).__module__
        state['__name__'] = type(self).__qualname__
        return state

    def __str__(self) -> str:
        """Returns the object as a string."""
        pp = pprint.PrettyPrinter(indent=2)

        return self.__class__.__name__ + pp.pformat(
            {k: self.__dict__[k] for k in self.__dict__.keys() if k != "_attributes_status"})

    def clone(self, create_new_id: bool = True) -> SceneParameters:
        """Returns a deep copy.

        Creating a clone of a clone will raise ``TypeError`` unless no new id is assigned.

        Args:
            create_new_id: Creates new UUID.

        """
        clone = copy.deepcopy(self)
        if create_new_id:
            if self.original_id is not None:
                # We are undecided about this contrain, might be reomved later
                raise ValueError("Creating a clone of a clone is not allowed")
            clone.original_id = clone.id
            clone.id = str(uuid.uuid4())

        return clone

    def is_clone_of(self, original: SceneParameters) -> bool:
        """Returns True if this parameters have been cloned form the given orignal.

        Args:
            original: Returns True is this parameter is a clone of the given original.

        """
        return original.id == self.original_id

    def is_cloned(self) -> bool:
        """Returns True if this parameters have been cloned."""
        return self.original_id is not None

    def get_status(self, attribute: str) -> str:
        """Returns the status default, custom ,sampled or resampled for an attribute.

        SceneParameters allows to track the status of attributes thant can be sampled.
        When setting or sampeling attributes externally the functions use `get_status`,
        `mark_custom` and `mark_sampled` should be used to make use of the tracking.
        The possible status are:
            default: attribute has been initialized to default value.
            custom: attribute has been set manually.
            sampled: attribute has been sampled by sampler.
            resampled: attribute has been sampled again.
        """
        if attribute not in self._attributes_status:
            raise ValueError(f"{attribute} is not a tracked attribute")
        return self._attributes_status[attribute]

    def mark_custom(self, attribute: str):
        """Updates the status of the attribute to `custom`.

        SceneParameters allows to track the status of attributes thant can be sampled.
        The status can be obtained with  `get_status`.
        """
        # Get status will throw type error of atribute is untracked
        self.get_status(attribute)
        self._attributes_status[attribute] = 'custom'

    def mark_sampled(self, attribute: str):
        """Updates the status of the attribute to sampled or resampled.

        SceneParameters allows to track the status of attributes than can be sampled.
        This function marks this attribute as `sampled`
        If the attribute was set to sampled before if will be set to `resampled`.
        The status can be obtained with  `get_status`.
        """
        # Get status will throw type error of atribute is untracked
        status = self.get_status(attribute)
        if status in ('default', 'custom'):
            self._attributes_status[attribute] = 'sampled'
        elif status == 'sampled':
            self._attributes_status[attribute] = 'resampled'
        elif status == 'resampled':
            pass
        else:
            raise ValueError(f"Expected status default, custom, sampled \
            or resampled, got {self._attributes_status[attribute]}")

    @property
    def filename(self) -> str:
        """The filename of the generated image."""
        return self.id + ".png"

    @property
    def mask_filename(self) -> str:
        """The filename of the segmentation mask."""
        return self.id + "_mask.png"

    @property
    def obj_name_with_label_error(self) -> str:
        """Returns the object name taking into account the label error."""
        flip_obj_name = {
            'peaky': 'stretchy',
            'stretchy': 'peaky',
        }
        return {
            False: self.obj_name,
            True: flip_obj_name[self.obj_name]
        }[self.labeling_error]

    @property
    def obj_name_as_int(self) -> int:
        """Returns the integer which encodes the ``obj_name``."""
        return OBJ_NAME_TO_INT[self.obj_name]


def load_jsonl(path: str) -> List[SceneParameters]:
    """Loads SceneParameters from jsonl file."""
    with open(path) as f:
        return [SceneParameters.load(json.loads(line))
                for line in f.readlines()]


def split_peaky_stretchy(params: List[SceneParameters],
                         num_samples: int = None
                         ) -> Tuple[List[SceneParameters],
                                    List[SceneParameters]]:
    """Returns a tuple of SceneParameters split by their type (peaky or stretchy).

    Attrs:
        params: List of SceneParfameters to split by ``peaky`` and ``stretchy``
        num_samples: exact number of SceneParameters to select per class. None means all availabel.


    """
    return ([p for p in params if p.obj_name == 'peaky'][:num_samples],
            [p for p in params if p.obj_name == 'stretchy'][:num_samples])
