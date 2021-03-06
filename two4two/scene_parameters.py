"""This module contains classes to sample and describes individual scenes."""

from __future__ import annotations

import copy
import dataclasses
import importlib
import os
import pprint
from typing import Any, Dict, List, Optional, Sequence, Tuple
import uuid

from two4two import utils


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
        obj_name: Object name (either ``"sticky"`` or ``"stretchy"``).
        labeling_error: If ``True``, the ``obj_name_with_label_error``, will
            return the flipped obj_name. The ``obj_name`` attribute itself will
            not change.
        spherical: For ``1``,  spherical objects. For ``0``, cubes.
            Can have values in-between.
        bone_bend: Bending of the individual bone segments.
        bone_rotation: Rotation of the individual bone segments.
        obj_incline: Rotation of the object around the Y axis.
        obj_rotation: Rotation of the object around the Z axis.
        fliplr: Wheter the image should be flipped left to right.
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
    # TODO: once #38 is done. describe the coordinate system in full detail.
    obj_name: str = None
    labeling_error: bool = False
    spherical: float = None
    bone_bend: Sequence[float] = None
    bone_rotation: Sequence[float] = None
    obj_incline: int = None
    obj_rotation: float = None
    fliplr: bool = None
    position: float = None
    arm_position: float = None
    obj_color: utils.RGBAColor = None
    obj_scalar: float = None
    bg_scalar: float = None
    bg_color: utils.RGBAColor = None
    resolution: Tuple[int, int] = (128, 128)
    filename: Optional[str] = None

    VALID_VALUES = {
        'spherical': (0, 1),
        'bone_bend': utils.HALF_CIRCLE,
        'bone_rotation': utils.HALF_CIRCLE,
        'obj_name': set(['sticky', 'stretchy']),
        'labeling_error': set([False, True]),
        'obj_incline': utils.HALF_CIRCLE,
        'obj_rotation': utils.HALF_CIRCLE,
        'fliplr': set([True, False]),
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
        if type(self.bg_color) == list:
            self.bg_color = tuple(self.bg_color)
        if type(self.obj_color) == list:
            self.obj_color = tuple(self.obj_color)
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

        module = importlib.import_module(module)
        cls = getattr(module, cls_name)
        return cls(**state)

    def state_dict(self) -> Dict[str, Any]:
        """Returns the object as python dict.

        If ``self.filename`` is not set, a unique filename will be
        selected (using uuid4).
        """
        if self.filename is None:
            self.filename = str(uuid.uuid4()) + ".png"
        state = dataclasses.asdict(self)
        state['__module__'] = type(self).__module__
        state['__name__'] = type(self).__qualname__
        return state

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
    def mask_filename(self) -> str:
        """The filename of the segmentation mask."""
        base, ext = os.path.splitext(self.filename)
        mask_fname = f"{base}_mask{ext}"
        return mask_fname

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


def split_sticky_stretchy(params: List[SceneParameters],
                          num_samples: int = None
                          ) -> Tuple[Sequence[SceneParameters], Sequence[SceneParameters]]:
    """Returns a tuple of SceneParameters split by their type (sticky or stretchy).

    Attrs:
        params: List of SceneParfameters to split by ``sticky`` and ``stretchy``
        num_samples: exact number of SceneParameters to select per class. None means all availabel.


    """
    return [p for p in params if p.obj_name == 'sticky'][:num_samples], \
        [p for p in params if p.obj_name == 'stretchy'][:num_samples]
