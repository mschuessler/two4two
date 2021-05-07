"""This module contains the code to sample ``two4two.SceneParameters``."""

from __future__ import annotations

import dataclasses
from typing import Any, Callable, Dict, Optional, Union

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

    Attributes describe how the sampling is done. Concretely they provide the color maps for the
    object and the background and the distributors from which the value for the scene parameters are
    drawn.

    Distribution can be: * scipy-distribution from ``scipy.stats`` * callable functions returning a
    single value * a single (default) value. * a dictionary of all before-mentioned types containing
    the keys ``peaky``and ``stretchy``.

    These dictionaries are the easiest way to implement a bias. If you want an attribute to be
    sampled diffrently based on wheter it shows a peaky or stretchy, it is usually sufficient to
    change these dictionaries. See ``ColorBiasedSampler`` as an example.

    To implement more complex biases, you can inherit this class and modify how individual
    attributes are sampled, e.g., by introducing additional dependencies. Usually the best approach
    is to overwrite the sampling method (e.g. ``sample_obj_rotation_pitch``) and modify the sampling
    to be dependent on other attributes. Please be aware that you will then also need to implement
    interventional sampling, because in addition to sampling new parameters, we also want to
    controll an attribute sometimes. That means that we set the attribute to a specific value
    independent of the usual dependencies. If the intervention flag is true, the parameter should be
    sampled independent of any other attribute. For example, if the object color (obj_color)
    depends on the Peaky/Stretchy variable, it would need to be sampled independent
    if intervention = True.

    Since the default sampler implementation in this class is only dependent upon obj_name, so it is
    the only attribute considered in the intervention.

    For the valid values ranges, see ``SceneParameters.VALID_VALUES``.

    Attrs:
        bg_color_map: used color map for the background.
        obj_color_map: used color map for the object.
        spherical: distribution of ``SceneParameters.spherical``.
        bending: distribution of ``SceneParameters.bending``.
        obj_name: distribution of ``SceneParameters.obj_name``.
        arm_position: distribution of ``SceneParameters.arm_position_x`` and
            ``SceneParameters.arm_position_y``
        labeling_error: distribution of ``SceneParameters.labeling_error``.
        obj_rotation_roll: distribution of ``SceneParameters.obj_rotation_roll``.
        obj_rotation_pitch:distribution of ``SceneParameters.obj_rotation_pitch``.
        obj_rotation_yaw:distribution of ``SceneParameters.obj_rotation_pitch``.
        fliplr: distribution of ``SceneParameters.fliplr``.
        position: distribution of ``SceneParameters.position``.
        obj_color: distribution of ``SceneParameters.obj_color``.
        bg_color: distribution of ``SceneParameters.bg_color``.
    """

    obj_name: Discrete = utils.discrete({'peaky': 0.5, 'stretchy': 0.5})
    spherical: Continouos = scipy.stats.beta(0.3, 0.3)
    bending: Continouos = utils.truncated_normal(0, np.pi / 20, - np.pi / 10, np.pi / 10)
    arm_position: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(mean=0.5, std=0.2, lower=0, upper=0.52),
            'stretchy': utils.truncated_normal(mean=0.5, std=0.2, lower=0.48, upper=1.0)
        })
    labeling_error: Discrete = utils.discrete({True: 0., False: 1.})
    obj_rotation_roll: Continouos = utils.truncated_normal(0, 0.03 * np.pi / 4,
                                                           *utils.QUARTER_CIRCLE)
    obj_rotation_pitch: Continouos = utils.truncated_normal(0, 0.3 * np.pi / 4,
                                                            *utils.QUARTER_CIRCLE)
    obj_rotation_yaw: Continouos = scipy.stats.uniform(- np.pi, np.pi)
    fliplr: Discrete = utils.discrete({True: 0., False: 1.})
    position_x: Continouos = scipy.stats.uniform(-0.8, 0.8)
    position_y: Continouos = scipy.stats.uniform(-0.8, 0.8)
    obj_color: Continouos = scipy.stats.uniform(0., 1.)
    bg_color: Continouos = scipy.stats.uniform(0.05, 0.90)
    bg_color_map: str = 'coolwarm'
    obj_color_map: str = 'coolwarm'

    def sample(self, obj_name: Optional[str] = None) -> SceneParameters:
        """Returns a new SceneParameters with random values.

        If you create your own biased sampled dataset by inheriting from this class,
        you might want to change the order of how attributes are set.
        For example, if you want that ``obj_rotation_pitch`` should depend on the
        ``arm_position``then you should also sample the ``arm_position`` first.
        However, it is highly recommended to sample the object name first, as
        the sampling of the attribute might be dependent on the label
        (see the explanation of distributions in class description)

        Attrs:
            obj_name: Overides the sampled obj_name with the given namen. Usally only useful for
                manual sampling. Not recommeded when samplign larger sets.
        """
        params = SceneParameters()
        self.sample_obj_name(params)
        # The name flag allows to overide the sampling result. The sampling is still executed to
        # trigger any custom functionality that might be implented in subclasses.
        if obj_name and params.obj_name != obj_name:
            params.obj_name = obj_name

        self.sample_arm_position(params)
        self.sample_labeling_error(params)
        self.sample_spherical(params)
        self.sample_bending(params)
        self.sample_rotation(params)
        self.sample_fliplr(params)
        self.sample_position(params)
        self.sample_color(params)
        params.check_values()
        return params

    @staticmethod
    def _sample(obj_name: Optional[str], dist: Distribution, size: int = 1) -> Any:
        """Samples values from the distributon according to its type.

        The default number of values sampled is one, which can be changed with flag size.

        Distribution can be:
        * scipy-distribution from ``scipy.stats``
        * callable functions returning a single value
        * a single (default) value.
        * a dictionary of all before-mentioned types containing the keys ``peaky``and ``stretchy``.

        Will unpack np.ndarray, list, or tuple with a single element returned by distribution.

        """

        if size > 1:
            return tuple([Sampler._sample(obj_name, dist) for i in range(0, size)])

        if isinstance(dist, dict):
            # Rare edge case: If a dictionary was passed without the obj_name key
            # then the first distribution from the dictionary is used.
            if obj_name is None:
                dist = next(iter(dist.values()))
            else:
                dist = dist[obj_name]

        if hasattr(dist, 'rvs'):
            value = dist.rvs()  # type: ignore
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
            value = utils.numpy_to_python_scalar(value)

        return value

    @staticmethod
    def _sample_truncated(
        obj_name: Optional[str],
        dist: Distribution,
        size: int = 1,
        min: float = float(-np.inf),
        max: float = float(np.inf),
    ) -> Any:
        assert size == 1

        value = Sampler._sample(obj_name, dist, size)
        while not (min <= value <= max):
            value = Sampler._sample(obj_name, dist, size)
        return value

    def _sample_name(self) -> str:
        """Convienience function. Returns a sampled obj_name."""
        # obj_name is set to none, because the sampleing of the name should be, per definitiion,
        # idenpendet of the obj_name
        return self._sample(obj_name=None, dist=self.obj_name)

    def sample_obj_name(self, params: SceneParameters):
        """Samples the ``obj_name``."""
        params.obj_name = self._sample(None, self.obj_name)
        params.mark_sampled('obj_name')

    def sample_labeling_error(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``labeling_error``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.labeling_error = self._sample(obj_name, self.labeling_error)
        params.mark_sampled('labeling_error')

    def sample_spherical(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.spherical = self._sample(obj_name, self.spherical)
        params.mark_sampled('spherical')

    def sample_bending(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``bending``.

        Attrs:
            params: SceneParameters for which the bone roation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.bending = self._sample(obj_name, self.bending)
        params.mark_sampled('bending')

    def sample_rotation(self, params: SceneParameters, intervention: bool = False):
        """Convienience function bundeling all object rotation functions by calling them.

        Attrs:
            params: SceneParameters for which the object inclination is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_obj_rotation_roll(params, intervention=intervention)
        self.sample_obj_rotation_pitch(params, intervention=intervention)
        self.sample_obj_rotation_yaw(params, intervention=intervention)

    def sample_obj_rotation_roll(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_roll``.

        Attrs:
            params: SceneParameters for which the object inclination is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_roll = self._sample(obj_name, self.obj_rotation_roll)
        params.mark_sampled('obj_rotation_roll')

    def sample_obj_rotation_pitch(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_pitch``.

        Attrs:
            params: SceneParameters for which the rotation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_pitch = self._sample(obj_name, self.obj_rotation_pitch)
        params.mark_sampled('obj_rotation_pitch')

    def sample_obj_rotation_yaw(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_rotation_yaw``.

        Attrs:
            params: SceneParameters for which the rotation is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_rotation_yaw = self._sample(obj_name, self.obj_rotation_yaw)
        params.mark_sampled('obj_rotation_yaw')

    def sample_fliplr(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``fliplr``.

        Attrs:
            params: SceneParameters for which the fliping (left/right) is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.fliplr = self._sample(obj_name, self.fliplr)
        params.mark_sampled('fliplr')

    def sample_position(self, params: SceneParameters, intervention: bool = False):
        """Convienience function calling ``sample_position_x`` and ``sample_position_y``.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_position_x(params, intervention=intervention)
        self.sample_position_y(params, intervention=intervention)

    def sample_position_x(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``position_x`` of the object.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.position_x = self._sample(obj_name, self.position_x)
        params.mark_sampled('position_x')

    def sample_position_y(self, params: SceneParameters, intervention: bool = False):
        """Samples ``position_y`` of the object.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.position_y = self._sample(obj_name, self.position_y)
        params.mark_sampled('position_y')

    def sample_arm_position(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``arm_position``.

        Attrs:
            params: SceneParameters for which the arm_position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.arm_position = float(self._sample(obj_name, self.arm_position))
        params.mark_sampled('arm_position')

    def _object_cmap(self, params: SceneParameters) -> mpl.colors.Colormap:
        return plt.get_cmap(self.obj_color_map)

    def sample_color(self, params: SceneParameters, intervention: bool = False):
        """Convienience function calling ``sample_obj_color`` and ``sample_bg_color``.

        Attrs:
            params: SceneParameters for which the position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        self.sample_obj_color(params, intervention=intervention)
        self.sample_bg_color(params, intervention=intervention)

    def sample_obj_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.obj_color = float(self._sample(obj_name, self.obj_color))
        params.obj_color_rgba = tuple(self._object_cmap(params)(params.obj_color))  # type: ignore
        params.mark_sampled('obj_color')

    def _bg_cmap(self, params: SceneParameters) -> mpl.colors.Colormap:
        return plt.get_cmap(self.bg_color_map)

    def sample_bg_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``bg_color_rgba`` and ``bg_color``.

        Attrs:
            params: SceneParameters for which the labeling_error is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.bg_color = float(self._sample(obj_name, self.bg_color))
        params.bg_color_rgba = tuple(self._bg_cmap(params)(params.bg_color))  # type: ignore
        params.mark_sampled('bg_color')


@dataclasses.dataclass()
class SimpleColorMapSampler(Sampler):
    """An Sampler with a simpler colormap.

    This colormap allows for very simple experiements with human subjects. The simple colormap makes
    color biases easier to spot.
    """
    bg_color: Continouos = scipy.stats.uniform(0.05, 0.80)
    bg_color_map: str = 'binary'
    obj_color_map: str = 'seismic'


@dataclasses.dataclass()
class ColorBiasedSampler(SimpleColorMapSampler):
    """An example implementation of a color-biased SceneParameterSample.

    The color is sampled from a conditional distribution that is dependent on the object type.
    """

    obj_color: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy': utils.truncated_normal(0, 0.5, 0, 1),
        })


@dataclasses.dataclass()
class HighVariationSampler(Sampler):
    """A sampler producing more challenging images.

    This sampler allows for a higher variation in rotations and bending. Hence it creates a more
    challenging datset.
    """

    obj_rotation_roll: Continouos = scipy.stats.uniform(- np.pi / 3, 2 * np.pi / 3)
    obj_rotation_yaw: Continouos = scipy.stats.uniform(- np.pi, np.pi)
    obj_rotation_pitch: Continouos = scipy.stats.uniform(- np.pi / 3, 2 * np.pi / 3)
    bending: Continouos = scipy.stats.uniform(- np.pi / 8, np.pi / 4)


@dataclasses.dataclass()
class HighVariationColorBiasedSampler(HighVariationSampler):
    """A sampler producing more challenging images with a color bias that is depent on obj_name.

    This sampler allows for a higher variation in rotations and bending. Hence it creates a more
    challenging datset. This dataset is more challenging. So the bias is more likely to be used.
    """
    obj_color: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy': utils.truncated_normal(0, 0.5, 0, 1),
        })


@dataclasses.dataclass()
class MedVarColorSampler(Sampler):
    """A sampler with a more sophisticated color bias.

    The sample introduces a color bias only for the challenging cases where the arm position is hard
    to distinguish. Therefore, the bias is not evident in every image but informative enough to bias
    a model.
    """

    obj_color: Continouos = dataclasses.field(
        default_factory=lambda: {
            'peaky': utils.truncated_normal(1, 0.5, 0, 1),
            'stretchy': utils.truncated_normal(0, 0.5, 0, 1),
            'peaky_edge': utils.truncated_normal(1, 0.1, 0.7, 1),
            'stretchy_edge': utils.truncated_normal(0, 0.1, 0, 0.3),
        })

    def sample_obj_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        if params.arm_position > 0.45 and params.arm_position < 0.55:
            obj_name = obj_name + "_edge"
        params.obj_color = float(self._sample(obj_name, self.obj_color))
        params.obj_color_rgba = tuple(self._object_cmap(params)(params.obj_color))  # type: ignore
        params.mark_sampled('obj_color')


@dataclasses.dataclass()
class MedVarSampler(Sampler):
    """A sampler with a custom obj_color sampler that has no bias.

    This sampler is the base class for ``MedVarColorSampler'' it uses interventional sampleing to
    avoid introducing a bias.
    """

    def sample_obj_color(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``obj_color`` and ``obj_color_rgba``.

        Attrs:
            params: SceneParameters for which the obj_color is sampled and updated in place.
            intervention: Flag for interventional sampling. False will be ignored for this class.
        """
        # Since the color should be independent of the class we use interventional sampeling.
        super().sample_obj_color(params, intervention=True)


@dataclasses.dataclass()
class MedVarSpherSampler(MedVarSampler):
    """A sampler based on MedVar but with a Spherical bias.

    The sample introduces a spherical bias, but only for the cases that are not challing.
    Since this bias is not informative for all cases another bias can be added that will be used by
    the model if it provides information on the challenging cases.
    """

    def sample_spherical(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``spherical``..

        Attrs:
            params: SceneParameters for which the spherical attribute is sampled and updated.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        obj_name = self._sample_name() if intervention else params.obj_name
        params.spherical = self._sample(obj_name, self.spherical)

        if params.arm_position < 0.45:
            params.spherical = self._sample_truncated(
                obj_name, self.spherical, max=0.5)

        if params.arm_position > 0.55:
            params.spherical = self._sample_truncated(
                obj_name, self.spherical, min=0.5)
        params.mark_sampled('spherical')


@dataclasses.dataclass()
class MedVarSpherColorSampler(MedVarColorSampler, MedVarSpherSampler):
    """A sampler that combines the biases of ``MedVarSpherSampler'' and ``MedVarColorSampler''.

    See the the other two classes for documentation.
    """
    pass


@dataclasses.dataclass()
class MedVarNoArmsSampler(MedVarColorSampler):
    """A sampler based on MedVar with spherical and color bias but no arm information.

    This sampler uses interventional sampeling for the arm positon. The sampler is intended to
    produce only validation and test data.
    """
    def sample_arm_position(self, params: SceneParameters, intervention: bool = False):
        """Samples the ``arm_position``.

        Attrs:
            params: SceneParameters for which the arm_position is sampled and updated in place.
            intervention: Flag whether interventional sampling is applied. Details: see class docu.
        """
        super().sample_arm_position(params, intervention=True)
