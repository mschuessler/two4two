import dataclasses
import two4two
from two4two.plotvis import render_grid
import matplotlib.pyplot as plt
import numpy as np

def render_single_param(param: two4two.SceneParameters):
    [plt.imshow(img) for (img, mask, param) in two4two.blender.render([param])]

# %% md
# *bone roation*
# %%f86c8ddf48463f0e56421fb9f8f42103c87f0adab3af3f61
bone_rotating_sticky = two4two.SceneParameters()
# #   #     #
# 1         5
# # 2 # 3 # 4 #
# 0         6
# #         #
bone_rotating_sticky.bending = (0, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0, 0)
bone_rotating_sticky.check_values()
render_single_param(bone_rotating_sticky)
# %%
bending_sticky = two4two.SceneParameters()
# #   #     #
# 1         5
# # 2 # 3 # 4 #
# 0         6
# #         #
bending_sticky.bone_bend = (0, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0, 0)
render_single_param(bending_sticky)
bending_sticky.bone_bend = (np.pi, 0, 0, 0, 0, 0, 0)
render_single_param(bending_sticky)


num_images = 18


# %% md
# **bone bend**
# %%
stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = two4two.Sampler()
_ = [sampler.sample_bone_bend(params) for params in stickies + strechies]
strechies.sort(key=lambda x: x.bone_bend)
stickies.sort(key=lambda x: x.bone_bend)
render_grid(stickies + strechies)

# %%
stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = two4two.Sampler()
_ = [sampler.sample_bending(params) for params in stickies + strechies]
strechies.sort(key=lambda x: x.bending)
stickies.sort(key=lambda x: x.bending)
render_grid(stickies + strechies)


@dataclasses.dataclass()
class OnlyRotateSampler(two4two.Sampler):
    """Very boring."""

    def sample_bone_bend(self, params: two4two.SceneParameters):
        """Samples the ``bone_bend``."""
        # do noting

    def sample_bending(self, params: two4two.SceneParameters):
        """Samples the ``bending``."""
        params.bending = tuple([self._sample(params.obj_name, self.bending)] * 7)
        params.mark_sampled('bending')

@dataclasses.dataclass()
class OnlyRotateSPineSampler(two4two.Sampler):
    """Very boring."""

    def sample_bone_bend(self, params: two4two.SceneParameters):
        """Samples the ``bone_bend``."""
        # do noting

    def sample_bending(self, params: two4two.SceneParameters):
        """Samples the ``bending``."""
        params.bending = tuple([0, 0] + [self._sample(params.obj_name, self.bending)] * 3 + [0, 0])
        params.mark_sampled('bending')


@dataclasses.dataclass()
class BoringBendingandRoatation2Sampler(two4two.Sampler):
    """Very boring."""

    def sample_bone_bend(self, params: two4two.SceneParameters):
        """Samples the ``bone_bend``."""
        params.bone_bend = tuple([0, 0] + [self._sample(params.obj_name, self.bone_bend)] * 3 + [0, 0])
        params.mark_sampled('bone_bend')

    def sample_bending(self, params: two4two.SceneParameters):
        """Samples the ``bending``."""
        params.bending = tuple([0, 0] + [self._sample(params.obj_name, self.bending)] * 3 + [0, 0])
        params.mark_sampled('bending')

@dataclasses.dataclass()
class ExtremlyBoringBendingandRoatationSampler(two4two.Sampler):
    """Very boring."""

    def sample_bone_bend(self, params: two4two.SceneParameters):
        """Samples the ``bone_bend``."""
        params.bone_bend = tuple([0, 0] + [self._sample(params.obj_name, self.bone_bend)] * 3 + [0, 0])
        params.bending = params.bone_bend
        params.mark_sampled('bone_bend')
        params.mark_sampled('bending')

    def sample_bending(self, params: two4two.SceneParameters):
        """Samples the ``bending``."""
        # do nothing - already set


@dataclasses.dataclass()
class BoringBendingandRoatationSampler(two4two.Sampler):
    """Very boring."""

    def sample_bone_bend(self, params: two4two.SceneParameters):
        """Samples the ``bone_bend``."""
        params.bone_bend = tuple([self._sample(params.obj_name, self.bone_bend)] * 7)
        params
        params.mark_sampled('bone_bend')

    def sample_bending(self, params: two4two.SceneParameters):
        """Samples the ``bending``."""
        params.bending = tuple([self._sample(params.obj_name, self.bending)] * 7)
        params.mark_sampled('bending')


stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = two4two.Sampler()
_ = [sampler.sample_bending(params) for params in stickies + strechies]
_ = [sampler.sample_bone_bend(params) for params in stickies + strechies]
render_grid(stickies + strechies)

stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = BoringBendingandRoatationSampler()
_ = [sampler.sample_bending(params) for params in stickies + strechies]
_ = [sampler.sample_bone_bend(params) for params in stickies + strechies]
render_grid(stickies + strechies)


stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = BoringBendingandRoatation2Sampler()
_ = [sampler.sample_bending(params) for params in stickies + strechies]
_ = [sampler.sample_bone_bend(params) for params in stickies + strechies]
render_grid(stickies + strechies)


stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = ExtremlyBoringBendingandRoatationSampler()
#_ = [sampler.sample_bending(params) for params in stickies + strechies]
_ = [sampler.sample_bone_bend(params) for params in stickies + strechies]
render_grid(stickies + strechies)

stickies + strechies
