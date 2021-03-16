import two4two
from two4two.plotvis import render_grid
import matplotlib.pyplot as plt
import numpy as np
# %% md
# We can create SceneParameters which will be initialize with default values
# %%
base_param = two4two.SceneParameters()

# %% md
# To create image from the SceneParameters we have to pass them to the blender Module.
# Its designed to recieve a list of params an return an enumeraor to recieve the num_images
# once they are finsihed rendering. Here we use list comprehension to recieve the single
# Scene Parameter
# %%
[plt.imshow(img) for (img, mask, param) in two4two.blender.render([base_param])]
# %% md
# The renderer also returns an image mask which we visualize here:
# %%
[plt.imshow(mask) for (img, mask, param) in two4two.blender.render([base_param])]
# %% md
# In this example we will render quite a few single images, which is why we are
# defining a helper function to do this.
# %%
def render_single_param(param: two4two.SceneParameters):
    [plt.imshow(img) for (img, mask, param) in two4two.blender.render([param])]

# %% md
# The default SceneParameters alwasy depicts a sticky.
# One can obtain the exact same set of default values with a convienience function
# %%
base_sticky = two4two.SceneParameters.default_sticky()
render_single_param(base_sticky)
# %% md
# Similarly a conveience function exisit to retrieve a stretchy
# %%
base_stretchy = two4two.SceneParameters.default_stretchy()
render_single_param(base_stretchy)
# %% md
# The attributes of the SceneParameters can be changed manually.
# For example the attribute **fliplr** indicates if the object is flipped vertically (left/right)
# %%
fliped_sticky = two4two.SceneParameters()
fliped_sticky.fliplr
# %% md
# Since the paramters indicate the object is not flipped, we can change that by
# setting the attribute manually accordingly.
# %%
fliped_sticky.fliplr = True
render_single_param(fliped_sticky)

# %% md
# Next lets look at the attribute of **roation**
# Here we are setting it to the minimum recomemnded value.
# %%
rotating_sticky = two4two.SceneParameters()
rotating_sticky.obj_rotation_pitch = two4two.SceneParameters.VALID_VALUES['obj_rotation_pitch'][0]
render_single_param(rotating_sticky)
# %% md
# After that we are setting it to the recommended maximum value
# %%
rotating_sticky.obj_rotation_pitch = two4two.SceneParameters.VALID_VALUES['obj_rotation_pitch'][1]
render_single_param(rotating_sticky)
# %% md
# It is possible to set attributes outside of the recomemnded values
# %%
rotating_sticky.obj_rotation_pitch = 1.2
render_single_param(rotating_sticky)
# %% md
# To check wheter values are with recommended ranges you can use *check_value*
# %%
rotating_sticky.check_values()
# %% md
# The following examples will illustrate the other attibutes and their corresponding
# maximum and minimum recommended values.
# The enxt examples shows the **inclination**
# %%
rotating_yaw_sticky = two4two.SceneParameters()
rotating_yaw_sticky.obj_rotation_yaw = two4two.SceneParameters.VALID_VALUES['obj_rotation_yaw'][0]
render_single_param(rotating_yaw_sticky)
# %%
rotating_yaw_sticky.obj_rotation_yaw = two4two.SceneParameters.VALID_VALUES['obj_rotation_yaw'][1]
render_single_param(rotating_yaw_sticky)
# %%
rotating_roll_sticky = two4two.SceneParameters()
rotating_roll_sticky.obj_rotation_roll = two4two.SceneParameters.VALID_VALUES['obj_rotation_roll'][0]
render_single_param(rotating_roll_sticky)
# %%
rotating_roll_sticky.obj_rotation_roll = two4two.SceneParameters.VALID_VALUES['obj_rotation_roll'][1]
render_single_param(rotating_yaw_sticky)
# %% md
# We can also alter the **postion** in the scene
# %%
right_down_sticky = two4two.SceneParameters()
right_down_sticky.position = tuple([two4two.SceneParameters.VALID_VALUES['position'][0]] * 2)
render_single_param(right_down_sticky)
# %% md
# The 8 building blocks of sticky and stretchy can be altered to be more or less **spherical**
# %%
spherical_sticky = two4two.SceneParameters()
spherical_sticky.spherical = two4two.SceneParameters.VALID_VALUES['spherical'][1]
render_single_param(spherical_sticky)
# %%
cubic_stretchy = two4two.SceneParameters.default_stretchy()
cubic_stretchy.spherical = two4two.SceneParameters.VALID_VALUES['spherical'][0]
render_single_param(cubic_stretchy)
# %% md
# *bone roation*
# %%f86c8ddf48463f0e56421fb9f8f42103c87f0adab3af3f61
bone_rotating_sticky = two4two.SceneParameters()
# #   #     #
# 1         5
# # 2 # 3 # 4 #
# 0         6
# #         #
bone_rotating_sticky.bone_rotation = (0, np.pi/4, np.pi/4, np.pi/4, np.pi/4, 0, 0)
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

# %% md
# Yo have now seen some atributes that can be changed about sticky and stretchy
# However, in practice we usally do not create SceneParameters manually.
# Instead we use a sampler to sample these attributes from given distribtions.
# %%
sampler = two4two.Sampler()
# %%md
# Here we use the default provided sample to generate an examples.
# Try rerunning the cell and see how it changes
# %%
sampled_params = sampler.sample()
render_single_param(sampled_params)
# We can create several examples using list comprehension,
# ramdomly creating several strechies and stickies.
# Also there is a useful helper function which renders these examples in an image grid.
# %%
params = [sampler.sample() for i in range(18)]
render_grid(params)

# %% md
# A sample works by setting attributes using a distributon
# We can also use a sampler to sample individual attributes of SceneParameters.
# This is usefull to visualize how each attribute is sampled.
# Here we are defining 18 default strechies and 18 default stickies to then
# onyl sampler their color. We then sort them by their color and visulize them in a grid.
# %%
num_images = 18
stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]

_ = [sampler.sample_obj_color(params) for params in stickies + strechies]
strechies.sort(key=lambda x: x.obj_color_scalar)
stickies.sort(key=lambda x: x.obj_color_scalar)
render_grid(stickies + strechies)
# %% md
# In the following example we repeat this experiement with a diffrent sampler, which has a known **color bias**.
# In the grid you can see that stickies (left) are more frequently red and stretchies (rigth) are more frequently blue.
# %%
sampler = two4two.ColorBiasedSampler()
_ = [sampler.sample_obj_color(params) for params in stickies + strechies]
strechies.sort(key=lambda x: x.obj_color_scalar)
stickies.sort(key=lambda x: x.obj_color_scalar)
render_grid(stickies + strechies)

# %% md
# It is much easier to see the color bias when we leave all other attributes constant and order the objects by their color.
# Lets see the images our ColorBiasedSampler would create when it is also sampling all other attributes.
# %%
render_grid([sampler.sample() for i in range(num_images*2)], equal_class_distribution=False)
# %% md
# There are two ways you can create your **custom samplers**.
# For simple changes you can set some custom distributions in a given sampler.
# Lets reuse the Colorbiases samples but now we change the sampler
# to randomly flip objects vertically 50% of the time.
# We are also sampeling the arm postion because a vertical flip is not really visible for
# stretchy otherwise.
# %%
sampler.fliplr=two4two.utils.discrete({True: 0.5, False: 0.5})
_ = [sampler.sample_fliplr(params) for params in stickies + strechies]
_ = [sampler.sample_arm_position(params) for params in stickies + strechies]
render_grid(stickies + strechies)
# %% md
# Now lets create our own bias. In the following example we take the default sampler and visualize how it is sampeling
# the background color.
# %%
stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = two4two.Sampler()
_ = [sampler.sample_bg_color(params) for params in stickies + strechies]
strechies.sort(key=lambda x: x.bg_color_scalar)
stickies.sort(key=lambda x: x.bg_color_scalar)
render_grid(stickies + strechies)
# %% md
# The changes in the background color are barely noticeable. But they are there.
# We will now replace the background disitrbution by a conditional disitrbution which is slightly diffrent for sticky and stretchy.
# %%
stickies = [two4two.SceneParameters.default_sticky() for i in range(num_images)]
strechies = [two4two.SceneParameters.default_stretchy() for i in range(num_images)]
sampler = two4two.Sampler()
sampler.bg_color = {
    'sticky': two4two.utils.truncated_normal(0.8, 0.425, 0.05, 0.8),
    'stretchy': two4two.utils.truncated_normal(0.05, 0.425, 0.055, 0.8)}
_ = [sampler.sample_bg_color(params) for params in stickies + strechies]
strechies.sort(key=lambda x: x.bg_color_scalar)
stickies.sort(key=lambda x: x.bg_color_scalar)
render_grid(stickies + strechies)

np.mean([x.bg_color_scalar for x in stickies])
np.mean([x.bg_color_scalar for x in strechies])


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
_ = [sampler.sample_bone_rotation(params) for params in stickies + strechies]
strechies.sort(key=lambda x: x.bone_rotation)
stickies.sort(key=lambda x: x.bone_rotation)
render_grid(stickies + strechies)
