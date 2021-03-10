import two4two
import matplotlib.pyplot as plt
import numpy as np

# We can create SceneParameters which will be initialize with default values
base_param = two4two.SceneParameters()


# TODO: add explanation here
[plt.imshow(img) for (img, mask, param) in two4two.blender.render([base_param])]
[plt.imshow(mask) for (img, mask, param) in two4two.blender.render([base_param])]

def render_single_param(param: two4two.SceneParameters):
    [plt.imshow(img) for (img, mask, param) in two4two.blender.render([param])]



# The default SceneParameters depicts a sticky
# One can obtain the exact same set of default values with
base_sticky = two4two.SceneParameters.default_sticky()
render_single_param(base_sticky)

base_stretchy = two4two.SceneParameters.default_stretchy()
render_single_param(base_stretchy)

rotating_sticky = two4two.SceneParameters()
rotating_sticky.obj_rotation = two4two.SceneParameters.VALID_VALUES['obj_rotation'][0]
render_single_param(rotating_sticky)
rotating_sticky.obj_rotation = two4two.SceneParameters.VALID_VALUES['obj_rotation'][1]
render_single_param(rotating_sticky)
inclined_sticky = two4two.SceneParameters()
inclined_sticky.obj_incline = two4two.SceneParameters.VALID_VALUES['obj_incline'][0]
render_single_param(inclined_sticky)
inclined_sticky.obj_incline = two4two.SceneParameters.VALID_VALUES['obj_incline'][1]
render_single_param(inclined_sticky)

fliped_sticky = two4two.SceneParameters()
fliped_sticky.fliplr = True
render_single_param(fliped_sticky)


spherical_sticky = two4two.SceneParameters()
spherical_sticky.spherical = two4two.SceneParameters.VALID_VALUES['spherical'][1]
render_single_param(spherical_sticky)
cubic_stretchy = two4two.SceneParameters.default_stretchy()
cubic_stretchy.spherical = two4two.SceneParameters.VALID_VALUES['spherical'][0]
render_single_param(cubic_stretchy)

bending_sticky = two4two.SceneParameters()
bending_sticky.bone_bend = (0, -np.pi/4, 0, 0, 0, 0, 0)
bending_sticky.check_values()
[plt.imshow(img) for (img, mask, param) in two4two.blender.render([bending_sticky])]
# However, we usally do not create SceneParameters manually and

[plt.imshow(img) for (img, mask, param) in two4two.blender.render([base_param])]
obj_color_clones = []
    for i in range(num_clones):
        clone = param.clone()
        samplerFunction(clone)
        clone.check_values()
        clones.append(clone)
    return clones
