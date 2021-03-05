from typing import List

from two4two import scene_parameters
from two4two import utils
from two4two.plotvis import render_grid

my_scene = scene_parameters.SceneParameters()
my_scene.check_values()

sampler = scene_parameters.SampleSceneParameters()

base_param = sampler.sample()
base_param.check_values()
params = [base_param.clone() for i in range(5)]
[sampler.sample_spherical(param) for param in params]
render_grid(params,equal_class_distribution=False)
sorted(params,key=lambda p: p.spherical)
params = [base_param.clone() for i in range(5)]
[sampler.sample_obj_rotation(param) for param in params]
render_grid(params,equal_class_distribution=False)


params = range()


def resample_cloes(samplerFunction: [scene_parameters.Distribution],
param: scene_parameters.SceneParameters, num_clones: int = 5) -> List[scene_parameters.SceneParameters]:
    clones = []
    for i in range(num_clones):
        clone = param.clone()
        samplerFunction(clone)
        clones.append(clone)
    return clones
