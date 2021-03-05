from typing import Callable, List

from two4two import scene_parameters
from two4two import utils
from two4two.plotvis import render_grid

# Better than sampeling: initlilase with default
# my_scene = scene_parameters.SceneParameters()

sampler = scene_parameters.SampleSceneParameters()
sampler.obj_name = 'sticky'

base_param = sampler.sample()


def resample_clones(samplerFunction:
                    Callable[[scene_parameters.SceneParameters], scene_parameters.SceneParameters],
                    param: scene_parameters.SceneParameters, num_clones: int = 5
                    ) -> List[scene_parameters.SceneParameters]:
    clones = []
    for i in range(num_clones):
        clone = param.clone()
        samplerFunction(clone)
        clone.check_values()
        clones.append(clone)
    return clones

var_color_params = resample_clones(sampler.sample_obj_color, base_param, num_clones = 10)

render_grid(var_color_params, equal_class_distribution=False)
