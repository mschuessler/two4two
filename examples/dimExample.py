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

base_param = sampler.sample()
base_param.check_values()
params = [base_param.clone() for i in range(5)]
[sampler.sample_obj_rotation(param) for param in params]
render_grid(params,equal_class_distribution=False)
