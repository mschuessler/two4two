import bpy
import json
import uuid
import numpy as np
import os, sys

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)

from two4two.parameters import Parameters
from two4two.data_generator import DataGenerator

def render(param_file, save_location, save_param_file):
    save_json = os.path.join(save_location, save_param_file)
    with open(save_json, mode='x') as fsave:
        with open(param_file) as fparam:
            for line in fparam.readlines():
                params = json.loads(line)
                parameters = Parameters()
                parameters.__dict__.update(params)

                if parameters.filename is None:
                    img_id = uuid.uuid1(np.random.randint(int(1e14)))
                    basename = "{}.png".format(img_id)
                    parameters.filename = basename
                
                parameters.save_parameters(fsave)

                scene = DataGenerator(parameters)
                render_name = os.path.join(save_location, parameters.filename)
                scene.render(render_name)

if __name__ == '__main__':
    render(sys.argv[-3], sys.argv[-2], sys.argv[-1])