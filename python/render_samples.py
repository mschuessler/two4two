import bpy
import json
import uuid
import numpy as np
import os, sys
from tqdm import tqdm

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from parameters import Parameters
from data_generator import DataGenerator

def render(param_file, save_location):
    save_json = os.path.join(save_location, 'params.json')
    with open(save_json, mode='x') as fsave:
        with open(param_file) as fparam:
            for line in tqdm(fparam.readlines()):
                params = json.loads(line)
            
                img_id = uuid.uuid1(np.random.randint(int(1e14)))
                basename = "{}.png".format(img_id)

                parameters = Parameters()
                parameters.__dict__.update(params)
                parameters.filename = basename
                parameters.save_parameters(fsave)
                
                scene = DataGenerator(parameters)
                render_name = os.path.join(save_location, basename)
                scene.render(render_name)
                

if __name__ == '__main__':
    render(sys.argv[-2], sys.argv[-1])