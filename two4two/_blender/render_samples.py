import bpy
import json
import uuid
import numpy as np
import os, sys
from pathlib import Path

blend_dir = os.path.dirname(bpy.data.filepath)
package_base_dir = str(Path(__file__).parents[2])

if package_base_dir not in sys.path:
   sys.path.append(package_base_dir)

from two4two.scene_parameters import SceneParameters
from two4two._blender.data_generator import DataGenerator

def render(param_file, save_location):
    with open(param_file) as fparam:
        for line in fparam.readlines():
            params = SceneParameters(**json.loads(line))
            scene = DataGenerator(params)
            render_name = os.path.join(save_location, params.filename)
            scene.render(render_name)

if __name__ == '__main__':
    render(sys.argv[-2], sys.argv[-1])