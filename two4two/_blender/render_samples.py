"""render sample comand line tool.

This is invoked interally by ``two4two.blender.render``.
"""


import json
import os
from pathlib import Path
import sys

import bpy

# the two4two package is not visible for the blender python.
# we therfore add the package directory to the path.
blend_dir = os.path.dirname(bpy.data.filepath)
package_base_dir = str(Path(__file__).parents[2])

if package_base_dir not in sys.path:
    sys.path.append(package_base_dir)

from two4two._blender.scene import Scene  # noqa: E402
from two4two.scene_parameters import SceneParameters  # noqa: E402


def _render_files(param_file: str, save_location: str, save_blender_file: str):
    with open(param_file) as fparam:
        for line in fparam.readlines():
            params = SceneParameters.load(json.loads(line))
            scene = Scene(params)
            image_fname = os.path.join(save_location, params.filename)

            mask_fname = os.path.join(save_location, params.mask_filename)
            scene.render(image_fname, mask_fname, )

            if save_blender_file == "True":
                scene.save_blender_file(
                    os.path.join(save_location, f"{params.id}.blender"))


if __name__ == '__main__':
    try:
        # starts coverage if tests are running
        # otherwise nothing happens
        import coverage
        coverage.process_startup()
    except ImportError:
        pass
    _render_files(sys.argv[-3], sys.argv[-2], sys.argv[-1])
