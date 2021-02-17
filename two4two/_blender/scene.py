"""Rendering of the blender scene."""

import os
import sys
from typing import Sequence

import bpy
import numpy as np

from two4two import scene_parameters
from two4two._blender import butils
from two4two._blender.blender_object import BlenderObject


class Scene():
    """A Blender scene.

    Args:
        parameters: SceneParameters
    """

    def _set_pose(self,
                  bend_angles: Sequence[float],
                  rotation_angles: Sequence[float]):
        self.obj.set_pose(bend_angles, rotation_angles)
        self.obj.center()

    def _set_rotation(self,
                      incline: float,
                      rotation: float,
                      flipped: bool):
        self.obj.rotate(incline, 'Y')
        self.obj.rotate(rotation, 'Z')
        if flipped:
            self._flip_orientation()
        self.obj.center()

    def _flip_orientation(self):
        self.obj.rotate(np.pi, 'Z')

    def _set_position(self, x: float, y: float):
        self.obj.move((0, x, y))

    def _setup_scene(
        self,
        background_color: scene_parameters.RGBAColor,
        light_size: float = 20.
    ):
        scene = bpy.context.scene

        bpy.ops.object.light_add(type='AREA',
                                 location=(10, 10, 10))
        bpy.context.active_object.name = 'light'
        light = bpy.data.objects['light']
        light.data.shape = 'ELLIPSE'
        light.data.size = light_size
        light.data.size_y = 15
        light.data.energy = 5000 / 20 * light_size * 4
        light.rotation_euler = (np.pi / 4, np.pi / 4, np.pi / 2)

        bpy.ops.object.camera_add(location=(9, 0, 0),
                                  rotation=(np.pi / 2, 0, np.pi / 2))
        scene.camera = bpy.context.object

        bpy.ops.mesh.primitive_plane_add(size=20,
                                         enter_editmode=False,
                                         location=(self.obj.boundaries[0][0] - 10, 0, 0),
                                         rotation=(0, np.pi / 2, 0))
        bpy.context.active_object.name = 'background'
        background = bpy.context.view_layer.objects.active
        mat = bpy.data.materials.new(name='background_material')
        background.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = background_color

    def color_code_object(self):
        """Color code objects to create pixelwise segmentation masks."""
        colors = [
            (1.0, 0.06, 0.0, 1),
            (0.9399999999999997, 1.0, 0.0, 1),
            (0.0, 1.0, 0.06000000000000005, 1),
            (0.0, 0.9399999999999997, 1.0, 1),
            (0.06000000000000005, 0.0, 1.0, 1),
            (1.0, 0.0, 0.9399999999999997, 1)]
        black = (0, 0, 0, 1)
        blue = (0, 0, 1, 1)

        butils.select_object()
        butils.edit_mode()
        bpy.ops.mesh.separate(type='LOOSE')
        butils.object_mode()

        # TODO(leon) for loop
        active_object = butils.set_active('object.003')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[0]

        active_object = butils.set_active('object.004')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[2]

        active_object = butils.set_active('object.005')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[4]

        active_object = butils.set_active('object')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = blue

        active_object = butils.set_active('object.001')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = black

        active_object = butils.set_active('object.002')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[1]

        active_object = butils.set_active('object.006')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[3]

        active_object = butils.set_active('object.007')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[5]

    def render(self, filename: str):
        """Renders the scene and saves it at the given ``filename``."""
        bpy.data.scenes['Scene'].render.filepath = filename
        bpy.ops.render.render(write_still=True)

    def __init__(self,
                 parameters: scene_parameters.SceneParameters):
        butils.clear_all()
        self.obj = BlenderObject(parameters.obj_name,
                                 parameters.spherical,
                                 parameters.arm_position)

        self.obj.add_material(parameters.obj_color)

        blend_dir = os.path.dirname(bpy.data.filepath)
        if blend_dir not in sys.path:
            sys.path.append(blend_dir)

        self._set_pose(parameters.bone_bend,
                       parameters.bone_rotation)
        self._set_rotation(parameters.obj_incline,
                           parameters.obj_rotation,
                           parameters.flip)
        x, y = parameters.position
        self._set_position(x, y)
        self._setup_scene(parameters.bg_color)

        res_x, res_y = parameters.resolution
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.resolution_x = res_x
        bpy.context.scene.render.resolution_y = res_y
