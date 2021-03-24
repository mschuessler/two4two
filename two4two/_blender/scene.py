"""Rendering of the blender scene."""

import os
import sys
from typing import Sequence

import bpy
import imageio
import numpy as np
from skimage import color

from two4two import scene_parameters
from two4two import utils
from two4two._blender import blender_object
from two4two._blender import butils


class Scene():
    """A Blender scene.

    Args:
        parameters: SceneParameters
    """

    def _set_pose(self,
                  bending: float):
        self.obj.set_pose(blender_object.BoneRotation.from_bending(bending))
        self.obj.center()

    def _set_rotation(self,
                      yaw: float,
                      roll: float,
                      pitch: float,
                      ):
        # TODO(martin) add link to image when #89 is done

        self.obj.rotate(roll, 'Y')
        self.obj.rotate(pitch, 'Z')
        self.obj.rotate(yaw, 'X')
        self.obj.center()

    def _set_position(self, x: float, y: float):
        self.obj.move((0, x, y))

    def _setup_scene(
        self,
        background_color: utils.RGBAColor,
    ):
        light_size = 20.
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
        self.background = bpy.context.view_layer.objects.active
        mat = bpy.data.materials.new(name='background_material')
        self.background.data.materials.append(mat)
        self.background.active_material.diffuse_color = background_color

    def render_segmentation_mask(self, mask_filename: str):
        """Renders segmentation mask."""
        base, ext = os.path.splitext(mask_filename)
        emission_fname = base + "_emission" + ext
        try:
            color_hues = self._render_shadeless(
                [self.background] + list(self.obj.blocks.values()),
                emission_fname,
                restore=False,
            )

            emission_mask_rgba = imageio.imread(emission_fname)
            emission_mask = color.rgb2hsv(emission_mask_rgba[:, :, :3])
            distances = []
            for idx, hue in enumerate(color_hues):
                distances.append(
                    np.abs(emission_mask[:, :, 0] - hue)
                )
            segmentation_mask = np.argmin(np.stack(distances, axis=0), axis=0).astype(np.uint8)
            imageio.imwrite(mask_filename, segmentation_mask)
        finally:
            os.remove(emission_fname)

    def _render_shadeless(
        self,
        blender_objects: Sequence[bpy.types.Object],
        path: str,
        restore: bool = True,
    ) -> Sequence[float]:
        """Renders the scene without shades.

        For each object, a unique color is used.

        Args:
            blender_objects: list of blender objects to encode
            path: save rendered image at this location
            restore: restore blender scene. Useful for debugging.
        """
        render = bpy.context.scene.render
        cycles = bpy.context.scene.cycles

        old_filepath = render.filepath
        old_engine = render.engine
        old_cycles_filter_type = cycles.filter_type
        old_cycles_filter_width = cycles.filter_width
        old_cycles_samples = cycles.samples
        # old_use_antialiasing = render.use_antialiasing

        # Override some render settings to have flat shading
        render.filepath = path
        render.engine = 'CYCLES'

        # disable anti-aliasing

        cycles.filter_type = 'BLACKMAN_HARRIS'
        # ensures that each pixel is sampled at the same location
        cycles.filter_width = 0.01
        # one ray sample per pixel
        cycles.samples = 1

        n_objs = len(blender_objects)
        hues = [i / n_objs for i in range(n_objs)]

        object_colors = [color.hsv2rgb([hue, 1, 1]).tolist() + [1, ]
                         for hue in hues]
        old_materials = []
        for i, obj in enumerate(blender_objects):
            old_materials.append(obj.data.materials[0])

            # create new material
            bpy.ops.object.material_slot_add()
            bpy.ops.material.new()
            mat = bpy.data.materials["Material"]
            mat.name = f"material_{i}"
            mat.use_nodes = True
            nodes = mat.node_tree.nodes
            nodes.clear()
            # and use emission material
            nodes.new("ShaderNodeEmission")
            node_emission = nodes["Emission"]

            # create material output node
            node_output = nodes.new(type='ShaderNodeOutputMaterial')
            node_output.location = 400, 0
            # link emission to output node
            links = mat.node_tree.links
            links.new(node_emission.outputs[0], node_output.inputs[0])

            node_emission.inputs["Color"].default_value = object_colors[i]
            node_emission.inputs["Strength"].default_value = 0.05
            obj.data.materials[0] = mat

        # Render the scene
        bpy.ops.render.render(write_still=True)

        if restore:
            # Undo the above; first restore the materials to objects
            for mat, obj in zip(old_materials, blender_objects):
                obj.data.materials[0] = mat

            # restore settings
            render.filepath = old_filepath
            render.engine = old_engine
            cycles.filter_type = old_cycles_filter_type
            cycles.filter_width = old_cycles_filter_width
            cycles.samples = old_cycles_samples
        return hues

    def _fliplr_image(self, image_filename: str):
        """Overwrites the image with itself flipped left to right."""
        img = imageio.imread(image_filename)
        img_flip_lr = img[:, ::-1]
        imageio.imwrite(image_filename, img_flip_lr)

    def render(self,
               image_filename: str,
               mask_filename: str = None
               ):
        """Renders the scene and saves it at the given ``filename``."""
        bpy.data.scenes['Scene'].render.filepath = image_filename
        bpy.ops.render.render(write_still=True)

        if self.parameters.fliplr:
            self._fliplr_image(image_filename)

        if mask_filename is not None:
            self.render_segmentation_mask(mask_filename)
            if self.parameters.fliplr:
                self._fliplr_image(mask_filename)

    def save_blender_file(self, filename: str):
        """Saves the current blender state to the given filename."""
        bpy.ops.wm.save_as_mainfile(filepath=filename)

    def __init__(self,
                 parameters: scene_parameters.SceneParameters,
                 ):
        butils.clear_all()
        self.parameters = parameters

        self.obj = blender_object.Two4TwoBlenderObject(
            parameters.obj_name,
            parameters.spherical,
            parameters.arm_position)
        self.obj.add_material(parameters.obj_color_rgba)

        blend_dir = os.path.dirname(bpy.data.filepath)
        if blend_dir not in sys.path:
            sys.path.append(blend_dir)

        self._set_pose(parameters.bending)

        self._set_rotation(
            parameters.obj_rotation_yaw,
            parameters.obj_rotation_roll,
            parameters.obj_rotation_pitch,
        )
        x = parameters.position_x
        y = parameters.position_y
        self._set_position(x, y)

        self._setup_scene(parameters.bg_color_rgba)

        res_x, res_y = parameters.resolution
        bpy.context.scene.render.engine = 'CYCLES'
        # bpy.context.scene.use_nodes = True
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.resolution_x = res_x
        bpy.context.scene.render.resolution_y = res_y
