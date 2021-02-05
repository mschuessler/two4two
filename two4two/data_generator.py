import bpy
import sys
import os
import numpy as np

import two4two.butils

from two4two.blender_object import BlenderObject


class DataGenerator():

    def random_pose(self, bend_range, rotation_range):
        bend_angles = np.random.uniform(-bend_range*np.pi/4, bend_range*np.pi/4, size=7)
        rotation_angles = np.random.uniform(-rotation_range*np.pi/4, rotation_range*np.pi/4, size=7)
        self.obj.set_pose(bend_angles, rotation_angles)
        self.obj.center()

        return bend_angles, rotation_angles

    def set_pose(self, bend_angles, rotation_angles):
        self.obj.set_pose(bend_angles, rotation_angles)
        self.obj.center()

    def random_rotation(self,
                        incline_variation=0.3,
                        rotation_variation=0.8,
                        random_flip=False):

        incline = np.random.uniform(-np.pi/4, np.pi/4)
        self.obj.rotate(incline_variation*incline, 'Y')
        head_pos = np.random.uniform(-np.pi/4, np.pi/4)
        self.obj.rotate(rotation_variation*head_pos, 'Z')
        flipped = False
        if random_flip:
            if np.random.uniform() < 0.5:
                flipped = True
                self.flip_orientation()
        self.obj.center()

        return incline_variation*incline, rotation_variation*head_pos, flipped

    def set_rotation(self, incline, rotation, flipped):
        self.obj.rotate(incline, 'Y')
        self.obj.rotate(rotation, 'Z')
        if flipped:
            self.flip_orientation()
        self.obj.center()

    def flip_orientation(self):
        self.obj.rotate(np.pi, 'Z')

    def position_randomly_within_frame(self):
        x, y = np.random.uniform(-0.5, 0.5, size=2)
        self.obj.move((0, x, y))

        return x, y

    def set_position(self, x, y):
        self.obj.move((0, x, y))

    def scene(self,
              background_color,
              light_size=20):

        scene = bpy.context.scene

        bpy.ops.object.light_add(type='AREA',
                                 location=(10, 10, 10))
        bpy.context.active_object.name = 'light'
        light = bpy.data.objects['light']
        light.data.shape = 'ELLIPSE'
        light.data.size = light_size
        light.data.size_y = 15
        light.data.energy = 5000 / 20 * light_size * 4
        light.rotation_euler = (np.pi/4, np.pi/4, np.pi/2)

        bpy.ops.object.camera_add(location=(9, 0, 0),
                                  rotation=(np.pi/2,0,np.pi/2))
        scene.camera = bpy.context.object

        bpy.ops.mesh.primitive_plane_add(size=20,
                                         enter_editmode=False,
                                         location=(self.obj.boundaries[0][0] - 10,0,0),
                                         rotation=(0,np.pi/2,0))
        bpy.context.active_object.name = 'background'
        background = bpy.context.view_layer.objects.active
        mat = bpy.data.materials.new(name='background_material')
        background.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = background_color

    def color_code_object(self):
        colors = [
            (1.0, 0.06, 0.0, 1),
            (0.9399999999999997, 1.0, 0.0, 1),
            (0.0, 1.0, 0.06000000000000005, 1),
            (0.0, 0.9399999999999997, 1.0, 1),
            (0.06000000000000005, 0.0, 1.0, 1),
            (1.0, 0.0, 0.9399999999999997, 1)]
        black = (0, 0, 0, 1)
        blue = (0, 0, 1, 1)

        two4two.butils.select_object()
        two4two.butils.edit_mode()
        bpy.ops.mesh.separate(type='LOOSE')
        two4two.butils.object_mode()

        active_object = two4two.butils.set_active('object.003')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[0]

        active_object = two4two.butils.set_active('object.004')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[2]

        active_object = two4two.butils.set_active('object.005')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[4]


        active_object = two4two.butils.set_active('object')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = blue

        active_object = two4two.butils.set_active('object.001')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = black

        active_object = two4two.butils.set_active('object.002')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[1]

        active_object = two4two.butils.set_active('object.006')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[3]

        active_object = two4two.butils.set_active('object.007')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = colors[5]

    def render(self,
              file_name):
        bpy.data.scenes['Scene'].render.filepath = file_name
        bpy.ops.render.render( write_still=True )


    def __init__(self,
                 parameters):

        two4two.butils.clear_all()

        self.obj = BlenderObject(parameters.obj_name,
                                 parameters.spherical,
                                 parameters.arm_position)

        self.obj.add_material(parameters.obj_color)

        blend_dir = os.path.dirname(bpy.data.filepath)
        if blend_dir not in sys.path:
            sys.path.append(blend_dir)

        self.set_pose(parameters.bone_bend,
                       parameters.bone_rotation)
        self.set_rotation(parameters.obj_incline,
                           parameters.obj_rotation,
                           parameters.flip)
        x,y = parameters.position
        self.set_position(x,y)
        self.scene(parameters.bg_color)

        res_x, res_y = parameters.resolution
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'
        bpy.context.scene.render.resolution_x = res_x
        bpy.context.scene.render.resolution_y = res_y
