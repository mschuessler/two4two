import bpy
import numpy as np
import butils
from mathutils import Vector

class BlenderObject():

    def create_model(self):
        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=True,
                                        location=(0,0,0))

        bpy.context.active_object.name = 'object'

        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=False,
                                        location=(0,1,0))

        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=False,
                                        location=(0,-1,0))

        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=False,
                                        location=(0,2,0))

        # Defining arms
        if self.right == 'sticky':
            arm_pos = 1.5
        elif self.right == 'stretchy':
            arm_pos = 2.5
        else:
            raise "Arms must be either 'sticky' or 'stretchy'."
            
        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=False,
                                        location=(0,arm_pos  + self.arm_shift,-1))

        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=False,
                                        location=(0,arm_pos + self.arm_shift,1))


        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=False,
                                        location=(0,-1.5,1))

        bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                        enter_editmode=False,
                                        location=(0,-1.5,-1))

        butils.object_mode()

    def create_armature(self):
        bpy.ops.object.armature_add(enter_editmode=True,
                                    radius=np.sqrt(1.25),
                                    location=(0,-1.5,-1),
                                    rotation=(-np.arctan(0.25),0,0))

        bpy.context.active_object.name = 'skeleton'

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked":False},
                                      TRANSFORM_OT_translate={"value":(0, -0.5, 1)})

        bpy.ops.armature.select_hierarchy()

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked":False},
                                      TRANSFORM_OT_translate={"value":(0, 1, 0)})

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked":False},
                                      TRANSFORM_OT_translate={"value":(0, 1, 0)})

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked":False},
                                      TRANSFORM_OT_translate={"value":(0, 1, 0)})
        if self.right == 'sticky':
            bpy.context.active_bone.select_tail = False
            bpy.context.active_bone.select_head = True

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked":False},
                                          TRANSFORM_OT_translate={"value":(0, 0.5+self.arm_shift, 1)})

        bpy.context.active_bone.select_tail = False
        bpy.context.active_bone.select_head = True
        
        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked":False},
                                      TRANSFORM_OT_translate={"value":(0, 0.5+self.arm_shift, -1)})

    def make_parent(self):
        butils.select('object', add_to_selection=False)
        butils.select('skeleton')
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    def set_pose(self, bend_angles, rotation_angles):
        butils.set_active('skeleton')
        bpy.ops.object.posemode_toggle()

        for i,bone in enumerate(bpy.data.objects['skeleton'].data.bones):

            bone.select = True

            bend_angle = bend_angles[i]
            rotation_angle = rotation_angles[i]

            bpy.ops.transform.translate(value=(bend_angle, 0, 0),
                                    orient_type='GLOBAL',
                                    orient_matrix=((1,0,0),
                                                   (0,1,0),
                                                   (0,0,1)),
                                    orient_matrix_type='GLOBAL',
                                    constraint_axis=(False, True, False))

            bpy.ops.transform.rotate(value=rotation_angle, orient_axis='X')

            bone.select = False

        butils.object_mode()

        butils.set_active('object')
        bpy.ops.object.modifier_apply(apply_as='DATA',
                                      modifier='Armature')
        butils.set_active('skeleton')
        bpy.ops.object.modifier_apply(apply_as='DATA',
                                      modifier='Armature')

    def create_bounding_box(self):
        ((min_x, max_x), (min_y, max_y), (min_z, max_z)) = butils.get_boundaries('object')

        bpy.ops.mesh.primitive_cube_add(size=1,
                                        location=(min_x+0.5, min_y+0.5, min_z+0.5),
                                        enter_editmode=False)

        bpy.context.active_object.name = 'bounding_box'

        bpy.ops.transform.resize(value=(max_x-min_x, max_y-min_y, max_z-min_z),
                                 center_override=(min_x, min_y, min_z))

        butils.select('bounding_box')
        butils.select('skeleton', add_to_selection=True)

        bpy.ops.object.parent_set(type='OBJECT',
                                  keep_transform=True)

    def remove_bounding_box(self):
        butils.select('bounding_box')
        butils.select('skeleton', add_to_selection=True)
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.select_all(action='DESELECT')
        butils.select('bounding_box')
        bpy.ops.object.delete(use_global=False,
                              confirm=False)

        butils.set_active('object')
        bpy.ops.object.transform_apply()
        butils.set_active('skeleton')
        bpy.ops.object.transform_apply()

    def move(self, translation_vec):
        self.create_bounding_box()
        butils.select('bounding_box')

        bpy.ops.transform.translate(value=translation_vec)
        self.remove_bounding_box()

    def rotate(self, angle, axis):
        self.create_bounding_box()
        butils.select('bounding_box')

        bpy.ops.transform.rotate(value=angle,
                                 orient_axis=axis)

        self.remove_bounding_box()

    def center(self):
        self.create_bounding_box()
        o = bpy.data.objects['bounding_box']
        local_bbox_center = 0.125 * sum((Vector(b) for b in o.bound_box), Vector())
        global_bbox_center = o.matrix_world @ local_bbox_center
        butils.select('bounding_box')
        bpy.ops.transform.translate(value=-0.5*global_bbox_center)
        self.remove_bounding_box()

    def add_material(self, color):
        active_object = butils.set_active('object')
        mat = bpy.data.materials.new(name='material')
        active_object.data.materials.append(mat)
        bpy.context.object.active_material.diffuse_color = color

    def set_spherical(self, amount, clamp_overlap=True):
        width = 0.05 + self.cube_size * 0.25 * amount
        obj = butils.set_active('object')

        bpy.ops.object.modifier_add(type='BEVEL')
        obj.modifiers['Bevel'].width = width
        bpy.ops.object.modifier_apply()
        bpy.ops.object.modifier_add(type='SUBSURF')
        obj.modifiers['Subdivision'].render_levels = 5
        bpy.ops.object.modifier_apply()

    @property
    def boundaries(self):
        return butils.get_boundaries('object')

    def __init__(self,
                 right,
                 spherical=0,
                 arm_shift=None):
        
        # Object Type. 'Sticky' or 'Stretchy'
        self.right = right
        
        if arm_shift is None:
            self.arm_shift, _ = butils.get_random_arm_shift()
        else:
            self.arm_shift = arm_shift

        self.num_of_cubes = 8
        base_width = 0.8
        self.cube_size = base_width + spherical * 0.1

        self.create_model()
        self.create_armature()
        self.make_parent()
        self.center()
        self.set_spherical(spherical)