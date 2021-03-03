"""module for the blender object of Sticky and Stretechy."""
from typing import Sequence, Tuple

import bpy
from mathutils import Vector
import numpy as np

from two4two._blender import butils
import two4two.utils


class BlenderObject():
    """Object that represents either a Sticky or Stretchy.

    Args:
        obj_name: Object name (either ``"sticky"`` or ``"stretchy"``).
        spherical: For ``1``,  spherical objects. For ``0``, cubes.
            Can have values in-between.
        arm_position: Absolute arm positions.
    """

    def _create_model(self):
        """Creates the blocks of the object.

        Overview how the individual blocks are named:

        [arm_left_top]                                                          [arm_right_top]
               [spline_left]  [spline_left_center]  [spline_right_center]   [spline_right]
        [arm_left_bottom]                                                       [arm_right_bottom]

        """
        arm_pos = 1.5 + self.arm_position
        object_locations = {
            'arm_left_top': (0, -1.5, 1),
            'arm_left_bottom': (0, -1.5, -1),
            'spline_left': (0, -1, 0),
            'spline_left_center': (0, 0, 0),
            'spline_right_center': (0, 1, 0),
            'spline_right': (0, 2, 0),
            'arm_right_top': (0, arm_pos, 1),
            'arm_right_bottom': (0, arm_pos, -1),
        }

        animal_blocks = bpy.data.collections.new("animal_blocks")
        for name, location in object_locations.items():
            bpy.ops.mesh.primitive_cube_add(size=self.cube_size,
                                            enter_editmode=True,
                                            location=location)
            obj = bpy.context.object
            self.blocks[name] = obj
            obj.name = name
            animal_blocks.objects.link(obj)
            butils.object_mode()

    def _create_armature(self):
        """Add bones."""
        # TODO(philipp): can you document how each bone is connected?
        bpy.ops.object.armature_add(enter_editmode=True,
                                    radius=np.sqrt(1.25),
                                    location=(0, -1.5, -1),
                                    rotation=(-np.arctan(0.25), 0, 0))
        bpy.context.active_object.name = 'skeleton'
        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked": False},
                                      TRANSFORM_OT_translate={"value": (0, -0.5, 1)})

        bpy.ops.armature.select_hierarchy()

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked": False},
                                      TRANSFORM_OT_translate={"value": (0, 1, 0)})

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked": False},
                                      TRANSFORM_OT_translate={"value": (0, 1, 0)})

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked": False},
                                      TRANSFORM_OT_translate={"value": (0, 1, 0)})

        if self.obj_name == 'sticky':
            # TODO(philipp): why is this if here? why different bones connection for sticky?
            bpy.context.active_bone.select_tail = False
            bpy.context.active_bone.select_head = True
            arm_shift = self.arm_position
        else:
            arm_shift = 1 - self.arm_position

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked": False},
                                      TRANSFORM_OT_translate={"value": (0, 0.5 + arm_shift, 1)})

        bpy.context.active_bone.select_tail = False
        bpy.context.active_bone.select_head = True

        bpy.ops.armature.extrude_move(ARMATURE_OT_extrude={"forked": False},
                                      TRANSFORM_OT_translate={"value": (0, 0.5 + arm_shift, -1)})

    def _attach_bones_to_blocks(self):
        """Combines bones with blocks."""
        butils.object_mode()
        bpy.ops.object.select_all(action='DESELECT')
        for name in self.blocks.keys():
            butils.select(name)
        butils.select('skeleton')
        bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    def set_pose(self, bone_bend: Sequence[float], bone_rotation: Sequence[float]):
        """Set bond bending and rotations."""
        # TODO(philipp): how are the rotations applied. Shouldn't there be 3 degrees of freedom?
        butils.set_active('skeleton')
        bpy.ops.object.posemode_toggle()

        for i, bone in enumerate(bpy.data.objects['skeleton'].data.bones):

            bone.select = True

            bend_angle = bone_bend[i]
            rotation_angle = bone_rotation[i]

            bpy.ops.transform.translate(
                value=(bend_angle, 0, 0),
                orient_type='GLOBAL',
                orient_matrix=((1, 0, 0),
                               (0, 1, 0),
                               (0, 0, 1)),
                orient_matrix_type='GLOBAL',
                constraint_axis=(False, True, False))

            bpy.ops.transform.rotate(value=rotation_angle, orient_axis='X')

            bone.select = False

        butils.object_mode()
        bpy.ops.object.select_all(action='DESELECT')
        for block_name in self.blocks.keys():
            butils.set_active(block_name)
            bpy.ops.object.modifier_apply(apply_as='DATA',
                                          modifier='Armature')
        butils.set_active('skeleton')
        bpy.ops.object.modifier_apply(apply_as='DATA',
                                      modifier='Armature')

    def create_bounding_box(self):
        """Add a bounding box around all blocks.

        The bounding box is saved a new object named ``bounding_box``.
        """
        ((min_x, max_x), (min_y, max_y), (min_z, max_z)) = self.boundaries

        bpy.ops.mesh.primitive_cube_add(size=1,
                                        location=(min_x + 0.5, min_y + 0.5, min_z + 0.5),
                                        enter_editmode=False)

        bpy.context.active_object.name = 'bounding_box'

        bpy.ops.transform.resize(value=(max_x - min_x, max_y - min_y, max_z - min_z),
                                 center_override=(min_x, min_y, min_z))

        butils.select('bounding_box')
        butils.select('skeleton', add_to_selection=True)

        bpy.ops.object.parent_set(type='OBJECT',
                                  keep_transform=True)

    def remove_bounding_box(self):
        """Remove bounding box."""
        butils.select('bounding_box')
        butils.select('skeleton', add_to_selection=True)
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.select_all(action='DESELECT')
        butils.select('bounding_box')
        bpy.ops.object.delete(use_global=False,
                              confirm=False)
        for block_name in self.blocks.keys():
            butils.set_active(block_name)
            bpy.ops.object.transform_apply()
        butils.set_active('skeleton')
        bpy.ops.object.transform_apply()

    def move(self, translation_vec: Tuple[float, float, float]):
        """Translate the object."""
        self.create_bounding_box()
        butils.select('bounding_box')

        bpy.ops.transform.translate(value=translation_vec)
        self.remove_bounding_box()

    def rotate(self, angle: float, axis: str):
        """Rotate the object along the given axis."""
        self.create_bounding_box()
        butils.select('bounding_box')

        bpy.ops.transform.rotate(value=angle,
                                 orient_axis=axis)

        self.remove_bounding_box()

    def center(self):
        """Center the bounding box."""
        # TODO(philpp): where is the center? what is the 0.125 constant?
        self.create_bounding_box()
        o = bpy.data.objects['bounding_box']
        local_bbox_center = 0.125 * sum((Vector(b) for b in o.bound_box), Vector())
        global_bbox_center = o.matrix_world @ local_bbox_center
        butils.select('bounding_box')
        bpy.ops.transform.translate(value=-0.5 * global_bbox_center)
        self.remove_bounding_box()

    def add_material(self, color: two4two.utils.RGBAColor):
        """Sets the color to all blocks."""
        for name in self.blocks:
            active_object = butils.set_active(name)
            mat = bpy.data.materials.new(name='material')
            active_object.data.materials.append(mat)
            bpy.context.object.active_material.diffuse_color = color

    def _set_spherical(self,
                       amount: float,
                       clamp_overlap: bool = True):
        """Make the squares more round."""
        # TODO(philpp): do you know where this equation comes from?
        width = 0.05 + self.cube_size * 0.25 * amount
        for name in self.blocks:
            block = butils.set_active(name)
            bpy.ops.object.modifier_add(type='BEVEL')
            block.modifiers['Bevel'].width = width
            bpy.ops.object.modifier_apply()
            bpy.ops.object.modifier_add(type='SUBSURF')
            block.modifiers['Subdivision'].render_levels = 5
            bpy.ops.object.modifier_apply()

    @property
    def boundaries(self) -> butils.BOUNDING_BOX:
        """The object boundaries."""
        return butils.get_boundaries(self.blocks.values())

    def __init__(self,
                 obj_name: str,
                 spherical: float = 0,
                 arm_position: float = None):

        # Object Type. 'sticky' or 'stretchy'
        self.obj_name = obj_name

        self.arm_position = arm_position
        self.blocks = {}
        self.num_of_cubes = 8
        base_width = 0.8
        self.cube_size = base_width + spherical * 0.1

        self._create_model()
        self._create_armature()
        self._attach_bones_to_blocks()
        self.center()
        self._set_spherical(spherical)
