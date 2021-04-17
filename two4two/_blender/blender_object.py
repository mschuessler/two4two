"""module for the blender object of Peaky and Stretechy."""

from __future__ import annotations

import dataclasses
from typing import Dict, Tuple

import bpy
from mathutils import Vector
import numpy as np

from two4two._blender import butils
import two4two.utils


@dataclasses.dataclass
class BoneRotation:
    r"""Specifies the rotation of the individual bones.

    The object has a bone between each block. Except in the center between
    ``spine_left_center`` and ``spine_right_center``. There, we add two bones
    as this gives a unique center point to the object. The object is then
    bended around that center.

    Here is the bone layout. ``[]`` marks a joint:

    ```
         arm_left_top
           \              center_left          /  --  arm_right_top
            \    left     |           right   /
             [] ------ [] -- [] -- [] ------ []
            /                   |             \
           /                    center_right   \ --  arm_right_bottom
          arm_left_bottom
    ```
    """

    center_right: float = 0.
    right: float = 0.
    arm_right_top: float = 0.
    arm_right_bottom: float = 0.

    center_left: float = 0.
    left: float = 0.
    arm_left_top: float = 0.
    arm_left_bottom: float = 0.

    @staticmethod
    def from_bending(bending: float, bend_arms: bool = False) -> BoneRotation:
        """Creates rotation for a single bending scalar.

        Args:
            bending: scalar bx which each bone should rotate (-2pi,+2pi)
            bend_arms: flag wheter arm bones should be rotated

        """
        return BoneRotation(
            center_right=bending / 2,
            right=bending,
            arm_right_top=bending if bend_arms else 0,
            arm_right_bottom=bending if bend_arms else 0,
            # switch sign for left
            center_left=-bending / 2,
            left=-bending,
            arm_left_top=bending if bend_arms else 0,
            arm_left_bottom=bending if bend_arms else 0,
        )


class Two4TwoBlenderObject():
    """Object that represents either a Peaky or Stretchy.

    Args:
        obj_name: Object name (either ``"peaky"`` or ``"stretchy"``).
        spherical: For ``1``,  spherical objects. For ``0``, cubes.
            Can have values in-between.
        arm_position: Absolute arm positions.
    """

    def _get_object_locations(self) -> Dict[str, Tuple[float, float, float]]:
        arm_pos = 1.5 + self.arm_position
        return {
            'arm_left_top': Vector((0., -1.5, 1.)),
            'arm_left_bottom': Vector((0., -1.5, -1.)),
            'spine_left': Vector((0., -1., 0)),
            'spine_left_center': Vector((0., 0., 0.)),
            'spine_right_center': Vector((0., 1., 0.)),
            'spine_right': Vector((0., 2., 0.)),
            'arm_right_top': Vector((0., arm_pos, 1.)),
            'arm_right_bottom': Vector((0., arm_pos, -1.)),
        }

    def _create_model(self):
        """Creates the blocks of the object.

        Overview how the individual blocks are named:

        [arm_left_top]                                                          [arm_right_top]
               [spine_left]  [spine_left_center]  [spine_right_center]   [spine_right]
        [arm_left_bottom]                                                       [arm_right_bottom]

        """
        object_locations = self._get_object_locations()

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
        r"""Add bones.

        We add a bone between each block. Except in the center between
        ``spine_left_center`` and ``spine_right_center``. There, we
        add two bones as this gives a unique center point to the object.
        It is then bended around that center.

        Here is the bone layout. ``[]`` marks a joint:
        ```
          arm_left_top
            \              center_left          /  --  arm_right_top
             \    left     |           right   /
              [] ------ [] -- [] -- [] ------ []
             /                   |             \
            /                    center_right   \ --  arm_right_bottom
           arm_left_bottom
        ```
        """
        blocks = self._get_object_locations()

        def block_offset(block1: str, block2: str) -> Vector:
            return blocks[block1] - blocks[block2]

        def add_bone(bone_name: str,
                     from_block: str,
                     to_block: str,
                     forked: bool = False):
            bpy.ops.armature.extrude_move(
                ARMATURE_OT_extrude={"forked": forked},
                TRANSFORM_OT_translate={
                    "value": block_offset(to_block, from_block)})
            bpy.context.active_bone.name = bone_name

        def select(bone_name: str, mode: str):
            bpy.ops.armature.select_all(action='DESELECT')
            bone = bpy.context.active_object.data.edit_bones[bone_name]
            bone.select = False
            if mode == 'tail':
                bone.select_tail = True
                bone.select_head = False
            elif mode == 'head':
                bone.select_tail = False
                bone.select_head = True
            else:
                raise ValueError(mode)

        def attach_bones_to_blocks(orientation: str):
            """Combines bones with blocks."""

            butils.object_mode()
            bpy.ops.object.select_all(action='DESELECT')
            for name in self.blocks.keys():
                if orientation in name:
                    butils.select(name)
            butils.select(f'skeleton_{orientation}')
            bpy.ops.object.parent_set(type='ARMATURE_AUTO')

        obj_center = (blocks['spine_right_center'] + blocks['spine_left_center']) / 2

        bpy.ops.object.armature_add(enter_editmode=True,
                                    radius=0.5,
                                    location=obj_center,
                                    rotation=(-np.pi / 2, 0, 0))
        bpy.context.active_object.name = 'skeleton_right'
        bpy.context.active_bone.name = 'center_right'

        select('center_right', 'tail')
        add_bone('right',
                 from_block='spine_right_center',
                 to_block='spine_right')

        select('right', 'tail')
        add_bone('arm_right_top',
                 from_block='spine_right',
                 to_block='arm_right_top')

        select('right', 'tail')
        add_bone('arm_right_bottom',
                 from_block='spine_right',
                 to_block='arm_right_bottom')

        butils.object_mode()
        attach_bones_to_blocks('right')

        # build the left part
        bpy.ops.object.armature_add(enter_editmode=True,
                                    radius=0.5,
                                    location=obj_center,
                                    rotation=(+np.pi / 2, 0, 0))
        bpy.context.active_object.name = 'skeleton_left'
        bpy.context.active_bone.name = 'center_left'

        select('center_left', 'tail')
        add_bone('left',
                 from_block='spine_left_center',
                 to_block='spine_left')

        select('left', 'tail')
        add_bone('arm_left_top',
                 from_block='spine_left',
                 to_block='arm_left_top')

        select('left', 'tail')
        add_bone('arm_left_bottom',
                 from_block='spine_left',
                 to_block='arm_left_bottom')
        attach_bones_to_blocks('left')

    def set_pose(self,
                 bending: BoneRotation,
                 bone_translation: BoneRotation = None):
        """Set bond bending and rotations.

        Attrs:
            bending: Rotation of the individual bone segments.
            bone_translation: Bending of the individual bone segments.
                (Attribute removed from SceneParameters but functionality remains implemented)
        """

        if bone_translation is None:
            # creates an emptpy translation of 0.
            bone_translation = BoneRotation()

        for orientation in ('right', 'left'):
            skeleton = f'skeleton_{orientation}'
            butils.set_active(skeleton)
            bpy.ops.object.posemode_toggle()

            for bone in bpy.data.objects[skeleton].data.bones:
                rotation_angle = getattr(bending, bone.name)
                bone.select = True
                bend_angle = getattr(bone_translation, bone.name)
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
                if orientation in block_name:
                    butils.set_active(block_name)
                    bpy.ops.object.modifier_apply(apply_as='DATA',
                                                  modifier='Armature')
            butils.set_active(skeleton)
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
        butils.select('skeleton_right', add_to_selection=True)
        butils.select('skeleton_left', add_to_selection=True)

        bpy.ops.object.parent_set(type='OBJECT',
                                  keep_transform=True)

    def remove_bounding_box(self):
        """Remove bounding box."""
        butils.select('bounding_box')
        butils.select('skeleton_right', add_to_selection=True)
        butils.select('skeleton_left', add_to_selection=True)
        bpy.ops.object.parent_clear(type='CLEAR_KEEP_TRANSFORM')
        bpy.ops.object.select_all(action='DESELECT')
        butils.select('bounding_box')
        bpy.ops.object.delete(use_global=False,
                              confirm=False)
        for block_name in self.blocks.keys():
            butils.set_active(block_name)
            bpy.ops.object.transform_apply()
        butils.set_active('skeleton_right')
        butils.set_active('skeleton_left')
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
        return butils.get_boundaries(list(self.blocks.values()))

    def __init__(self,
                 obj_name: str,
                 spherical: float = 0,
                 arm_position: float = 0):

        # Object Type. 'peaky' or 'stretchy'
        self.obj_name = obj_name

        self.arm_position = arm_position
        self.blocks: Dict[str, bpy.types.Object] = {}
        self.num_of_cubes = 8
        base_width = 0.8
        self.cube_size = base_width + spherical * 0.1

        self._create_model()
        self._create_armature()
        self.center()
        self._set_spherical(spherical)
