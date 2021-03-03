"""module for blender util functions."""

from typing import Sequence, Tuple

import bpy


def clear_all():
    """Resets blender initial state and removes any objects."""
    bpy.ops.wm.read_factory_settings(use_empty=True)
    object_mode()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def edit_mode():
    """Enter blender edit mode."""
    if not bpy.context.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')


def object_mode():
    """Enter blender object mode."""
    if not bpy.context.mode == 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def select(object_name: str, add_to_selection: bool = True):
    """Enters blender object mode and select the named object.

    Args:
        object_name: name of the object to select.
        add_to_selection: whether to add to the existing selection.
    """
    object_mode()
    if not add_to_selection:
        bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[object_name].select_set(True)


def set_active(object_name: str) -> bpy.types.Object:
    """Set object active."""
    active_object = bpy.data.objects[object_name]
    bpy.context.view_layer.objects.active = active_object
    return active_object


BOUNDING_BOX = Tuple[
    Tuple[float, float],
    Tuple[float, float],
    Tuple[float, float],
]


def get_boundaries(
    objects: Sequence[bpy.types.Object]
) -> BOUNDING_BOX:
    """Returns the bounding box of the objects."""
    glob_vertex_coordinates = [obj.matrix_world @ v.co
                               for obj in objects
                               for v in obj.data.vertices]

    min_x = min([co.x for co in glob_vertex_coordinates])
    max_x = max([co.x for co in glob_vertex_coordinates])

    min_y = min([co.y for co in glob_vertex_coordinates])
    max_y = max([co.y for co in glob_vertex_coordinates])

    min_z = min([co.z for co in glob_vertex_coordinates])
    max_z = max([co.z for co in glob_vertex_coordinates])

    boundaries = ((min_x, max_x),
                  (min_y, max_y),
                  (min_z, max_z))
    return boundaries
