import bpy


def clear_all():
    object_mode()
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def edit_mode():
    if not bpy.context.mode == 'EDIT':
        bpy.ops.object.mode_set(mode='EDIT')


def object_mode():
    if not bpy.context.mode == 'OBJECT':
        bpy.ops.object.mode_set(mode='OBJECT')


def select(object_name, add_to_selection=True):
    object_mode()
    if not add_to_selection:
        bpy.ops.object.select_all(action='DESELECT')
    bpy.data.objects[object_name].select_set(True)


def select_object():
    select('object')
    select('skeleton', True)


def set_active(object_name):
    active_object = bpy.data.objects[object_name]
    bpy.context.view_layer.objects.active = active_object
    return active_object


def get_boundaries(objects):
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
