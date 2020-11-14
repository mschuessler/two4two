import os, sys
import bpy

blend_dir = os.path.dirname(bpy.data.filepath)
if blend_dir not in sys.path:
   sys.path.append(blend_dir)

from parameters import Parameters

if __name__ == '__main__':
    parameters = Parameters()
    parameters.generate_many(sys.argv[-4],
                             sys.argv[-3],
                             sys.argv[-2],
                             sys.argv[-1])