import numpy as np
import json
import butils
import tqdm.auto as tqdm_auto
import tempfile
from color_generator import ColorGenerator

class Parameters():
        
    def generate_parameters(self,
                            obj_name,
                            spherical=None,
                            bone_bend_range=0.3,
                            bone_rotation_range=0.2,
                            random_flip=False,
                            obj_incline=0.1,
                            obj_rotation=0.7):
        
        self.obj_name = obj_name
        
        if spherical is None: 
            self.spherical = np.random.beta(0.3, 0.3)
        else:
            self.spherical = spherical

        self.bone_bend = np.random.uniform(-bone_bend_range*np.pi/4,
                                           bone_bend_range*np.pi/4,
                                           size=7).tolist()
        self.bone_rotation = np.random.uniform(-bone_rotation_range*np.pi/4,
                                               bone_rotation_range*np.pi/4,
                                               size=7).tolist()

        self.obj_incline = obj_incline*np.random.uniform(-np.pi/4, np.pi/4)
        self.obj_rotation = obj_rotation*np.random.uniform(-np.pi/4, np.pi/4)

        if random_flip:
            self.flip = bool(np.random.uniform() < 0.5)
        else:
            self.flip = False

        self.position = np.random.uniform(-0.5, 0.5, size=2).tolist()

        rv_move_arm_right = butils.get_truncated_normal(0, 0.30, 0, 0.55)
        rv_move_arm_left = butils.get_truncated_normal(0, 0.30, -0.55, 0)
        if obj_name == 'sticky':
            rv_color = butils.get_truncated_normal(1, 0.6, 0, 1)
            self.arm_shift = rv_move_arm_right.rvs()
        elif obj_name == 'stretchy':
            rv_color = butils.get_truncated_normal(0, 0.6, 0, 1)
            self.arm_shift = rv_move_arm_left.rvs()
        else:
            raise Exception(obj_name)

        self.obj_color = ColorGenerator('seismic').get_color(rv_color.rvs())

        back_color = np.random.uniform(0.05, 0.80)
        self.back_color = ColorGenerator('binary').get_color(back_color)
    
    def save_parameters(self, file_handle):
        params = self.__dict__
        file_handle.writelines(json.dumps(params) + '\n')
        
    def generate_many(self, n, save_location, object_types, structure_types):
        
        n = int(n)
        
        if object_types == 'sticky':
            obj_type = np.full(n, object_types)
        elif object_types == 'stretchy':
            obj_type = np.full(n, object_types)
        elif object_types == 'random':
            obj_type = np.full(n, 'stretchy')
            mask = np.random.randint(2, size=n)
            obj_type[np.where(mask)] = 'sticky'
            print(obj_type)
        else:
            raise Exception(object_types)
        
        if structure_types == 'cubes_and_spheres':
            structure = np.random.randint(2, size=n)
        elif structure_types == 'cubes':
            structure = np.zeros(n)
        elif structure_types == 'spheres':
            structure = np.ones(n)
        elif structure_types == 'random':
            structure = np.random.uniform(size=n)
        else:
            raise Exception(structure_types)
            
        with open(save_location, mode='x') as f:
            for i in tqdm_auto.trange(n):
                self.generate_parameters(obj_type[i],
                                         structure[i])
                self.save_parameters(f)
   
    def __init__(self, obj_name=None):
       
        if obj_name is not None:
            self.generate_parameters(obj_name)
        else:
            self.obj_name = None
            self.spherical = None
            self.bone_bend = None
            self.bone_rotation = None
            self.obj_incline = None
            self.obj_rotation = None
            self.flip = None
            self.position = None
            self.arm_shift = None
            self.obj_color = None
            self.back_color = None