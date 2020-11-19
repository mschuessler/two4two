import json
import os

from parameters import Parameters

class Chunk():
    
    def remove_chunks(self):
        for f in self.file_names:
            os.remove(f)
    
    def __init__(self, param_file, chunk_size):
        
        parameter_list = []
        self.file_names = []
        
        with open(param_file) as fparam:
            for line in fparam.readlines():
                params = json.loads(line)
                
                parameters = Parameters()
                parameters.__dict__.update(params)
                
                parameter_list.append(parameters)
        
        print(len(parameter_list))
        padding = len(parameter_list) % chunk_size
        for _ in range(padding):
            parameter_list.append(None)
        print(len(parameter_list))
        
        num_chunks = len(parameter_list) // chunk_size
        print(num_chunks)
        param_idx = 0
        
        for idx in range(num_chunks):
            
            file_name = os.path.basename(param_file)
            file_name = file_name.split('.')
            file_name = file_name[0] + '_chunk_' + str(idx + 1) + '.' + file_name[1]
            file_path = os.path.dirname(param_file)
            
            save_name = os.path.join(file_path, file_name)
            
            self.file_names.append(save_name)
            
            with open(save_name, mode='x') as fsave:
                for _ in range(chunk_size):
                    if parameter_list[param_idx] is not None:
                        parameter_list[param_idx].save_parameters(fsave)
                        param_idx += 1
                    else:
                        pass