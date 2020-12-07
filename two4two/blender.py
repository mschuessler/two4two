import subprocess
import time
from two4two.chunk import Chunk
import os

class Blender():
    
    def StartNew(self):
        """ Start a new subprocess if there is work to do """
        if self.next_chunk < self.num_of_chunks:
            args = [
                self.blender_path,
                '--background',
                '-noaudio',
                '--python',
                self.render_script,
                '--',
                self.parameter_chunks[self.next_chunk],
                self.output_dir,
                'params_chunk_{}.json'.format(self.next_chunk)
            ]
            proc = subprocess.Popen(args)
            self.next_chunk += 1
            self.processes.append(proc)

    def CheckRunning(self):
        """ Check any running processes and start new ones if there are spare slots."""
        for p in range(len(self.processes),0,-1): # Check the processes in reverse order
            p -= 1
            if self.processes[p].poll() is not None: # If the process hasn't finished will return None
                del self.processes[p] # Remove from list - this is why we needed reverse order

        while (len(self.processes) < self.n_processes) and (self.next_chunk < self.num_of_chunks): # More to do and some spare slots
            self.StartNew()
    
    def __init__(self,
                 parameter_file,
                 output_dir,
                 n_processes,
                 chunk_size):
        
        self.package_directory = '/home/philipp/242'
        self.blender_path = os.path.join(self.package_directory, 'blender/blender')
        self.render_script = os.path.join(self.package_directory, 'two4two/render_samples.py')
        
        self.output_dir = output_dir
        parameter_output = os.path.join(self.output_dir, 'parameters.json')
        assert not os.path.exists(parameter_output)
        
        chunks = Chunk(parameter_file, chunk_size)
        
        self.n_processes = n_processes
        self.parameter_chunks = chunks.file_names
        self.processes = []
        self.num_of_chunks = len(self.parameter_chunks)
        self.next_chunk = 0
        
        print('Split {} into {} chunks.'.format(parameter_file, self.num_of_chunks))
        
        self.CheckRunning()
        while (len(self.processes) > 0):
            time.sleep(0.1)
            self.CheckRunning()
        
        parameter_output = os.path.join(self.output_dir, 'parameters.json')
        with open(parameter_output, mode='x') as fparams:
            for i in range(self.num_of_chunks):
                file = os.path.join(self.output_dir, 
                                    'params_chunk_{}.json'.format(i))
                with open(file) as f:
                     fparams.write(f.read())
                os.remove(file)
            
        chunks.remove_chunks()