import subprocess
import time
from chunk import Chunk
import os

class Blender():

    def StartNew(self):
        """ Start a new subprocess if there is work to do """
        if self.next_chunk < self.num_of_chunks:
            proc = subprocess.Popen(['/home/philipp/242/blender/blender',
                                     '--background',
                                     '--python',
                                     '/home/philipp/242/python/render_samples.py',
                                     self.parameter_chunks[self.next_chunk],
                                     self.output_dir,
                                     'params_chunk_{}.json'.format(self.next_chunk)])
            print("Started to Process {}".format(self.parameter_chunks[self.next_chunk]))
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
        
        chunks = Chunk(parameter_file, chunk_size)
        
        self.output_dir = output_dir
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