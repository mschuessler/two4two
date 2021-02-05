from typing import Sequence
import subprocess
import time
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
            ]
            proc = subprocess.Popen(args)
            self.next_chunk += 1
            self.processes.append(proc)

    def CheckRunning(self):
        """ Check any running processes and start new ones if there are spare slots."""
        # Check the processes in reverse order
        for p in range(len(self.processes), 0, -1):
            p -= 1
            # If the process hasn't finished will return None
            if self.processes[p].poll() is not None:
                # Remove from list - this is why we needed reverse order
                del self.processes[p]

        # More to do and some spare slots
        while (len(self.processes) < self.n_processes) and (self.next_chunk < self.num_of_chunks):
            self.StartNew()

    @staticmethod
    def _split_param_file(parameter_file: str, chunk_size: int) -> Sequence[str]:
        with open(parameter_file) as f:
            lines = f.readlines()

        num_chunks = len(lines) // chunk_size
        if len(lines) % chunk_size != 0:
            num_chunks += 1

        splits = []
        name, ext = os.path.splitext(parameter_file)
        for idx in range(num_chunks):
            filename = f'{name}_chunk_{idx + 1}.{ext}'
            splits.append(filename)
            with open(filename, 'x') as f:
                f.writelines(lines[idx * chunk_size:(idx + 1) * chunk_size])
        return splits

    def __init__(self,
                 parameter_file,
                 output_dir,
                 n_processes,
                 chunk_size):

        self.package_directory = "REPLACE-WITH-PWD"
        self.blender_path = os.path.join(self.package_directory, 'blender/blender')
        self.render_script = os.path.join(self.package_directory, 'two4two/render_samples.py')

        self.output_dir = output_dir
        parameter_output = os.path.join(self.output_dir, 'parameters.json')
        assert not os.path.exists(parameter_output)

        self.n_processes = n_processes
        self.parameter_chunks = self._split_param_file(parameter_file, chunk_size)
        self.processes = []
        self.num_of_chunks = len(self.parameter_chunks)
        self.next_chunk = 0

        print('Split {} into {} chunks.'.format(parameter_file, self.num_of_chunks))

        self.CheckRunning()
        while (len(self.processes) > 0):
            time.sleep(0.1)
            self.CheckRunning()

        # parameter_output = os.path.join(self.output_dir, 'parameters.json')
        # with open(parameter_output, mode='x') as fparams:
        #     for i in range(self.num_of_chunks):
        #         file = os.path.join(self.output_dir,
        #                             'params_chunk_{}.json'.format(i))
        #         with open(file) as f:
        #              fparams.write(f.read())
        #         os.remove(file)

        # chunks.remove_chunks()
