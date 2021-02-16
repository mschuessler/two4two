from typing import Sequence, Iterator, Optional, Tuple
import subprocess
import time
import json
import tempfile
import os

import numpy as np
import imageio

from two4two import scene_parameters


def render(
    params: Sequence[scene_parameters.SceneParameters],
    n_processes: int = 0,
    chunk_size: int = 100,
    output_dir: Optional[str] = None,
) -> Iterator[Tuple[np.array, scene_parameters.SceneParameters]]:
    """
    Renders the given parameters to images using Blender.

    Args:
        params: List of scence parameters.
        n_processes: Number of concurrent processes to run. The Default
            ``0`` means as many processes as cpus.
        chunk_size: Number of parameters to render per processes.
        output_dir: Save rendered images to this directory.

    Yields:
        tuples of (image, scene parameters).
    """

    def process_chunk():
        """ Start a new subprocess if there is work to do """
        nonlocal next_chunk
        if next_chunk < num_of_chunks:
            parameter_file = parameter_chunks[next_chunk]
            args = [
                blender_path,
                '--background',
                '-noaudio',
                '--python',
                render_script,
                '--',
                parameter_chunks[next_chunk],
                output_dir,
            ]
            proc = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=subprocess.PIPE)
            processes.append((proc, parameter_file))
            next_chunk += 1

    def read_from_param_file(
        param_filename: str
    ) -> Iterator[Tuple[np.array, scene_parameters.SceneParameters]]:
        with open(param_filename) as f:
            for line in f.readlines():
                params = scene_parameters.SceneParameters(
                    **json.loads(line))
                img_fname = os.path.join(os.path.dirname(param_filename),
                                         params.filename)
                yield imageio.imread(img_fname), params

    def maybe_start_new_process() -> Iterator[Tuple[np.array, scene_parameters.SceneParameters]]:
        """ Check any running processes and start new ones if there are spare slots."""
        # Check the processes in reverse order

        for p in range(len(processes), 0, -1):
            p -= 1
            # If the process hasn't finished will return None
            process, param_file = processes[p]
            if process.poll() is not None:
                # Remove from list - this is why we needed reverse order
                del processes[p]
                for img, params in read_from_param_file(param_file):
                    yield img, params

        # More to do and some spare slots
        while (len(processes) < n_processes) and (next_chunk < num_of_chunks):
            process_chunk()

    def split_param_file(parameter_file: str, chunk_size: int) -> Sequence[str]:
        with open(parameter_file) as f:
            lines = f.readlines()

        num_chunks = len(lines) // chunk_size
        if len(lines) % chunk_size != 0:
            num_chunks += 1

        splits = []
        name, ext = os.path.splitext(parameter_file)
        for idx in range(num_chunks):
            filename = f'{name}_chunk_{idx + 1}{ext}'
            splits.append(filename)
            with open(filename, 'x') as f:
                f.writelines(lines[idx * chunk_size:(idx + 1) * chunk_size])
        return splits

    if n_processes == 0:
        n_processes = os.cpu_count() or 1

    package_directory = "REPLACE-WITH-PWD"
    blender_path = os.path.join(package_directory, 'blender/blender')
    render_script = os.path.join(package_directory, 'two4two/render_samples.py')
    chunk_size = chunk_size

    # process and the processes chunk file
    processes: Tuple[subprocess.Popen, str] = []

    use_tmp_dir = output_dir is None
    if use_tmp_dir:
        output_dir = tempfile.TemporaryDirectory()

    parameter_file = os.path.join(output_dir, 'parameters.json')

    # dump parameters
    with open(parameter_file, 'x') as f:
        for param in params:
            f.writelines([json.dumps(param.state_dict())])

    parameter_chunks = split_param_file(parameter_file, chunk_size)
    num_of_chunks = len(parameter_chunks)
    next_chunk = 0

    while True:
        for img, param in maybe_start_new_process():
            yield img, param
        time.sleep(0.1)
        if next_chunk == num_of_chunks and not processes:
            break
