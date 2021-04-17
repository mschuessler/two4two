"""Module to render images using blender."""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, Iterator, Optional, Sequence, Tuple

import imageio
import numpy as np

from two4two import scene_parameters
from two4two import utils


def _download_blender(blender_dir: str):
    """Downloads blender to the given directory."""
    download_script = os.path.join(os.path.dirname(__file__),
                                   'download_blender.sh')

    args = [
        'sh',
        '-c',
        f"{download_script} \"{blender_dir}\"",
    ]
    print(f"Downloading Blender to {blender_dir}")
    subprocess.check_output(args)


def ensure_blender_available(blender_dir: Optional[str] = None,
                             download_blender: bool = False):
    """Ensures blender is available in the given directory."""
    blender_dir = blender_dir or os.path.join(os.environ['HOME'], '.cache', 'two4two')

    blender_binary = os.path.join(blender_dir, "blender_bin")
    found_blender = os.path.exists(blender_binary)

    if not found_blender and not download_blender:
        raise FileNotFoundError(
            "Please download blender version 2.83.9 or "
            "set ``download_blender=True`` to download it automatically!")
    elif not found_blender and download_blender:
        _download_blender(blender_dir)
    else:  # found blender
        pass


def _load_images_from_param_file(
    param_filename: str,
    delete: bool,
) -> Iterator[Tuple[np.ndarray, np.ndarray, scene_parameters.SceneParameters]]:
    """Yields tuples of image and scene parameters.

    Args:
        param_filename: read images from this jsonl parameter file.
        delete: delete images after reading them
    """
    with open(param_filename) as f:
        for line in f.readlines():
            params = scene_parameters.SceneParameters.load(json.loads(line))
            dirname = os.path.dirname(param_filename)
            img_fname = os.path.join(dirname, params.filename)
            mask_fname = os.path.join(dirname, params.mask_filename)
            img = imageio.imread(img_fname)
            mask = imageio.imread(mask_fname)
            yield img, mask, params
            if delete:
                os.remove(img_fname)


def _split_param_file(parameter_file: str, chunk_size: int) -> Sequence[str]:
    """Splits ``parameter_file`` into files with atmost ``chunk_size`` lines.

    Returns:
        List of chunked parameter files.
    """
    with open(parameter_file) as f:
        lines = f.readlines()

    num_chunks = len(lines) // chunk_size
    if len(lines) % chunk_size != 0:
        num_chunks += 1

    chunk_parameter_files = []
    name, ext = os.path.splitext(parameter_file)
    for idx in range(num_chunks):
        filename = f'{name}_chunk_{idx + 1}{ext}'
        chunk_parameter_files.append(filename)
        with open(filename, 'x') as f:
            f.writelines(lines[idx * chunk_size:(idx + 1) * chunk_size])
    return chunk_parameter_files


def _get_finished_processes(
    processes: Dict[str, subprocess.Popen],
    print_output: bool
) -> Sequence[str]:
    """Returns the keys of any finished processes."""
    finised_processes = []
    for chunk, process in processes.items():
        try:
            stdout, stderr = process.communicate(timeout=.2)
            if stdout is not None and print_output:
                print(stdout.decode('utf-8'))
            if stderr is not None and print_output:
                print(stderr.decode('utf-8'), file=sys.stderr)
        except subprocess.TimeoutExpired:
            if process.returncode is None:
                continue
        if process.returncode != 0:
            raise subprocess.CalledProcessError(
                process.returncode,
                process.args,
            )
        else:  # zero return code
            finised_processes.append(chunk)
    return finised_processes


SEGMENTATION_INT_TO_NAME: Dict[int, str] = {
    0: 'background',
    1: 'arm_left_top',
    2: 'arm_left_bottom',
    3: 'spine_left',
    4: 'spine_left_center',
    5: 'spine_right_center',
    6: 'spine_right',
    7: 'arm_right_top',
    8: 'arm_right_bottom',
}


SEGMENTATION_NAME_TO_INT: Dict[str, int] = {
    'background': 0,
    'arm_left_top': 1,
    'arm_left_bottom': 2,
    'spine_left': 3,
    'spine_left_center': 4,
    'spine_right_center': 5,
    'spine_right': 6,
    'arm_right_top': 7,
    'arm_right_bottom': 8,
}


def render(
    params: Sequence[scene_parameters.SceneParameters],
    n_processes: int = 0,
    chunk_size: int = 100,
    output_dir: Optional[str] = None,
    blender_dir: Optional[str] = None,
    download_blender: bool = False,
    print_output: bool = False,
    print_cmd: bool = False,
    save_blender_file: bool = False
) -> Iterator[Tuple[np.ndarray, np.ndarray, scene_parameters.SceneParameters]]:
    """Renders the given parameters to images using Blender.

    This function yields tuples of (image, mask, scene parameters). The order is
    not preserved. Each block of the object is encoded as a unique integer value
    in the mask. See ``SEGMENTATION_NAME_TO_INT``. For example, the left most
    spine is encoded as ``3``.


    Args:
        params: List of scence parameters.
        n_processes: Number of concurrent processes to run. The Default
            ``0`` means as many processes as cpus.
        chunk_size: Number of parameters to render per processes.
        output_dir: Save rendered images to this directory. If ``None``, the images
            will not be saves permanently.
        download_blender: flag to automatically downloads blender.
        blender_dir: blender directory to use. Default ``~/.cache/two4two``.
        print_output: Print the output of blender.
        print_cmd: Print executed subcommand (useful for debugging).
        save_blender_file: If ``True``, the blender file will be saved to
            "{params.id}.blender".

    Raises:
        FileNotFoundError: if no blender installation is found in ``blender_dir``.

    Yields:
        tuples of (image, mask, scene parameters).
    """

    def process_chunk():
        """Start a new subprocess if there is work to do."""
        nonlocal next_chunk
        parameter_file = parameter_chunks[next_chunk]

        execute_blender_script = os.path.join(
            os.path.dirname(__file__), 'execute_blender.sh')
        args = [
            execute_blender_script,
            blender_dir,
            render_script,
            parameter_file,
            output_dir,
            str(save_blender_file),
        ]
        if print_cmd:
            print("Command to execute Blender:")
            print(" ".join(args))
        env = os.environ.copy()
        # ensure no matplotlib backend is set for subprocess.
        env.pop('MPLBACKEND', None)
        proc = subprocess.Popen(args,
                                stdin=subprocess.PIPE,
                                stdout=subprocess.PIPE,
                                env=env)
        processes[parameter_file] = proc
        next_chunk += 1

    n_processes = n_processes or (os.cpu_count() or 1)

    blender_dir = blender_dir or os.path.join(os.environ['HOME'], '.cache', 'two4two')

    ensure_blender_available(blender_dir, download_blender)

    package_directory = os.path.dirname(__file__)
    render_script = os.path.join(package_directory,
                                 '_blender',
                                 'render_samples.py')

    # process and the processes chunk file
    processes: Dict[str, subprocess.Popen] = {}

    use_tmp_dir = output_dir is None
    try:
        output_dir = utils.get(output_dir, tempfile.mkdtemp())
        parameter_file = os.path.join(output_dir, 'parameters.jsonl')

        # dump parameters
        with open(parameter_file, 'x') as f:
            for param in params:
                f.write(json.dumps(param.state_dict()) + '\n')

        parameter_chunks = _split_param_file(parameter_file, chunk_size)
        num_of_chunks = len(parameter_chunks)
        next_chunk = 0

        while next_chunk < num_of_chunks or processes:
            finished_chunks = _get_finished_processes(processes, print_output)
            for chunk in finished_chunks:
                for img, mask, param in _load_images_from_param_file(chunk, delete=use_tmp_dir):
                    yield img, mask, param
                del processes[chunk]

            if len(processes) < n_processes and next_chunk < num_of_chunks:
                process_chunk()
    finally:
        if use_tmp_dir:
            shutil.rmtree(output_dir)  # type: ignore


def render_single(
    param: scene_parameters.SceneParameters,
    blender_dir: Optional[str] = None,
    download_blender: bool = False,
    print_output: bool = False,
    print_cmd: bool = False,
    save_blender_file: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """Convienience function. Renders a single images with Blender using the given SceneParameters.

    For rendering more than one image use ``blender.render`` which will be more efficent.
    Args:
        param: The scence parameters describing the image.
        output_dir: Save rendered images to this directory. If ``None``, the images
            will not be saves permanently.
        download_blender: flag to automatically downloads blender.
        blender_dir: blender directory to use. Default ``~/.cache/two4two``.
        print_output: Print the output of blender.
        print_cmd: Print executed subcommand (useful for debugging).
        save_blender_file: If ``True``, the blender file will be saved to
            "{params.id}.blender".

    Raises:
        FileNotFoundError: if no blender installation is found in ``blender_dir``.

    Yields:
        tuple of (rendered image, segmentation mask of image).
    """
    result = list(render([param],
                         n_processes=1,
                         chunk_size=1,
                         blender_dir=blender_dir,
                         download_blender=download_blender,
                         print_output=print_output,
                         print_cmd=print_cmd,
                         save_blender_file=save_blender_file))

    return (result[0][0], result[0][1])
