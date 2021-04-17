"""Comand line util to generate datasets."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import pickle
import shutil
from typing import Any, Optional, Tuple

import toml
import tqdm

import two4two
from two4two import utils


@dataclasses.dataclass(frozen=True)
class RenderSplitArgs:
    """Comand line arguments.

    Args:
        sampler: module and class of the sample (e.g `two4two.Sampler`)
        n_samples: number of samples to draw
        output_dir: save split to this directory
        force_overwrite: overwrite existing directory
        n_processes: number of concurrent process
        blender_dir: blender directory
        download_blender: download blender if not found
        debug: print blender debug information
    """
    sampler: str
    n_samples: int
    output_dir: str
    force_overwrite: bool
    n_processes: int
    blender_dir: Optional[str]
    resample_attributes: Tuple[str]
    n_resample: int
    download_blender: bool
    debug: bool


def render_dataset_split(args: RenderSplitArgs):
    """Samples and renders a single split of the dataset."""
    sampler_cls = utils.import_class(*utils.split_class(args.sampler))
    sampler: two4two.Sampler = sampler_cls()

    two4two.blender.ensure_blender_available(args.blender_dir, args.download_blender)

    print("Sampling Parameters...")
    params = [sampler.sample() for _ in tqdm.trange(args.n_samples)]
    original_params = copy.copy(params)

    for attr_name in args.resample_attributes:
        for _ in range(args.n_resample):
            for original_param in original_params:
                # TODO(leon) this is unfortunatly not what we want to do.
                # We need to sample from the marginal distribution.
                param = original_param.clone()
                sample_method = getattr(sampler, 'sample_' + attr_name)
                sample_method(param)
                params.append(param)

    if os.path.exists(args.output_dir) and args.force_overwrite:
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=False)

    with open(os.path.join(args.output_dir, 'sampler.pickle'), 'wb') as f_pickle:
        pickle.dump(sampler, f_pickle)

    with open(os.path.join(args.output_dir, 'args.json'), 'w') as f_json:
        json.dump(dataclasses.asdict(args), f_json)

    print(f"Rendering {len(params)} images...")
    for _ in tqdm.tqdm(two4two.render(
        params,
        n_processes=args.n_processes,
        output_dir=args.output_dir,
        blender_dir=args.blender_dir,
        download_blender=args.download_blender,
        print_output=args.debug,
        print_cmd=args.debug,
    ), total=len(params)):
        pass


def render_dataset(
    config_file: Optional[str] = None,
    default_blender_dir: Optional[str] = None,
    default_download_blender: bool = False,
):
    """Entry point to render a dataset."""

    if config_file is None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            'config_file',
            help='Path to the config file. See `datasets.toml` for an example.',
            nargs=1)
        parser.add_argument(
            '--download_blender',
            default=False,
            action='store_true',
            help='Download blender if not found.',
        )
        parser.add_argument(
            '--blender_dir',
            default=None,
            help='Download blender to this directory.',
        )
        args = parser.parse_args()
        config_file = args.config_file[0]
        default_blender_dir = args.blender_dir
        default_download_blender = args.download_blender

    with open(config_file) as f:
        dataset_configs = toml.load(f)

    config_none_ok = ['blender_dir', 'resample_attributes', 'n_resample']

    for dataset_name, config in dataset_configs.items():
        sampler = config.pop('sampler')
        output_dir = config.pop('output_dir')
        blender_dir = config.pop('blender_dir', default_blender_dir)
        n_processes = config.pop('n_processes', 6)
        n_samples = config.pop('n_samples', None)
        resample_attributes = config.pop('resample_attributes', tuple())
        n_resample = config.pop('n_resample', 0)
        force_overwrite = config.pop('force_overwrite', False)
        download_blender = config.pop('download_blender', False) or default_download_blender
        debug = config.pop('debug', False)

        for dset_name, dset_args in config.items():
            def get_arg(name: str, default: Any) -> Any:
                val = dset_args.pop(name, default)
                if val is None and name not in config_none_ok:
                    raise ValueError(
                        f"No value given for {name} in dataset {dset_name}.")
                return val

            cli_args = RenderSplitArgs(
                sampler=get_arg('sampler', sampler),
                n_samples=get_arg('n_samples', n_samples),
                output_dir=os.path.join(output_dir, dset_name),
                force_overwrite=get_arg('force_overwrite', force_overwrite),
                n_processes=get_arg('n_processes', n_processes),
                blender_dir=get_arg('blender_dir', blender_dir),
                resample_attributes=get_arg('resample_attributes', resample_attributes),
                n_resample=get_arg('n_resample', n_resample),
                download_blender=get_arg('download_blender', download_blender),
                debug=get_arg('debug', debug),
            )
            if len(dset_args):
                unparsed_keys = ", ".join(list(args.keys()))
                raise ValueError(f"The following keys where to not parsed: {unparsed_keys}")
            render_dataset_split(cli_args)
