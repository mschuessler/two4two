"""Comand line util to generate datasets."""

from __future__ import annotations

import argparse
import dataclasses
import os
import shutil
import textwrap
from typing import List, Optional

import tqdm

import two4two
from two4two import utils


@dataclasses.dataclass
class ComandLineArgs:
    """Comand line arguments."""
    sampler: str
    n_samples: int
    output_dir: str
    force_overwrite: bool
    n_processes: int
    blender_dir: str
    download_blender: bool
    debug: bool

    @staticmethod
    def parse_args(argv: Optional[List[str]] = None) -> ComandLineArgs:
        """Parses the comand line arguments."""
        parser = argparse.ArgumentParser(description='Generate Two4Two Datasets.')
        parser.add_argument(
            '--sampler',
            help=textwrap.dedent("""\
            class name of the sampler. Can be the "Sampler" class or a custom sampler
            of the form "your.module.YourSampler". If a custom sampler is given, the
            module must be importable from the current enviroment.
            """),
            type=str, required=True,
        )
        parser.add_argument(
            '--n_samples',
            help='Number of concurrent processes.',
            type=int, required=False,
            default=0,
        )
        parser.add_argument(
            '--output_dir',
            help='save dataset to this directory.',
            type=str, required=True,
        )
        parser.add_argument(
            '--force_overwrite',
            help='Overwrites the output directory if exists.',
            default=False, action='store_true',
        )
        parser.add_argument(
            '--n_processes',
            help='Number of concurrent processes.',
            required=False, default=0, type=int,
        )
        parser.add_argument(
            '--blender_dir',
            help='Directory of the downloaded blender.',
            default=None, required=False,
            type=str,
        )
        parser.add_argument(
            '--download_blender',
            help='Automatically download blender if not found.',
            default=False, action='store_true',
        )
        parser.add_argument(
            '--debug',
            help='Debug flag to print blender outputs.',
            default=False, action='store_true',
        )
        args = parser.parse_args(args=argv)
        return ComandLineArgs(
            args.sampler,
            args.n_samples,
            args.output_dir,
            args.force_overwrite,
            args.n_processes,
            args.blender_dir,
            args.download_blender,
            args.debug,
        )


def main():
    """The main method of the comand line util."""
    args = ComandLineArgs.parse_args()
    # import sampler class
    sampler_cls = utils.import_class(*utils.split_class(args.sampler))
    sampler = sampler_cls()

    print("Sampling Parameters...")
    params = [sampler.sample() for _ in tqdm.trange(args.n_samples)]

    if os.path.exists(args.output_dir) and args.force_overwrite:
        shutil.rmtree(args.output_dir)
    os.makedirs(args.output_dir, exist_ok=False)

    print("Rendering...")
    for _ in tqdm.tqdm(two4two.render(
        params,
        n_processes=args.n_processes,
        output_dir=args.output_dir,
        blender_dir=args.blender_dir,
        download_blender=args.download_blender,
        print_output=args.debug,
        print_cmd=args.debug,
    ), total=args.n_samples):
        pass
