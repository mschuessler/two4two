"""Comand line util to generate datasets."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import pdb
import pickle
import shutil
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import toml
import tqdm

import two4two
from two4two import utils


@dataclasses.dataclass
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
    sampler_config: dict[str, Any]
    split: str
    split_interventions: tuple[str, ...]
    output_dir: str
    n_samples: int
    force_overwrite: bool
    n_processes: int
    blender_dir: Optional[str]
    interventions: tuple[tuple[str, ...], ...]
    unbiased: bool
    download_blender: bool
    debug: bool
    run_xgb: bool = True
    xgb_n_train: int = 10_000
    xgb_n_test: int = 2_000

    def get_original(self) -> InterventionArgs:
        """Returns the original split without any interventions."""
        return InterventionArgs(
            modified_attributes=self.split_interventions,
            split_args=self,
            dirname=self.split,

        )

    def get_interventions(self) -> Sequence[InterventionArgs]:
        """Returns the all the interventions."""
        return [
            InterventionArgs(
                modified_attributes=interventions,
                split_args=self,
                dirname=f'{self.split}_{"_".join(interventions)}',
            )
            for interventions in self.interventions]

    def load_sampler(self) -> two4two.Sampler:
        """Returns the sampler object."""
        sampler_cls = utils.import_class(*utils.split_class(self.sampler))
        return sampler_cls(**self.sampler_config)


@dataclasses.dataclass
class InterventionArgs:
    """Arguments for a single intervention split.

    Args:
        modified_attributes: List of attributes to resample.
        split_args: The render arguments of the split.
        dirname: Directory name
    """
    modified_attributes: tuple[str, ...]
    split_args: RenderSplitArgs
    dirname: str

    @staticmethod
    def from_dict(state: dict[str, Any]) -> InterventionArgs:
        """Loads InterventionArgs from dictonary."""
        return InterventionArgs(
            modified_attributes=tuple(state['modified_attributes']),
            split_args=RenderSplitArgs(**state['split_args']),
            dirname=state['dirname'],
        )

    def is_original(self) -> bool:
        """Returns `True` if no interventions are done."""
        return (
            self.modified_attributes == tuple(self.split_args.split_interventions)
            and self.dirname == self.split_args.split)

    def get_original_split(self) -> InterventionArgs:
        """Returns the original split without any interventions."""
        return InterventionArgs(
            modified_attributes=self.split_args.split_interventions,
            split_args=self.split_args,
            dirname=self.split_args.split,
        )

    def get_key(self) -> str:
        """Returns a unique key of split and all interventions."""
        if self.modified_attributes:
            return self.split_args.split + '_' + '_'.join(self.modified_attributes)
        else:
            return self.split_args.split


@dataclasses.dataclass
class XGBResult:
    """Result of a XGB classifier fitted on a dataset."""

    sampler: str
    interventions: Sequence[str]
    model: 'xgb.XGBModel'
    feature_importance: dict[str, float]
    accuracy: float
    n_train: int
    n_test: int

    def print_summary(self):
        """Prints summary of the result."""
        print(f"# Summary for XGB classifier {self.sampler}")
        if self.interventions:
            print(f"Interventions: {','.join(self.interventions)}")
        print(f"Test Accuracy: {self.accuracy:.4f}")
        print("Feature importances: ")
        print("Feature Name, Importance")
        for name, importance in self.feature_importance.items():
            print(f"{name}, {importance}")


def run_xgb(
    sampler: two4two.Sampler,
    unbiased: bool = False,
    interventions: Sequence[str] = [],
    n_train: int = 10_000,
    n_test: int = 2_000,
) -> XGBResult:
    import xgboost as xgb
    """Trains a XGB model on the given dataset."""

    n_samples = n_train + n_test
    if unbiased:
        params = [sampler.sample_unbiased() for _ in tqdm.trange(n_samples)]
    else:
        params = [sampler.sample() for _ in tqdm.trange(n_samples)]

    params = sampler.make_interventions(params, interventions)
    df = pd.DataFrame([dataclasses.asdict(param) for param in params])
    feature_names = [
        'spherical',
        'bending',
        'obj_rotation_roll',
        'obj_rotation_pitch',
        'obj_rotation_yaw',
        'fliplr',
        'position_x',
        'position_y',
        'arm_position',
        'obj_color',
        'bg_color'
    ]

    df_feat = df[feature_names]

    labels = np.array([
        two4two.scene_parameters.OBJ_NAME_TO_INT[name]
        for name in df['obj_name']])

    train_X = df_feat[:-n_test]
    test_X = df_feat[-n_test:]
    train_y = labels[:-n_test]
    test_y = labels[-n_test:]

    xgb_cls = xgb.XGBClassifier(max_depth=3, n_estimators=100, learning_rate=0.05)
    gbm = xgb_cls.fit(np.array(train_X), np.array(train_y))

    predictions = gbm.predict(np.array(copy.deepcopy(test_X)))
    sampler_cls = type(sampler)
    return XGBResult(
        sampler=f'{sampler_cls.__module__}.{sampler_cls.__qualname__}',
        interventions=interventions,
        model=gbm,
        feature_importance=dict(zip(df_feat.keys(),
                                    map(float, gbm.feature_importances_))),
        accuracy=float((predictions == test_y).mean()),
        n_train=n_train,
        n_test=n_test,
    )


all_attributes = [
    'arm_position',
    'bending',
    'bg_color',
    'obj_color',
    'obj_rotation_pitch',
    'obj_rotation_roll',
    'obj_rotation_yaw',
    'position_x',
    'position_y',
    'spherical',
]


def render_dataset_split(args: RenderSplitArgs):
    """Samples and renders a single split of the dataset."""
    sampler = args.load_sampler()

    two4two.blender.ensure_blender_available(args.blender_dir, args.download_blender)

    print("Sampling Parameters...")
    if args.unbiased:
        original_params = [sampler.sample_unbiased() for _ in tqdm.trange(args.n_samples)]
    else:
        original_params = [sampler.sample() for _ in tqdm.trange(args.n_samples)]

    original_args = args.get_original()
    params_per_split = {
        original_args.get_key(): (original_args, original_params)
    }

    interventions = args.get_interventions()

    for intervention in interventions:
        intervention_params = sampler.make_interventions(
            original_params, intervention.modified_attributes)

        params_per_split[intervention.get_key()] = (intervention, intervention_params)
    xgb_results: list[XGBResult] = []
    for key, (intervention, params) in params_per_split.items():
        split_name = intervention.dirname
        output_dir = os.path.join(args.output_dir, split_name)
        if os.path.exists(output_dir) and args.force_overwrite:
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=False)

        with open(os.path.join(output_dir, 'sampler.pickle'), 'wb') as f_pickle:
            pickle.dump(sampler, f_pickle)

        with open(os.path.join(output_dir, 'intervention.json'), 'w') as f_json:
            json.dump(dataclasses.asdict(intervention), f_json)

        if args.run_xgb:
            xgb_result = run_xgb(sampler,
                                 args.unbiased,
                                 intervention.modified_attributes,
                                 args.xgb_n_train, args.xgb_n_test)
            xgb_result.print_summary()

            with open(os.path.join(output_dir, 'xgb_results.pickle'), 'wb') as f_pickle:
                pickle.dump(xgb_result, f_pickle)
            xgb_results.append(xgb_result)

        print(f"Rendering {len(params)} images...")
        for _ in tqdm.tqdm(two4two.render(
            params,
            n_processes=args.n_processes,
            output_dir=output_dir,
            blender_dir=args.blender_dir,
            download_blender=args.download_blender,
            print_output=args.debug,
            print_cmd=args.debug,
        ), total=len(params), desc=split_name):
            pass

    output_dir = os.path.join(args.output_dir, args.split)
    with open(os.path.join(output_dir, 'xgb_accuracies.json'), 'w') as f_json:
        json.dump([
            {
                'sampler': r.sampler,
                'interventions': r.interventions,
                'feature_importance': r.feature_importance,
                'accuracy': r.accuracy,
            }
            for r in xgb_results], f_json)

    with open(os.path.join(output_dir, 'split_args.json'), 'w') as f_json:
        json.dump(dataclasses.asdict(args), f_json)

    xgb_results.append(xgb_result)


def load_configs(
    config_file: str,
    default_blender_dir: Optional[str] = None,
    default_download_blender: bool = False,
    split_by: int = 1,
    run_xgb: bool = False,
) -> list[RenderSplitArgs]:
    """Loads the config to render each dataset split."""
    with open(config_file) as f:
        dataset_configs = toml.load(f)

    config_none_ok = ['blender_dir', 'interventions', 'n_resample']

    args = []
    for config in dataset_configs['dataset']:
        sampler = config.pop('sampler')
        sampler_config = config.pop('sampler_config', {})
        split_interventions = tuple(config.pop('split_interventions', []))
        output_dir = config.pop('output_dir')
        blender_dir = config.pop('blender_dir', default_blender_dir)
        n_processes = config.pop('n_processes', 6)
        n_samples = config.pop('n_samples', None)
        force_overwrite = config.pop('force_overwrite', False)
        unbiased = config.pop('unbiased', False)
        download_blender = config.pop('download_blender', False) or default_download_blender
        debug = config.pop('debug', False)
        xgb_n_train = config.pop('xgb_n_train', 10_000)
        xgb_n_test = config.pop('xgb_n_test', 2_000)

        for dset_args in config['split']:
            dset_name = dset_args.pop('name')

            def get_arg(name: str, default: Any) -> Any:
                val = dset_args.pop(name, default)
                if val is None and name not in config_none_ok:
                    raise ValueError(
                        f"No value given for {name} in dataset {dset_name}.")
                return val

            interventions_list = dset_args.pop('interventions', [])
            interventions = tuple(tuple(intervention)
                                  for intervention in interventions_list)

            split_interventions = tuple(get_arg(
                'split_interventions', split_interventions))
            sampler_config.update(dset_args.pop('sampler_config', {}))

            n_samples = get_arg('n_samples', n_samples)
            assert n_samples % split_by == 0
            n_samples = n_samples // split_by

            cli_args = RenderSplitArgs(
                sampler=get_arg('sampler', sampler),
                sampler_config=sampler_config,
                split=dset_name,
                split_interventions=split_interventions,
                interventions=interventions,
                n_samples=n_samples,
                output_dir=output_dir,
                force_overwrite=get_arg('force_overwrite', force_overwrite),
                unbiased=get_arg('unbiased', unbiased),
                n_processes=get_arg('n_processes', n_processes),
                blender_dir=get_arg('blender_dir', blender_dir),
                download_blender=get_arg('download_blender', download_blender),
                run_xgb=run_xgb,
                xgb_n_train=get_arg('xgb_n_train', xgb_n_train),
                xgb_n_test=get_arg('xgb_n_test', xgb_n_test),
                debug=get_arg('debug', debug),
            )
            args.append(cli_args)
    return args


def render_dataset(
    config_file: Optional[str] = None,
    default_blender_dir: Optional[str] = None,
    default_download_blender: bool = False,
    split_by: int = 1,
    keep_only_tar: bool = False,
):
    """Entry point to render a dataset."""

    try:
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
            parser.add_argument(
                '--split_by',
                default=1,
                type=int,
                help='Divide number of samples (usefull for distributed sampling).',
            )
            parser.add_argument(
                '--keep-only-tar',
                default=False,
                action='store_true',
                help='Remove output dir and only keep `.tar` file.',
            )
            parser.add_argument(
                '--skip-xgb',
                default=False,
                action='store_true',
                help='Run xgb model.',
            )
            args = parser.parse_args()
            config_file = args.config_file[0]
            default_blender_dir = args.blender_dir
            default_download_blender = args.download_blender
            keep_only_tar = args.keep_only_tar
            split_by = args.split_by
            run_xgb = not args.skip_xgb

        configs = load_configs(
            config_file,
            default_blender_dir,
            default_download_blender,
            split_by,
            run_xgb,
        )

        for cli_args in configs:
            print("Rendering:")
            print(cli_args)
            print()
            render_dataset_split(cli_args)

        output_dirs = set([cli_args.output_dir for cli_args in configs])
        print(output_dirs)

        for output_dir in output_dirs:
            tar_name = f'{output_dir.rstrip("/")}.tar'
            utils.make_tarfile(tar_name, output_dir, compression=None)
            if keep_only_tar:
                print(f'Removing `{output_dir}`.')
                print(f'Keeping tar file: {tar_name}')
                shutil.rmtree(output_dir)
    except Exception:
        pdb.post_mortem()
