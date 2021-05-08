"""Comand line util to generate datasets."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import os
import pickle
import shutil
from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
import toml
import tqdm
import xgboost as xgb

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
    split: str
    output_dir: str
    n_samples: int
    force_overwrite: bool
    n_processes: int
    blender_dir: Optional[str]
    interventions: tuple[tuple[str]]
    download_blender: bool
    debug: bool
    xgb_n_train: int = 10_000
    xgb_n_test: int = 2_000


@dataclasses.dataclass
class XGBResult:
    """Result of a XGB classifier fitted on a dataset."""

    sampler: str
    interventions: Sequence[str]
    model: xgb.XGBModel
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
    interventions: Sequence[str] = [],
    n_train: int = 10_000,
    n_test: int = 2_000,
) -> XGBResult:
    """Trains a XGB model on the given dataset."""

    params = [sampler.sample() for _ in range(n_train + n_test)]
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


def render_dataset_split(args: RenderSplitArgs):
    """Samples and renders a single split of the dataset."""
    sampler_cls = utils.import_class(*utils.split_class(args.sampler))
    sampler: two4two.Sampler = sampler_cls()

    two4two.blender.ensure_blender_available(args.blender_dir, args.download_blender)

    print("Sampling Parameters...")
    original_params = [sampler.sample() for _ in tqdm.trange(args.n_samples)]

    params_per_split = {
        args.split: original_params,
    }

    for intervention_attributes in args.interventions:
        intervention_key = f'{args.split}_{"_".join(intervention_attributes)}'
        params_per_split[intervention_key] = sampler.make_interventions(
            original_params, intervention_attributes)

    # first split has no interventions

    interventionss = [(), ] + list(args.interventions)  # type: ignore

    xgb_results: list[XGBResult] = []
    for (split_name, params), interventions in zip(
            params_per_split.items(), interventionss):
        output_dir = os.path.join(args.output_dir, split_name)
        if os.path.exists(output_dir) and args.force_overwrite:
            shutil.rmtree(output_dir)

        os.makedirs(output_dir, exist_ok=False)

        with open(os.path.join(output_dir, 'sampler.pickle'), 'wb') as f_pickle:
            pickle.dump(sampler, f_pickle)

        with open(os.path.join(output_dir, 'args.json'), 'w') as f_json:
            json.dump(dataclasses.asdict(args), f_json)

        xgb_result = run_xgb(sampler, interventions,
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
    xgb_results.append(xgb_result)


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

    config_none_ok = ['blender_dir', 'interventions', 'n_resample']

    for config in dataset_configs['dataset']:
        sampler = config.pop('sampler')
        output_dir = config.pop('output_dir')
        blender_dir = config.pop('blender_dir', default_blender_dir)
        n_processes = config.pop('n_processes', 6)
        n_samples = config.pop('n_samples', None)
        force_overwrite = config.pop('force_overwrite', False)
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

            cli_args = RenderSplitArgs(
                sampler=get_arg('sampler', sampler),
                split=dset_name,
                interventions=dset_args.pop('interventions', []),
                n_samples=get_arg('n_samples', n_samples),
                output_dir=output_dir,
                force_overwrite=get_arg('force_overwrite', force_overwrite),
                n_processes=get_arg('n_processes', n_processes),
                blender_dir=get_arg('blender_dir', blender_dir),
                download_blender=get_arg('download_blender', download_blender),
                xgb_n_train=get_arg('xgb_n_train', xgb_n_train),
                xgb_n_test=get_arg('xgb_n_test', xgb_n_test),
                debug=get_arg('debug', debug),
            )
            if len(dset_args):
                unparsed_keys = ", ".join(list(dset_args.keys()))
                raise ValueError(f"The following keys where to not parsed: {unparsed_keys}")
            render_dataset_split(cli_args)
