"""test for ``two4two.cli_tool``."""

import os
import pathlib

from two4two import cli_tool
import two4two.scene_parameters


def test_cli_tool(tmp_path: pathlib.Path):
    """Runs comand line tool."""
    with open('test/data/test_datasets.toml') as f:
        toml_str = f.read()

    config_fname = str(tmp_path / 'test_datasets.toml')
    with open(config_fname, 'w') as f:
        f.write(toml_str.replace(
            'OUTPUT_DIR_PLACEHOLDER', str(tmp_path / 'no_bias')))

    cli_tool.render_dataset(config_fname)

    assert (tmp_path / 'no_bias' / 'train').exists()
    assert (tmp_path / 'no_bias' / 'train' / 'parameters.jsonl').exists()
    assert len(list((tmp_path / 'no_bias' / 'train').iterdir())) > 3

    param_fname = str(tmp_path / 'no_bias' / 'train' / 'parameters.jsonl')
    dataset_dir = str(tmp_path / 'no_bias' / 'train')

    for param in two4two.scene_parameters.load_jsonl(param_fname):
        assert os.path.exists(os.path.join(dataset_dir, param.filename))
        assert os.path.exists(os.path.join(dataset_dir, param.mask_filename))
