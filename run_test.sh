#! /usr/bin/env bash
#
# this script should pass before committing

set -e
# TODO(leon): add clean files of two4two/ to be flake checked for tests (see #3)

python -m pytest -v --flake8 -s test/ \
    two4two/blender.py \
    two4two/scene_parameters.py \
    -m "not slow"
