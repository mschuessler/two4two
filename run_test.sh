#! /usr/bin/env bash
#
# this script should pass before committing

set -e
# TODO(leon): add clean files of two4two/ to be flake checked for tests (see #3)

CLEAN_FILES=two4two/blender.py
python -m pytest -v --flake8 -s test/ \
    "$CLEAN_FILES" \
    -m "not slow"
