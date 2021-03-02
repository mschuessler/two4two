#! /usr/bin/env bash
#
# this script should pass before committing

set -e
# TODO(leon): add clean files of two4two/ to be flake checked for tests (see #3)

python -m pytest -v -s test/  \
    -m "not slow"

./run_flake8.sh

