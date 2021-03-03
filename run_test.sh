#! /usr/bin/env bash
#
# this script should pass before committing

set -e

python -m pytest -v -s test/  \
    -m "not slow"

./run_flake8.sh

