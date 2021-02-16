#! /usr/bin/env bash
#
# this script should pass before committing

set -e
# TODO(leon): also add two4two to checked directory (see #3)
python -m pytest -v --flake8 -s test/  -m "not slow"
