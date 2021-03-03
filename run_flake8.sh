#! /usr/bin/env bash
#
# this script runs flake8 on all clean files.

set -e
echo "Running flake8..."



flake8 \
    ./test/test_blender.py \
    ./test/test_scene_parameters.py \
    ./setup.py \
    ./two4two/scene_parameters.py \
     ./two4two/utils.py \
    ./two4two/_blender/scene.py \
    ./two4two/_blender/render_samples.py \
    ./two4two/_blender/__init__.py \
    ./two4two/__init__.py \
    ./two4two/blender.py \

# Reset
Color_Off='\033[0m'       # Text Reset
Green='\033[0;32m'        # Green

echo -e "${Green}PASSED$Color_Off"

# TODO: clean following files
# ./two4two/_blender/butils.py
# ./two4two/_blender/blender_object.py
