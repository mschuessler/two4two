#! /usr/bin/env bash

set -e  # error -> exit
set -u  # undef -> error

blender_dir=$1
render_script=$2
parameter_file=$3
output_dir=$4

source "$blender_dir/venv/bin/activate"


echo "# Script Arguments:"
echo "render_script: $render_script"
echo "parameter_file: $parameter_file"
echo "output_dir: $output_dir"
echo ""
echo "# Enviroment:"
printenv

echo ""

blender_bin=$(cat $blender_dir/blender_bin)

# PYTHONPATH enables venv packages for blender python
PYTHONPATH="$blender_dir/venv/lib/python3.7/site-packages" \
    $blender_bin \
        --background \
        -noaudio \
        --python-use-system-env \
        --python \
        $render_script \
        -- \
        $parameter_file \
        $output_dir \
