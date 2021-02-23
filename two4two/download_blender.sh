#! /usr/bin/env bash
# Script to download blender to the given directory.

# stop at errors
set -e
# a undefined variable is an error
set -u

# blender download directory
OUTDIR="$1"

BLENDER_DOWNLOAD="https://download.blender.org/release/Blender2.83/blender-2.83.9-linux64.tar.xz"
PIP_DOWNLOAD="https://bootstrap.pypa.io/get-pip.py"

mkdir -p "$OUTDIR"

(
    cd $OUTDIR
    if [[ ! -e "blender.tar.xz" ]]; then
        curl -L "$BLENDER_DOWNLOAD" -o "blender.tar.xz"
    fi
    curl -L "$PIP_DOWNLOAD" -o "get-pip.py"
    tar -xf blender.tar.xz
    PYTHON="blender/2.83/python/bin/python3.7m"
    PIP3="$PYTHON -m pip"
    mv blender-2.83.9-linux64 blender
    $PYTHON get-pip.py
    $PIP3 install -U pip
    $PYTHON -m venv "$OUTDIR/venv"
    source "$OUTDIR/venv/bin/activate"
    # print installation to stdout
    >&2 python -m pip install numpy==1.17.0 scipy==1.6.1 matplotlib==3.3.4 scikit-image==0.18.0
)

