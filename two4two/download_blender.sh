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
    $PIP3 install numpy scipy matplotlib ipykernel
)

