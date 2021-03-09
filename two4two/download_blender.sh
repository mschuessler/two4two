#! /usr/bin/env bash
# Script to download blender to the given directory.

# stop at errors
set -e
# a undefined variable is an error
set -u

# blender download directory
OUTDIR="$1"

BLENDER_DOWNLOAD_LINUX="https://download.blender.org/release/Blender2.83/blender-2.83.9-linux64.tar.xz"
BLENDER_DOWNLOAD_MAC="https://download.blender.org/release/Blender2.83/blender-2.83.9-macOS.dmg"
PIP_DOWNLOAD="https://bootstrap.pypa.io/get-pip.py"

mkdir -p "$OUTDIR"

(
    cd $OUTDIR

    if [ "$(expr substr $(uname -s) 1 5)" == "Linux" ]; then
      if [[ ! -e "blender.tar.xz" ]]; then
          curl -L "$BLENDER_DOWNLOAD_LINUX" -o "blender.tar.xz"
      fi
      tar -xf blender.tar.xz
      echo "`pwd`/blender-2.83.9-linux64/blender" > blender_bin
      PYTHON="`pwd`/blender-2.83.9-linux64/2.83/python/bin/python3.7m"
    elif [ "$(uname)" == "Darwin" ]; then
      if [[ ! -e "blender.dmg" ]]; then
        curl -L "$BLENDER_DOWNLOAD_MAC" -o "blender.dmg"
      fi
      hdiutil attach blender.dmg
      cp -r /Volumes/Blender/Blender.app .
      echo "`pwd`/Blender.app/Contents/MacOS/Blender" > blender_bin
      PYTHON="`pwd`/Blender.app/Contents/Resources/2.83/python/bin/python3.7m"
    fi
    curl -L "$PIP_DOWNLOAD" -o "get-pip.py"

    PIP3="$PYTHON -m pip"
    $PYTHON get-pip.py
    $PIP3 install -U pip
    $PYTHON -m venv "$OUTDIR/venv"
    source "$OUTDIR/venv/bin/activate"
    # print installation to stdout
    >&2 python -m pip install numpy==1.17.0 scipy==1.6.1 matplotlib==3.3.4 scikit-image==0.18.0
)
