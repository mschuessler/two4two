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
    else
      echo "We do not support your OS: ``uname`` - Only supporting Linux and Mac OS"
      exit 1
    fi
    curl -L "$PIP_DOWNLOAD" -o "get-pip.py"

    PIP3="$PYTHON -m pip"
    $PYTHON get-pip.py
    $PIP3 install -U pip
    $PYTHON -m venv "$OUTDIR/venv"
    source "$OUTDIR/venv/bin/activate"
    # use a fixed pip version that fits to the libraries below
    >&2 python -m pip install pip==19.0.3
    # >&2 prints installation to stdout
    >&2 python -m pip install \
            coverage==5.5      \
            cycler==0.10.0     \
            decorator==4.4.2   \
            imageio==2.9.0     \
            kiwisolver==1.3.1  \
            matplotlib==3.3.4  \
            networkx==2.5      \
            numpy==1.17.0      \
            Pillow==8.1.2      \
            pyparsing==2.4.7   \
            python-dateutil==2.8.1  \
            PyWavelets==1.1.1       \
            scikit-image==0.18.0    \
            scipy==1.6.1            \
            six==1.15.0             \
            tifffile==2021.3.17
)
