#! /usr/bin/env bash

Color_Off='\033[0m'
Red='\033[0;31m'
Green='\033[0;32m'

function fail {
    echo -e "${Red}FAILED: $1$Color_Off"
    exit 1
}


set -e

two4two_create_dataset --help

# create temporary output dir
cwd=`pwd`
tmp_dir=`mktemp -d`

echo "Output directory: $tmp_dir"

two4two_create_dataset \
  --sampler "two4two.Sampler" \
  --n_samples 2 \
  --output_dir "$tmp_dir" \
  --n_processes 1 \
  --force_overwrite \
  --download_blender


test -e $tmp_dir/parameters.jsonl || fail "did not found parameters."
ls $tmp_dir/*.png || fail "did not find rendered images."
