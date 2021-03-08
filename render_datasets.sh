#! /usr/bin/env bash

set -e

data_dir="../data/no_bias"

mkdir -p $data_dir

two4two_create_dataset \
  --sampler "two4two.Sampler" \
  --n_samples 60_000 \
  --output_dir "$data_dir/train" \
  --n_processes 4 \
  --download_blender


two4two_create_dataset \
  --sampler "two4two.Sampler" \
  --n_samples 10_000 \
  --output_dir "$data_dir/validation" \
  --n_processes 4 \
  --download_blender


two4two_create_dataset \
  --sampler "two4two.Sampler" \
  --n_samples 10_000 \
  --output_dir "$data_dir/test" \
  --n_processes 4 \
  --download_blender
