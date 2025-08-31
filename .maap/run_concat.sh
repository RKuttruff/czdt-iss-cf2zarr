#!/usr/bin/env bash

# Get current location of build script
basedir=$( cd "$(dirname "$0")" ; pwd -P )
root_dir=$(dirname "${basedir}")

mkdir -p output
source activate cf2zarr
python -u "${root_dir}"/src/zarr_concat.py \
  $([ -n "$1" ] && echo $1)\
  --zarr-manifest $2 \
  --zarr-access $3 \
  $([[ $4 != "none" ]] && echo --duration $4) \
  --zarr-version $5 \
  --output $6
