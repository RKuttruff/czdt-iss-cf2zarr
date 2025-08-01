
import argparse
import os
import shutil
import sys

import boto3
import numpy as np
import pandas as pd
import xarray as xr

from zarr.codecs import BloscCodec as Blosc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.util import stage_s3, open_zarr, get_config

staging_dirs = []


def main(args):
    pattern = args.pattern
    # variables = args.variables
    output = args.output

    variables = []

    for v in args.variables:
        variables.extend(v.split(','))

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')

    config = get_config(args.config, client)
    dim = config['dimensions']['time']

    if args.zarr not in {'', 'none'}:
        credentials = session.get_credentials().get_frozen_credentials()
        ds, stage_dir = open_zarr(args.zarr, args.zarr_access, client, credentials)

        if stage_dir is not None:
            staging_dirs.append(stage_dir)

        print('Opened existing zarr dataset')
        print(ds)
    else:
        ds = None
        print('No existing zarr dataset, starting a new one')

    input_stage_dir = stage_s3(args.input_s3, client)

    new_ds = xr.open_mfdataset(os.path.join(input_stage_dir, pattern)).sortby(dim)
    print('Opened new dataset from input NetCDF files')
    print(new_ds)

    if variables is None:
        variables = []

    variable_name = list(new_ds.data_vars.keys())[0]  # Automatically pick the first variable
    if len(variables) == 0:
        if ds is None:
            variables = [variable_name]
        else:
            variables = list(ds.data_vars)
    elif variables[0] == '*':
        print('All variables selected - skipping subselection')
        variables = []

    if variables:
        print(f'Subselecting vars: {variables}')
        new_ds = new_ds[variables]

    if ds is not None:
        ds = xr.concat((ds, new_ds), dim=dim).sortby(dim)
        print('Concatenated datasets')
        print(ds)
    else:
        ds = new_ds

    time_coord = config['coordinates']['time']

    # Dedup time steps

    times = ds[time_coord].to_numpy()

    if any(np.diff(times).astype(int) == 0):
        print(f'Warning: duplicate time steps detected')

        prev = None
        drop = []

        for i, v in enumerate(times.astype(int)):
            if v == prev:
                drop.append(i - 1)

            prev = v

        print(f'Dropping {len(drop):,} time steps at indices: {drop}')

        ds = ds.drop_duplicates(dim=dim, keep='first')

    if args.duration is not None:
        ds_duration = pd.Timedelta((ds[time_coord][-1] - ds[time_coord][0]).data.item())

        print(f'new dataset duration: {ds_duration}')

        if ds_duration > args.duration:
            print('Dataset duration exceeds max duration provided')

            idx = 0

            while pd.Timedelta((ds[time_coord][-1] - ds[time_coord][idx]).data.item()) > args.duration:
                idx += 1

            ds = ds.isel(time=slice(idx, None))

            print(f'Dropped {idx:,} time steps. New dataset duration: '
                  f'{pd.Timedelta((ds[time_coord][-1] - ds[time_coord][0]).data.item())}')

    chunk_config = {config['dimensions'][d]: config['chunks'][d] for d in config['chunks']}

    # exit()

    print(f'Setting chunk config: {chunk_config}')

    for var in ds.data_vars:
        ds[var] = ds[var].chunk(chunk_config)

    compressor = Blosc(cname="blosclz", clevel=9)
    encoding = {vname: {'compressor': compressor} for vname in ds.data_vars}

    if output.startswith('s3://'):
        output_path = output
    else:
        output_path = os.path.join('output', output)

    print(f'Writing zarr to {output_path}')

    ds.to_zarr(
        output_path,
        mode='w-',
        encoding=encoding,
        consolidated=True,
        write_empty_chunks=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        nargs='?',
        default=None,
        help='Path to config file'
    )

    parser.add_argument(
        '-i', '--input-s3',
        required=True,
        help='S3 URL prefix of input files to stage'
    )

    parser.add_argument(
        '-z', '--zarr',
        required=False,
        default='',
        help='S3 URL of existing zarr data to append to'
    )

    parser.add_argument(
        '--zarr-access',
        required=False,
        default='stage',
        choices=['stage', 'mount'],
        help='stage: Download zarr data from S3 to local filesystem; mount: mount S3 to local filesystem'
    )

    parser.add_argument(
        '-p', '--pattern',
        default='*.nc',
        help='Glob pattern to match'
    )

    parser.add_argument(
        '-d', '--duration',
        type=pd.Timedelta,
        default=None,
        help='If set, this is the maximum difference in max-min time of the output dataset. Defined as an ISO 8601 '
             'Duration (or anything else parseable by pandas.Timedelta)'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output zarr filename'
    )

    parser.add_argument(
        '--variables',
        required=False,
        nargs='*',
        help='Variables to convert'
    )

    args = parser.parse_args()

    print(args)

    try:
        main(args)
    finally:
        for sd in staging_dirs:
            try:
                print(f'Cleaning up staging dir: {sd}')
                shutil.rmtree(sd)
            except:
                print(f'Failed to remove staging dir: {sd}')
