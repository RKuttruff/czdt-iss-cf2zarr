
import argparse
import json
import os
import shutil
import sys
import tempfile
from urllib.parse import urlparse

import boto3
import numpy as np
import pandas as pd
import xarray as xr

from zarr.codecs import BloscCodec as Blosc
from numcodecs.blosc import Blosc as BloscZ2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.util import open_zarr, get_config

staging_dirs = []


def __get_zarr_urls(args, client):
    if args.zarr is not None:
        return list(args.zarr)
    else:
        parsed_url = urlparse(args.zarr_manifest)

        with tempfile.NamedTemporaryFile(suffix='.json', mode='w+b') as temp:
            client.download_fileobj(parsed_url.netloc, parsed_url.path.lstrip('/'), temp)
            temp.seek(0)
            return json.load(temp)


def main(args):
    output = args.output

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')

    config = get_config(args.config, client)
    dim = config['dimensions']['time']

    datasets = []

    for z_url in __get_zarr_urls(args, client):
        credentials = session.get_credentials().get_frozen_credentials()
        ds, stage_dir = open_zarr(z_url, args.zarr_access, client, credentials)

        if stage_dir is not None:
            staging_dirs.append(stage_dir)

        print(f'Opened zarr dataset at {z_url}')

        datasets.append(ds)

    print(f'Opened {len(datasets):,} zarr datasets')

    ds = xr.concat(datasets, dim=dim).sortby(dim)

    print('New dataset:')
    print(ds)

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

    print(f'Setting chunk config: {chunk_config}')

    for var in ds.data_vars:
        ds[var] = ds[var].chunk(chunk_config)

    if args.zarr_version == 3:
        compressor = Blosc(cname="blosclz", clevel=9)
        encoding = {vname: {'compressors': [compressor]} for vname in ds.data_vars}
        to_zarr_kwargs = {}
    else:
        # TODO: There MUST be a much better way to detect we're converting from Zarr3 to Zarr2
        #  which requires clearing all encoding settings (leaving _FillValue since I think it may be important)
        if 'serializer' in ds[list(ds.data_vars)[0]].encoding:
            print('Detected conversion of zarr v3 data to zarr v2, clearing encoding data')

            for var in ds.variables:
                ds[var].encoding = {enc: ds[var].encoding[enc] for enc in ds[var].encoding if enc == '_FillValue'}

        compressor = BloscZ2(cname="blosclz", clevel=9)
        encoding = {vname: {'compressor': compressor} for vname in ds.data_vars}
        to_zarr_kwargs = {
            'consolidated': True
        }

    if output.startswith('s3://'):
        output_path = output
    else:
        output_path = os.path.join('output', output)

    print(f'Writing to zarr (v{args.zarr_version}) file: {output_path}')

    import warnings

    with warnings.catch_warnings(action='ignore'):
        ds.to_zarr(
            output_path,
            mode='w-',
            encoding=encoding,
            write_empty_chunks=False,
            zarr_format=args.zarr_version,
            **to_zarr_kwargs
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'config',
        nargs='?',
        default=None,
        help='Path to config file'
    )

    input_group = parser.add_mutually_exclusive_group(required=True)

    input_group.add_argument(
        '-z', '--zarr',
        nargs='+',
        help='S3 URLs of zarr data arrays to concatenate'
    )

    input_group.add_argument(
        '-m', '--zarr-manifest',
        help='S3 URL to file containing a simple JSON list of zarr input URLs'
    )

    parser.add_argument(
        '--zarr-access',
        required=False,
        default='stage',
        choices=['stage', 'mount'],
        help='stage: Download zarr data from S3 to local filesystem; mount: mount S3 to local filesystem'
    )

    parser.add_argument(
        '-d', '--duration',
        type=pd.Timedelta,
        default=None,
        help='If set, this is the maximum difference in max-min time of the output dataset. Defined as an ISO 8601 '
             'Duration (or anything else parseable by pandas.Timedelta)'
    )

    parser.add_argument(
        '-v', '--zarr-version',
        type=int,
        choices=[2, 3],
        default=3,
        help='Version of zarr standard to output'
    )

    parser.add_argument(
        '-o', '--output',
        required=True,
        help='Output zarr filename'
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
