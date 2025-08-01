import argparse
import os
import re
import shutil
import sys
from datetime import datetime
from pathlib import PurePath
from typing import Tuple

import boto3
import numpy as np
import pandas as pd
import rioxarray
import xarray as xr
import yamale
import yaml

from odc.geo.geobox import GeoBox
from odc.geo.xr import xr_reproject as reproject
from rioxarray.merge import merge_datasets
from yamale.validators import Validator, DefaultValidators
from zarr.codecs import BloscCodec as Blosc

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, 'schema', 'geotiff_schema.yaml')
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.util import stage_s3, get_config

DT_UNITS = ['year', 'month', 'day', 'hour', 'minute', 'second', 'microsecond']
UNIT_STARTS = dict(year=0, month=1, day=1, hour=0, minute=0, second=0, microsecond=0)


staging_dirs = []


class PythonRegexValidator(Validator):
    tag = "py_re"

    def _is_valid(self, value):
        try:
            re.compile(value)
            return True
        except:
            return False


class GeoTiffBandMapValidator(Validator):
    tag = "geotiff_band_map"

    def _is_valid(self, value):
        # Value must be dict
        if not isinstance(value, dict):
            return False

        # Must contain at least one mapping
        if len(value) == 0:
            return False

        keys = list(value.keys())
        values = list(value.values())

        # Value must be dict of int -> str
        if any([not isinstance(k, int) for k in keys]):
            return False
        if any([not isinstance(v, str) for v in values]):
            return False

        # Dict keys must be integers starting at 1 and incrementing by one
        if set(keys) != set(range(1, len(keys)+1)):
            return False

        # Values must all be unique
        if len(set(values)) != len(values):
            return False

        return True

    def fail(self, value):
        return (f'{self.get_name()} must be a flat dict mapping incrementing integers starting at one to a set of '
                f'unique strings')


VALIDATORS = DefaultValidators.copy()
VALIDATORS[PythonRegexValidator.tag] = PythonRegexValidator
VALIDATORS[GeoTiffBandMapValidator.tag] = GeoTiffBandMapValidator


def _open_tiff(path, config):
    tiff_ds = rioxarray.open_rasterio(path).to_dataset('band').rename(config['band_map'])

    # print(f'debug (open1): {tiff_ds=}')

    # Coerce the data arrays to dask to handle very large datasets
    chunk_config = config.get('chunks', {
        'time': 24,
        'latitude': 90,
        'longitude': 90,
    })

    for var in tiff_ds.data_vars:
        tiff_ds[var] = tiff_ds[var].chunk(x=chunk_config['longitude'], y=chunk_config['latitude'])

    # print(f'debug (open2): {tiff_ds=}')

    return tiff_ds


def _get_bbox_from_config(config) -> Tuple[float, float, float, float]:
    if 'bbox' in config:
        return (
            config['bbox']['min_lon'],
            config['bbox']['min_lat'],
            config['bbox']['max_lon'],
            config['bbox']['max_lat'],
        )
    else:
        return -180.0, -90.0, 180.0, 90.0


def main(args):
    config_path = args.config
    pattern = args.pattern
    output = args.output

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')

    # schema = yamale.make_schema(SCHEMA_PATH, validators=VALIDATORS)
    # data = yamale.make_data(config_path)
    #
    # yamale.validate(schema, data, strict=True)
    #
    # with open(config_path, 'r') as fp:
    #     config = yaml.safe_load(fp)

    config = get_config(config_path, client, schema_path=SCHEMA_PATH, schema_validators=VALIDATORS, raw=True)

    input_stage_dir = stage_s3(args.input_s3, client)
    staging_dirs.append(input_stage_dir)

    times = {}
    filename_pattern = re.compile(config['filename_pattern'])

    input_tiffs = []

    for root, dirs, files in os.walk(input_stage_dir):
        for filename in files:
            path = os.path.join(root, filename)
            if PurePath(path).match(pattern):
                input_tiffs.append(path)

    if len(input_tiffs) == 0:
        raise ValueError('no tiffs found in input dir')

    chunk_config = config.get('chunks', {
        'time': 24,
        'latitude': 90,
        'longitude': 90,
    })

    for tiff in input_tiffs:
        match = filename_pattern.match(os.path.basename(tiff))
        if match is None:
            raise ValueError(f'Input tiff {os.path.basename(tiff)} does not match pattern {config["filename_pattern"]}')

        ts_string = match.groupdict()[config['timestamp']['group']]
        ts = datetime.strptime(ts_string, config['timestamp']['dt_string'])

        if 'round_down_to' in config['timestamp']:
            ts = ts.replace(
                **{u: UNIT_STARTS[u] for u in DT_UNITS[DT_UNITS.index(config['timestamp']['round_down_to'])+1:]}
            )

        print(f'Mapped input {tiff} to time {ts}')
        times.setdefault(ts, []).append(tiff)

    print(f'Mapped inputs to {len(times)} times')

    reprojected_slices = []
    resampling_method = config.get('resampling_method', 'nearest')

    gbox = GeoBox.from_bbox(
        _get_bbox_from_config(config),
        "epsg:4326",
        resolution=config['resolution_deg'],
    )

    print(f'Target bbox for mapping: {gbox}')

    for timestamp in sorted(times.keys()):
        print(f'Opening and merging {len(times[timestamp])} tiffs for timestamp {timestamp}')

        tiffs = [_open_tiff(f, config) for f in times[timestamp]]

        if len(tiffs) > 1:
            times[timestamp] = merge_datasets(tiffs)
        else:
            times[timestamp] = tiffs[0]

        # print(f'debug (postMerge): {times[timestamp]}')

        if times[timestamp].rio.crs.to_epsg() != 4326:
            if config['resolution_deg'] <= 0:
                raise ValueError('resolution_deg must be greater than zero')

            print('Reprojecting to EPSG:4326')
            reprojected = reproject(
                src=times[timestamp],
                how=gbox,
                resampling=resampling_method,
                dst_nodata=config.get('nodata', 'auto'),
            )

            for var in reprojected.data_vars:
                print(reprojected[var].dtype, times[timestamp][var].dtype)
                if reprojected[var].dtype != times[timestamp][var].dtype:
                    print(f'Casting {var} from {reprojected[var].dtype} back to {times[timestamp][var].dtype}')
                    reprojected[var] = reprojected[var].astype(times[timestamp][var].dtype)
        else:
            print('Data in required projection. Using native data')
            reprojected = times[timestamp].rename(x='longitude', y='latitude')

        # print(f'debug (post reproj): {reprojected}')
        # print(f'debug (post reproj): {reprojected[list(reprojected.data_vars)[0]]}')

        # if 'nodata' in config:
        #     for var in reprojected.data_vars:
        #         # print(f'Masking nodata values of {config["nodata"]} in {var}')
        #         # reprojected[var] = reprojected[var].where(reprojected[var] != config['nodata'])
        #         reprojected[var].attrs['_FillValue'] = config['nodata']

        print('Adding timestamp')
        reprojected = reprojected.expand_dims('time').assign_coords(
            time=[np.datetime64(timestamp, 'ns')]
        )

        print(f'Finished dataset for timestamp:\n{reprojected}')
        reprojected_slices.append(reprojected)

    final_ds = xr.concat(reprojected_slices, dim='time').sortby('time')
    print(f'Concatenated all timestamps into single dataset:\n{final_ds}')

    if args.duration is not None:
        ds_duration = pd.Timedelta((final_ds['time'][-1] - final_ds['time'][0]).data.item())

        print(f'new dataset duration: {ds_duration}')

        if ds_duration > args.duration:
            print('Dataset duration exceeds max duration provided')

            idx = 0

            while pd.Timedelta((final_ds['time'][-1] - final_ds['time'][idx]).data.item()) > args.duration:
                idx += 1

            final_ds = final_ds.isel(time=slice(idx, None))

            print(f'Dropped {idx:,} time steps. New dataset duration: '
                  f'{pd.Timedelta((final_ds["time"][-1] - final_ds["time"][0]).data.item())}')

    print(f'Setting chunk config: {chunk_config}')

    for var in final_ds.data_vars:
        final_ds[var] = final_ds[var].chunk(chunk_config)

    compressor = Blosc(cname="blosclz", clevel=9)
    encoding = {vname: {'compressor': compressor} for vname in final_ds.data_vars}

    if 'nodata' in config:
        for var in final_ds.data_vars:
            encoding[var]['fill_value'] = config['nodata']

    print(f'Writing to zarr file: {os.path.join("output", output)}')

    final_ds.to_zarr(
        os.path.join('output', output),
        mode='w-',
        encoding=encoding,
        consolidated=True,
        write_empty_chunks=False
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '-i', '--input-s3',
        required=True,
        help='S3 URL prefix of input files to stage'
    )

    parser.add_argument(
        '-c', '--config',
        required=True,
        help='YAML config file for input dataset'
    )

    parser.add_argument(
        '-p', '--pattern',
        default='*.tif',
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

    args = parser.parse_args()

    print(args, flush=True)

    try:
        main(args)
    finally:
        for sd in staging_dirs:
            try:
                print(f'Cleaning up staging dir: {sd}')
                shutil.rmtree(sd)
            except:
                print(f'Failed to remove staging dir: {sd}')
