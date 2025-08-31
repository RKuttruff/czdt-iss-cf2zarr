import json
import os
import tempfile
from typing import Optional, Tuple
from urllib.parse import urlparse

import xarray as xr
import yamale
import yaml
from botocore.credentials import Credentials
from s3fs import S3FileSystem, S3Map

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCHEMA_PATH = os.path.join(SCRIPT_DIR, 'schema', 'dataset_schema.yaml')


DEFAULT_CONFIG = {
    'chunks': {
        'time': 24,
        'latitude': 90,
        'longitude': 90
    },
    'dimensions': {
        'time': 'time',
        'latitude': 'latitude',
        'longitude': 'longitude'
    },
    'coordinates': {
        'time': 'time',
        'latitude': 'latitude',
        'longitude': 'longitude'
    }
}


def _localize_config(url: str, client) -> str:
    """
    Ensure the config file is on the local fs. For local files, this is a no-op. For S3 URLs,
    download to a temporary location.

    :param url: URL or path of the config file
    :param client: S3 client
    :return: Localized config file path
    """

    parsed_url = urlparse(url)

    if parsed_url.scheme in {'file', ''}:
        return url.removeprefix('file://')
    elif parsed_url.scheme == 's3':
        fd, local_file = tempfile.mkstemp(prefix='config', suffix='_temp.yaml')

        with os.fdopen(fd, 'wb') as f:
            client.download_fileobj(parsed_url.netloc, parsed_url.path.lstrip('/'), f)

        return local_file
    else:
        raise ValueError(f'Unsupported URL scheme: {parsed_url.scheme}')


def get_config(
        path: str,
        client,
        schema_path: str = None,
        schema_validators=None,
        raw=False
) -> dict:
    if path is None:
        print('No config file provided, using default')
        config = DEFAULT_CONFIG
    else:
        path = _localize_config(path, client)

        if schema_path is None:
            schema_path = SCHEMA_PATH

        schema = yamale.make_schema(schema_path, validators=schema_validators)
        data = yamale.make_data(path)

        yamale.validate(schema, data, strict=True)

        with open(path, 'r') as fp:
            config_data = yaml.safe_load(fp)

        print(f'Validated and loaded config from {path}')

        if not raw:
            config = {
                'chunks': config_data['chunks'],
                'dimensions': config_data.get('dimensions', DEFAULT_CONFIG['dimensions']),
            }

            config['coordinates'] = config_data.get('coordinates', config['dimensions'])
        else:
            config = config_data

    print(f'Final config:\n{json.dumps(config, indent=2)}')

    return config


def stage_s3(prefix_url: str, client) -> str:
    staging_dir = tempfile.mkdtemp(dir=os.getcwd())

    print(f'Created data staging directory: {staging_dir}')

    parsed_url = urlparse(prefix_url)

    if parsed_url.scheme != 's3':
        raise ValueError(f'Expected s3 URL, got {parsed_url.scheme}')

    bucket = parsed_url.netloc
    prefix = parsed_url.path.lstrip('/')

    strip_index = prefix.rfind('/')
    if strip_index != -1:
        strip_prefix = prefix[:strip_index+1]
    else:
        strip_prefix = prefix

    paginator = client.get_paginator('list_objects_v2')

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in [o['Key'] for o in page.get('Contents', [])]:
            if obj.endswith('/'):
                head = client.head_object(Bucket=bucket, Key=obj)

                if head['ContentLength'] == 0:
                    print(f'Skipping directory object {obj}')
                    continue

            dst = os.path.join(staging_dir, obj.removeprefix(strip_prefix))
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            print(f'Downloading s3://{bucket}/{obj} to {dst}')
            client.download_file(bucket, obj, dst)

    return staging_dir


def open_zarr(zarr_url: str, method: str, client, credentials: Credentials) -> Tuple[xr.Dataset, Optional[str]]:
    if method == 'stage':
        print('Staging zarr data to local')
        local_dir = stage_s3(zarr_url.rstrip('/'), client)
        zarr_dir = os.path.join(local_dir, os.path.basename(zarr_url.rstrip('/')))

        print(f'Opening staged zarr data at {zarr_dir}')
        return xr.open_zarr(zarr_dir, consolidated=True), local_dir
    elif method == 'mount':
        s3 = S3FileSystem(
            False,
            key=credentials.access_key,
            secret=credentials.secret_key,
            token=credentials.token,
            client_kwargs=dict(region_name='us-west-2')
        )

        store = S3Map(root=zarr_url, s3=s3, check=False)
        return xr.open_zarr(store, consolidated=True), None
    else:
        raise ValueError(f'Unsupported zarr open method: {method}')
