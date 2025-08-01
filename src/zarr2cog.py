import argparse
import os
import shutil
import sys
from urllib.parse import urlparse
from datetime import datetime
from pathlib import Path
from typing import Tuple


import boto3
import xarray as xr
import pystac
from shapely.geometry import Polygon, mapping
from xarray import DataArray

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from src.util import open_zarr

staging_dirs = []

# TODO: Permit user to define these
DRIVER_KWARGS = {}
DRIVER_OPTIONS = [
    'blocksize', 'compress', 'level', 'max_z_error', 'max_z_error_overview', 'quality', 'jxl_lossless',
    'jxl_effort', 'jxl_distance', 'jxl_alpha_distance', 'num_threads', 'nbits', 'predictor', 'bigtiff',
    'resampling', 'overview_resampling', 'warp_resampling', 'overviews', 'overview_count', 'overview_compress',
    'overview_quality', 'overview_predictor', 'geotiff_version', 'sparse_ok', 'statistics', 'tiling_scheme',
    'zoom_level', 'zoom_level_strategy', 'target_srs', 'res', 'extent', 'aligned_levels', 'add_alpha'
]


def open_zarr_dataset(zarr_url, zarr_access):
    parsed_url = urlparse(zarr_url)

    if parsed_url.scheme in ('', 'file'):
        print(f'Opening local zarr dataset at {zarr_url}')
        ds = xr.open_zarr(zarr_url, consolidated=True)
        print(ds)
        return ds

    session = boto3.Session(profile_name=os.getenv('AWS_PROFILE', None))
    client = session.client('s3')
    credentials = session.get_credentials().get_frozen_credentials()

    ds, stage_dir = open_zarr(zarr_url, zarr_access, client, credentials)

    if stage_dir is not None:
        staging_dirs.append(stage_dir)

    print(f'Opened zarr dataset at {zarr_url}')
    print(ds)
    return ds


def extract_bounds(data):
    """Extract spatial bounds from xarray DataArray.
    
    Args:
        data (xr.DataArray): Rasterized data with x/y coordinates
        
    Returns:
        tuple: (minx, miny, maxx, maxy) in EPSG:4326
    """
    x_coords = data.coords['x'].values
    y_coords = data.coords['y'].values
    
    minx, maxx = float(x_coords.min()), float(x_coords.max())
    miny, maxy = float(y_coords.min()), float(y_coords.max())
    
    return (minx, miny, maxx, maxy)


def create_stac_item(cog_path, datetime_obj, bounds, var_attrs, global_attrs, collection_id):
    """Create STAC Item from COG file metadata.
    
    Args:
        cog_path (str): Path to COG file
        datetime_obj (datetime): Timestamp for the data
        bounds (tuple): Spatial bounds (minx, miny, maxx, maxy)
        var_attrs (dict): Variable attributes from Zarr
        global_attrs (dict): Global attributes from Zarr dataset
        collection_id (str): Collection ID to reference
        
    Returns:
        pystac.Item: STAC item object
    """
    # Create bounding box and geometry
    minx, miny, maxx, maxy = bounds
    bbox = [minx, miny, maxx, maxy]
    
    # Create polygon geometry
    geometry = mapping(Polygon.from_bounds(minx, miny, maxx, maxy))
    
    # Generate item ID from filename
    item_id = Path(cog_path).stem
    
    # Create STAC item
    item = pystac.Item(
        id=item_id,
        geometry=geometry,
        bbox=bbox,
        datetime=datetime_obj,
        properties={}
    )
    
    # Add variable-specific properties
    for key, value in var_attrs.items():
        if isinstance(value, (str, int, float, bool)):
            item.properties[f"var:{key}"] = value
    
    # Add selected global properties
    global_props = ['Title', 'Institution', 'Source', 'Conventions']
    for prop in global_props:
        if prop in global_attrs:
            item.properties[f"global:{prop}"] = global_attrs[prop]
    
    # Add COG asset with relative path
    # Extract just the filename from the full path for relative referencing
    cog_filename = os.path.basename(cog_path)
    
    # Create relative path from STAC item location to COG file
    # STAC items will be nested: collections/{collection_id}/{item_id}/{item_id}.json
    # COG files are in root output directory, so we need to go up 3 levels
    relative_cog_path = f"../../../{cog_filename}"
    
    item.add_asset(
        key="cog",
        asset=pystac.Asset(
            href=relative_cog_path,
            media_type="image/tiff; application=geotiff; profile=cloud-optimized",
            roles=["data"]
        )
    )
    
    # Link to collection
    item.collection_id = collection_id
    
    return item


def create_stac_collection_from_items(items, global_attrs, concept_id, var_name):
    """Create STAC Collection from actual items with computed extents.
    
    Args:
        items (list): List of STAC items for this collection
        global_attrs (dict): Global attributes from Zarr dataset
        
    Returns:
        pystac.Collection: STAC collection with proper extents
    """
    # Get collection info from first item
    first_item = items[0]
    collection_id = first_item.collection_id
    # Create collection
    collection = pystac.Collection(
        id=collection_id,
        description=f"{concept_id} {var_name} data collection",
        extent=None
    )
    
    collection.add_items(items)
    collection.update_extent_from_items()
    
    # Add global properties at collection level
    collection.extra_fields = global_attrs.copy()
    
    # Add metadata derived from collection_id
    collection.extra_fields["variable_name"] = var_name
    collection.extra_fields["concept_id"] = concept_id
    
    return collection


def create_stac_catalog(concept_id, all_collections, global_attrs):
    """Create STAC catalog with collections and items.
    
    Args:
        concept_id (str): Base concept identifier
        all_collections (list): All STAC collections created
        global_attrs (dict): Global attributes from Zarr dataset
        output_dir (str): Output directory path
        
    Returns:
        pystac.Catalog: Root STAC catalog
    """
    # Create root catalog
    catalog = pystac.Catalog(
        id=f"{concept_id}-catalog",
        description=f"STAC catalog for {concept_id} dataset"
    )
    
    # Add global dataset properties to catalog
    catalog.extra_fields = {}
    global_props = ['Title', 'Institution', 'Source', 'Conventions']
    for prop in global_props:
        if prop in global_attrs:
            catalog.extra_fields[f"global:{prop}"] = global_attrs[prop]
    catalog.extra_fields["concept_id"] = concept_id
    
    for collection in all_collections:        
        # Add collection to catalog
        catalog.add_child(collection)
    
    return catalog


def convert_timeslice_to_cog(input_data: DataArray, time, var_name, lat_c, lon_c, 
                             output_filename_prefix,
                             output_dir="output") -> Tuple[DataArray, str]:
    data = input_data.sel(time=time)
    data = data.rio.write_crs("epsg:4326")
    data = data.rename({lon_c: 'x', lat_c: 'y'})
    dt = time.values.astype('datetime64[s]').item()
    data.attrs = {k.upper(): v for k, v in data.attrs.items()}

    try:
        latitude = data['y'].to_numpy()

        if latitude[1] - latitude[0] >= 0:
            print(f'Flipping latitude for {var_name}')
            data = data.isel({'y': slice(None, None, -1)})
    except Exception as e:
        print(f'Could not check latitude ordering for {var_name} due to {e}')

    filename = f'{output_filename_prefix}_{dt.strftime("%Y-%m-%dT%H%M%SZ")}_{var_name}.tif'

    out_path = os.path.join(output_dir, filename)

    print(f'Writing timestep {dt} to {out_path}')

    data.rio.to_raster(out_path, driver='COG', sharing=False, **DRIVER_KWARGS)
    return data, out_path


def main(args):
    zarr_url = args.zarr
    time_c = args.time
    lat_c = args.latitude
    lon_c = args.longitude

    ds = open_zarr_dataset(zarr_url, args.zarr_access)

    print(f'{len(ds.data_vars)} variables, {len(ds[time_c])} time steps')
    
    # Track all created collections for catalog creation
    all_collections = []

    for var_name in ds.data_vars:
        print(f'Iterating over variable {var_name}')
        var_items = []
        da = ds[var_name]

        for time in da[time_c]:
            dt = time.values.astype('datetime64[s]').item()
            data, tif_file = convert_timeslice_to_cog(da, time, var_name, lat_c, lon_c, args.output)
            
            # Create STAC item for this COG file
            try:
                # Extract spatial bounds
                bounds = extract_bounds(data)
                
                # Create collection ID
                collection_id = f"{args.concept_id}-{var_name}"
                
                # Create STAC item
                stac_item = create_stac_item(
                    cog_path=tif_file,
                    datetime_obj=dt,
                    bounds=bounds,
                    var_attrs=data.attrs,
                    global_attrs=ds.attrs,
                    collection_id=collection_id
                )
                
                # Add to catalog items list
                var_items.append(stac_item)
                
            except Exception as e:
                print(f'Warning: Failed to create STAC item for {tif_file}: {e}')
        if var_items:
            all_collections.append(create_stac_collection_from_items(var_items, ds.attrs, args.concept_id, var_name))

    # Create and save complete STAC catalog
    if all_collections:
        print(f"\nCreating STAC catalog from {len(all_collections)} collections...")
        try:
            catalog = create_stac_catalog(
                args.concept_id, 
                all_collections,
                ds.attrs
            )
            
            # Save catalog structure
            catalog.normalize_and_save(
                root_href='output',
                catalog_type=pystac.CatalogType.SELF_CONTAINED
            )
            
            collections = list(catalog.get_collections())
            print(f"✓ Created STAC catalog with {len(collections)} collections")
            for collection in collections:
                item_count = len(list(collection.get_items()))
                print(f"  - {collection.id}: {item_count} items")
            print(f"✓ Catalog saved to: output/catalog.json")
            
        except Exception as e:
            print(f'Warning: Failed to create STAC catalog: {e}')
    else:
        print("No STAC items created, skipping catalog generation")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument(
        'zarr',
        help='S3 URL zarr data to convert'
    )

    parser.add_argument(
        '--zarr-access',
        required=False,
        default='stage',
        choices=['stage', 'mount'],
        help='stage: Download zarr data from S3 to local filesystem; mount: mount S3 to local filesystem'
    )

    parser.add_argument(
        '-t', '--time',
        default='time',
        help='Name of the time coordinate'
    )

    parser.add_argument(
        '--latitude',
        default='latitude',
        help='Name of the latitude coordinate'
    )

    parser.add_argument(
        '--longitude',
        default='longitude',
        help='Name of the longitude coordinate'
    )

    parser.add_argument(
        '-o', '--output',
        required=False,
        default='cog',
        help='Output cog filename prefix'
    )

    parser.add_argument(
        '--concept_id',
        required=True,
        help='Concept ID for STAC collection naming (e.g., "C1276812838-GES_DISC")'
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
