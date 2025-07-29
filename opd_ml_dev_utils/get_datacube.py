"""
Some helper function to obtain datacubes (helpful for development)
"""

import hashlib
import os
import pickle
from typing import Optional, Union

import pystac_client
import stackstac
import xarray as xr
from dask import array as da
from openeo_pg_parser_networkx.pg_schema import BoundingBox, TemporalInterval
from openeo_processes_dask.process_implementations.cubes import load_stac

from openeo_processes_dask_ml.process_implementations.constants import (
    DATACUBE_CACHE_DIR,
)


def _secure_hash_objects(*args):
    """Computes a stable, cryptographic hash from arbitrary objects."""
    hasher = hashlib.sha256()
    for obj in args:
        # Use repr() to get a more stable string representation than str()
        obj_bytes = repr(obj).encode("utf-8")
        hasher.update(obj_bytes)
    return hasher.hexdigest()


def _write_datacube_to_cache(datacube: xr.DataArray, path: str):
    if not os.path.exists(DATACUBE_CACHE_DIR):
        os.makedirs(DATACUBE_CACHE_DIR)
    with open(path, "wb") as file:
        pickle.dump(datacube, file)


def get_random_datacube(shape: tuple[int, ...], dims: tuple[str, ...]) -> xr.DataArray:
    if len(shape) != len(dims):
        raise ValueError("Length of shape and dim attributes must be the same")

    coords = {dim_name: range(dim_len) for dim_name, dim_len in zip(dims, shape)}

    dc = xr.DataArray(da.random.random(shape), dims=dims, coords=coords)

    return dc


def get_datacube_from_pickle_file(path: str) -> xr.DataArray:
    with open(path, "rb") as file:
        dc = pickle.load(file)
    if not isinstance(dc, xr.DataArray):
        raise TypeError("The provided file is not an xarray DataArray")
    return dc


def get_datacube_from_stackstac(
    bbox: tuple[float, float, float, float],
    time_period: str,  # Format: 'YYYY-MM-DD/YYYY-MM-DD' or 'YYYY-MM-DDTHH:MM:SSZ/YYYY-MM-DDTHH:MM:SSZ'
    collection: str = "sentinel-2-l2a",
    stac_url: str = "https://earth-search.aws.element84.com/v1",
    assets: list[str] = ["nir", "red"],  # Blue, Green, Red, NIR
    epsg: int = None,  # Target EPSG code, defaults to the most common for the area
    resolution: int = 10,  # Target resolution in meters
    cloud_cover_max: float = None,  # Maximum cloud cover percentage (0-100)
    chunksize: int = 2048,  # Spatial chunk size for Dask (useful for larger areas/times)
) -> xr.DataArray:
    """
    Forms an xarray datacube of Sentinel-2 images for a given bounding box
    and time period from the Element 84 STAC catalog using stackstac.

    Args:
        bbox (list[float]): Bounding box as [minx, miny, maxx, maxy].
        time_period (str): Time range as 'YYYY-MM-DD/YYYY-MM-DD'.
        collection (str): The STAC collection ID (default: 'sentinel-2-l2a').
        stac_url (str): The URL of the STAC catalog (default: Element 84).
        assets (list[str]): List of band names (asset keys) to include.
                             Defaults to B02, B03, B04 (10m visible) and B08 (10m NIR).
        epsg (int): The target EPSG code for the output datacube. If None,
                    stackstac attempts to find a suitable UTM zone.
        resolution (int): The target spatial resolution in meters.
        cloud_cover_max (float): Maximum percentage of cloud cover allowed for items.
        chunksize (int): Spatial chunk size for Dask. Set to None for no chunking
                         (loads everything into memory).

    Returns:
        xr.DataArray: An xarray DataArray containing the stacked Sentinel-2 imagery,
                      or None if no items were found.
    """

    hash_val = _secure_hash_objects(
        bbox,
        time_period,
        collection,
        stac_url,
        assets,
        epsg,
        resolution,
        cloud_cover_max,
        chunksize,
    )

    filename = "stackstac_" + hash_val + ".pickle"
    path = os.path.join(DATACUBE_CACHE_DIR, filename)

    if os.path.exists(path):
        return get_datacube_from_pickle_file(path)
    else:
        client = pystac_client.Client.open(stac_url)
        search_params = {
            "collections": [collection],
            "bbox": bbox,
            "datetime": time_period,
            "query": {},
        }
        if cloud_cover_max is not None:
            search_params["query"]["eo:cloud_cover"] = {"lt": cloud_cover_max}
        search = client.search(**search_params)

        # Get the item collection (lazy loading)
        item_collection = search.item_collection()

        if not item_collection:
            raise Exception("No items found for the specified criteria.")

        dc_lazy = stackstac.stack(
            item_collection,
            assets=assets,
            epsg=epsg,
            resolution=resolution,
            chunksize=chunksize,
            bounds_latlon=bbox,
        )

        dc = dc_lazy.compute()
        _write_datacube_to_cache(dc, path)

        return dc


def load_stac_with_cache(
    url: str,
    spatial_extent: Optional[BoundingBox] = None,
    temporal_extent: Optional[TemporalInterval] = None,
    bands: Optional[list[str]] = None,
    properties: Optional[dict] = None,
    resolution: Optional[float] = None,
    projection: Optional[Union[int, str]] = None,
    resampling: Optional[str] = None,
) -> xr.DataArray:
    hash_val = _secure_hash_objects(
        url,
        spatial_extent,
        temporal_extent,
        bands,
        properties,
        resolution,
        projection,
        resampling,
    )

    filename = hash_val + ".pickle"
    path = os.path.join(DATACUBE_CACHE_DIR, filename)

    if os.path.exists(path):
        return get_datacube_from_pickle_file(path)
    else:
        dc_lazy = load_stac(
            url,
            spatial_extent,
            temporal_extent,
            bands,
            properties,
            resolution,
            projection,
            resampling,
        )
        dc = dc_lazy.compute()

        _write_datacube_to_cache(dc, path)

        return dc
