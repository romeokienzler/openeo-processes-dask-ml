"""
Some helper function to obtain datacubes (helpful for development)
"""

import hashlib
import os
import pickle
from typing import Optional, Union

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

        if not os.path.exists(DATACUBE_CACHE_DIR):
            os.makedirs(DATACUBE_CACHE_DIR)

        with open(path, "wb") as file:
            pickle.dump(dc, file)

        return dc
