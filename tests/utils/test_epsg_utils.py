import numpy as np
import pytest
import xarray as xr
from dask import array as da
from stackstac.raster_spec import RasterSpec

from openeo_processes_dask_ml.process_implementations.exceptions import (
    ReferenceSystemNotFound,
)
from openeo_processes_dask_ml.process_implementations.utils import epsg_utils


@pytest.fixture
def dc() -> xr.DataArray:
    x = xr.DataArray(da.random.random(5), dims=["x"], coords={"x": range(5)})
    return x


def test_epsg_not_found(dc: xr.DataArray):
    with pytest.raises(ReferenceSystemNotFound):
        epsg_utils.get_epsg_from_datacube(dc)


def test_epsg_coord(dc: xr.DataArray):
    dc.coords["epsg"] = np.array(25832)
    epsg = epsg_utils.get_epsg_from_datacube(dc)
    assert epsg == 25832


def test_spatial_ref_coord(dc: xr.DataArray):
    dc.coords["spatial_ref"] = np.array(25832)
    epsg = epsg_utils.get_epsg_from_datacube(dc)
    assert epsg == 25832


def test_spec_att(dc: xr.DataArray):
    r = RasterSpec(25832, (1, 1, 2, 2), (10, 10))
    dc.attrs["spec"] = r
    epsg = epsg_utils.get_epsg_from_datacube(dc)
    assert epsg == 25832


def test_crs_att(dc: xr.DataArray):
    dc.attrs["crs"] = "epsg:25832"
    epsg = epsg_utils.get_epsg_from_datacube(dc)
    assert epsg == 25832
