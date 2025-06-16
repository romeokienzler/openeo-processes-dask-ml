import pytest
import xarray as xr
import dask.array as da

from openeo_processes_dask_ml.process_implementations.utils import dim_utils
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing
from openeo_processes_dask_ml.process_implementations.exceptions import (
    BandNotFoundException
)

@pytest.mark.parametrize(
    "band_dim_name", ["band", "bands", "b", "channel", "foobar"]
)
def test_get_band_dim_name(band_dim_name: str):
    dc = xr.DataArray(da.random.random((3,3)), dims=[band_dim_name, "x"])
    if band_dim_name != "foobar":
        assert dim_utils.get_band_dim_name(dc) == band_dim_name
    else:
        with pytest.raises(DimensionMissing):
            dim_utils.get_band_dim_name(dc)


@pytest.mark.parametrize(
    "band_name", ("b04", "B04", "foo")
)
def test_get_band_alternative_names(band_name: str):
    if band_name != "foo":
        alt_names = dim_utils.get_band_alternative_names(band_name)
        assert band_name.lower() in alt_names
        assert "red" in alt_names
    else:
        with pytest.raises(BandNotFoundException):
            dim_utils.get_band_alternative_names(band_name)


@pytest.mark.parametrize(
    "dc_bands, mlm_bands, dc_bands_selected",
    ((["red", "nir", "green"], ["red", "nir"], ["red", "nir"]),
     (["b04", "b08", "b09"], ["B04", "B08"], ["b04", "b08"]),
     (["B04", "B08", "foo"], ["red", "nir"], ["B04", "B08"])
    )
)
def test_get_dc_band_names(
        dc_bands: list[str], mlm_bands: list[str], dc_bands_selected: list[str]
):
    selected_bands = dim_utils.get_dc_band_names(dc_bands, mlm_bands)
    assert selected_bands == dc_bands_selected
