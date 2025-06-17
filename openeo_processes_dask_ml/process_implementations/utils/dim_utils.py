import xarray as xr

from openeo_processes_dask.process_implementations.exceptions import DimensionMissing
from openeo_processes_dask_ml.process_implementations.exceptions import (
    BandNotFoundException
)
_band_dim_options = ["band", "bands", "b", "channel", "channels"]


def get_band_dim_name(dc: xr.DataArray) -> str:
    """
    Identifies the bands dimension in a data cube.
    :param dc: The datacube
    :return: Name of the band dimension
    :raise DimensionMissing: When no bands dimension could be identified.
    """
    band_dim_name = None
    for b in _band_dim_options:
        if b not in dc.dims:
            continue
        band_dim_name = b

    if not band_dim_name:
        raise DimensionMissing(
            f"The loaded model requires a bands dimension in its input, but none "
            f"was found. If this is a mistake, please rename the bands dimension "
            f"to one of the following: {', '.join(_band_dim_options)}"
        )

    return band_dim_name


def get_band_alternative_names(band_name: str) -> list[str]:
    """
    Get alternative names for a band name, e.g. "b04" -> "red"
    :param band_name: The band to find altenative names
    :return: list of possible band name alternatives
    """
    band_name_lower = band_name.lower()
    band_name_groups = [
        # sentienl 2 bands
        ["b01", "coastal"],
        ["b02", "b2", "blue"],
        ["b03", "b3", "green"],
        ["b04", "b4", "red"],
        ["b05", "b5", "rededge1"],
        ["b06", "b6", "rededge2"],
        ["b07", "b7", "rededge3"],
        ["b08", "b8", "nir"],
        ["b8a", "b08a", "nir08", "nir08", "nir08a"],
        ["b09", "b9", "nir09"],
        ["b10", "cirrus"],
        ["b11", "swir16"],
        ["b12", "swir22"],
        ["aot"],
        ["scl"],
        ["snw"],

        #sentinel 1 bands
        ["hh"],
        ["hv"],
        ["vh"],
        ["vv"],

    ]

    for band_name_group in band_name_groups:
        if band_name_lower in band_name_group:
            return band_name_group

    raise BandNotFoundException(
        f"Could not find band name alternatives for band {band_name}"
    )


def get_dc_band_names(dc_band_names: list[str], band_names: list[str]) -> list[str]:
    """
    Get a list of band coordinates from the datacube that correspond to the list of band
    names submitted to the function, also considering alternative band names
    :param dc_band_names: band coordinates from the data cube
    :param band_names: the list of band names
    :return: list of band names in data cube
    """

    bands_in_dc = []

    for b_name in band_names:
        if b_name in dc_band_names:
            bands_in_dc.append(b_name)
            continue

        alt_names = get_band_alternative_names(b_name)
        for alt_name in alt_names:
            lower_dc_names = [n.lower() for n in dc_band_names]
            if alt_name in lower_dc_names:
                match = dc_band_names[lower_dc_names.index(alt_name)]
                bands_in_dc.append(match)
                break

    return bands_in_dc

