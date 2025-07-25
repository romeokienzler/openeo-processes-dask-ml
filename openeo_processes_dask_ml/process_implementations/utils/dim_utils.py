import xarray as xr
from openeo_processes_dask.process_implementations.exceptions import DimensionMissing

from openeo_processes_dask_ml.process_implementations.exceptions import (
    BandNotFoundException,
)

band_dim_options = ["band", "bands", "b", "channel", "channels"]
time_dim_options = ["time", "times", "t", "date", "dates", "DATE"]
x_dim_options = ["x", "lon", "lng", "longitude"]
y_dim_options = ["y", "lat", "latitude"]
batch_dim_options = ["batch", "batches"]


def _find_alternative_dim_name_in_datacube(
    dc: xr.DataArray, dim_name_options: list[str]
) -> str:
    """
    Identify a dimension in a datacube based on a list of possible dimension names
    :param dc: The datacube
    :param dim_name_options: Dimension names to be searched for in the datacube
    :return: The found dimension name
    """
    for dim_option in dim_name_options:
        if dim_option in dc.dims:
            return dim_option
    raise ValueError(
        f"The datacube does not contain one of the following dimensions: "
        f"{', '.join(dim_name_options)}"
    )


def get_band_dim_name(dc: xr.DataArray) -> str:
    """
    Identifies the bands dimension in a data cube.
    :param dc: The datacube
    :return: Name of the band dimension
    :raise DimensionMissing: When no bands dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, band_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find a band dimension in the datacube. "
            f"If this is a mistake, rename the band dimension to one of the following: "
            f"{', '.join(band_dim_options)}"
        )


def get_time_dim_name(dc: xr.DataArray) -> str:
    """
    Identifies the time dimension in a data cube.
    :param dc: The datacube
    :return: Name of the time dimension
    :raise DimensionMissing: When no time dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, time_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find a time dimension in the datacube. "
            f"If this is a mistake, rename the time dimension to one of the following: "
            f"{', '.join(time_dim_options)}"
        )


def get_x_dim_name(dc: xr.DataArray) -> str:
    """
    Identifies the X dimension in a data cube.
    :param dc: The datacube
    :return: Name of the X dimension
    :raise DimensionMissing: When no X dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, x_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find an X dimension in the datacube. "
            f"If this is a mistake, rename the X dimension to one of the following: "
            f"{', '.join(time_dim_options)}"
        )


def get_y_dim_name(dc: xr.DataArray) -> str:
    """
    Identifies the Y dimension in a data cube.
    :param dc: The datacube
    :return: Name of the Y dimension
    :raise DimensionMissing: When no Y dimension could be identified.
    """
    try:
        return _find_alternative_dim_name_in_datacube(dc, y_dim_options)
    except ValueError:
        raise DimensionMissing(
            f"Could not find a Y dimension in the datacube. "
            f"If this is a mistake, rename the Y dimension to one of the following: "
            f"{', '.join(time_dim_options)}"
        )


def get_spatial_dim_names(dc: xr.DataArray) -> tuple[str, str]:
    """
    Identifies the spatial dimensions in a data cube.
    :param dc: The datacube
    :return: Names of the spatial dimensions
    :raise DimensionMissing: When a spatial dimension could not be identified.
    """
    x_dim_name = get_x_dim_name(dc)
    y_dim_name = get_y_dim_name(dc)
    return x_dim_name, y_dim_name


def get_alternative_datacube_dim_name(dc: xr.DataArray, dim_name: str) -> str | None:
    """
    Search for an alternative dimension name in the datacube.
    E.g. searching for "bands" dimension name will return "band" dimension if in DC.
    :param dc: The datacube to be searched for dimensions
    :param dim_name: The dimension name to search for
    :return: The name of the dimension in the datacube, or None if no match was found
    """

    dc_dims = dc.dims

    if dim_name in dc_dims:
        return dim_name

    t_dim_names = time_dim_options
    if dim_name in t_dim_names:
        return next((t_dim for t_dim in t_dim_names if t_dim in dc_dims), None)

    b_dim_names = band_dim_options
    if dim_name in b_dim_names:
        return next((b_dim for b_dim in b_dim_names if b_dim in dc_dims), None)

    x_dim_names = x_dim_options
    if dim_name in x_dim_names:
        return next((x_dim for x_dim in x_dim_names if x_dim in dc_dims), None)

    y_dim_names = y_dim_options
    if dim_name in y_dim_names:
        return next((y_dim for y_dim in y_dim_names if y_dim in dc_dims), None)

    batch_dim_names = band_dim_options
    if dim_name in batch_dim_names:
        return next((b_dim for b_dim in batch_dim_names if b_dim in dc_dims), None)

    return None


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
        # sentinel 1 bands
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
