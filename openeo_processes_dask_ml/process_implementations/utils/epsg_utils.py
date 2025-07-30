import numpy as np
import stackstac.raster_spec
import xarray as xr

from openeo_processes_dask_ml.process_implementations.exceptions import (
    ReferenceSystemNotFound,
)


def get_epsg_from_datacube(dc: xr.DataArray) -> int:
    """
    Searches the datacube for common locations to store the EPSG COde
    :param dc: The datacube
    :return: EPSG Code (int)
    """
    coords = dc.coords
    attrs = dc.attrs

    # 1) Search for EPSG in Datacube coordinates
    if "epsg" in coords:
        # print(coords["epsg"])
        try:
            ref = coords["epsg"]
            if np.issubdtype(ref.dtype, np.integer):
                epsg = int(ref.data)
                return epsg
        except:
            pass

    if "spatial_ref" in coords:
        ref = coords["spatial_ref"]
        if np.issubdtype(ref.dtype, np.integer):
            try:
                epsg = int(ref.data)
                return epsg
            except:
                pass

    # do not use proj:code or proj:epsg:
    # they show epsg of data from which DC was constructed
    # DC epsg could be different to to e.g. reprojection

    # 2) Search for EPSG in attributes
    if "spec" in attrs:
        try:
            spec = attrs["spec"]
            epsg = int(spec.epsg)
            return epsg
        except:
            pass

    if "crs" in attrs:
        ref = attrs["crs"]  # attrs["crs"] should look like "epsg:xxxx"
        try:
            crs = ref.split(":")
            if crs[0] == "epsg":
                epsg = int(crs[1])
                return epsg
        except:
            pass

    raise ReferenceSystemNotFound(
        "Could not find Reference System EPSG code in datacube"
    )
