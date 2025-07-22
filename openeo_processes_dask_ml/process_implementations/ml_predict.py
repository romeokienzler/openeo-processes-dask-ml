import xarray as xr

from .data_model import MLModel


def ml_predict(
    data: xr.DataArray, model: MLModel, dimension: list[str]
) -> xr.DataArray:
    out = model.run_model(data)
    return out
