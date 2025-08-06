import torch
import xarray as xr

import pystac

from .data_model import MLModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TorchModel(MLModel):
    def __init__(self, stac_item: pystac.Item, model_asset_name: str = None):
        MLModel.__init__(self, stac_item, model_asset_name)

        self._model_on_device = None

    def create_model_object(self, filepath: str):
        # todo: consider checkpoint, JIT, export
        self._model_object = torch.jit.load(filepath)

    def init_model_for_prediction(self):
        self._model_on_device = self._model_object.to(DEVICE)
        self._model_on_device.eval()

    def uninit_model_after_prediction(self):
        self._model_on_device = self._model_on_device.to("cpu")
        del self._model_on_device
        self._model_on_device = None
        torch.cuda.empty_cache()

    def execute_model(self, batch: xr.DataArray) -> xr.DataArray:
        tensor = torch.from_numpy(batch.data).to(DEVICE)
        with torch.no_grad():
            out = self._model_on_device(tensor)

        self.postprocess_datacube_expression(out)

        out_postproc = self.postprocess_datacube_expression(out)

        out_dims = self.model_metadata.output[0].result.dim_order
        out_cube = xr.DataArray(out_postproc.numpy(), dims=out_dims)

        return out_cube
