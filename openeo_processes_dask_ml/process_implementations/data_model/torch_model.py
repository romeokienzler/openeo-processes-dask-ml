from .data_model import MLModel

import torch
import xarray as xr


class TorchModel(MLModel):
    def create_model_object(self, filepath: str):
        # todo: consider checkpoint, JIT, export
        self._model_object = torch.jit.load(filepath)
        self._model_object.eval()

    def execute_model(self, batch: xr.DataArray) -> xr.DataArray:
        tensor = torch.from_numpy(batch.data)
        with torch.no_grad():
            out = self._model_object(tensor)

        self.postprocess_datacube_expression(out)

        out_postproc = self.postprocess_datacube_expression(out)
        # todo: what if input is not batched?
        # add an artificial batch dimension

        out_dims = self.model_metadata.output[0].result.dim_order
        out_cube = xr.DataArray(
            out_postproc.numpy(),
            dims=out_dims
        )
        return out_cube
