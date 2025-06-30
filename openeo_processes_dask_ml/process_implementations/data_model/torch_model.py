from .data_model import MLModel

import torch


class TorchModel(MLModel):
    def create_model_object(self, filepath: str):
        # todo: consider checkpoint, JIT, export
        self._model_object = torch.jit.load(filepath)
        self._model_object.eval()

    def execute_model(self, batch):
        tensor = torch.from_numpy(batch.data)
        with torch.no_grad():
            out = self._model_object(tensor)
