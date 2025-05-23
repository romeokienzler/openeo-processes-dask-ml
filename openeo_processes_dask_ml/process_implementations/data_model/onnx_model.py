from .data_model import MLModel

import onnx
import onnxruntime as ort


class ONNXModel(MLModel):
    def create_model_object(self, model_filepath: str):

        onnx_model = onnx.load(model_filepath)

        try:
            onnx.checker.check_model(onnx_model, full_check=True)
        except Exception as e:
            raise Exception(
                f"The given model does not pass ONNX validation full_check "
                f"(onnx.checker.check_model).\n"
                f"{str(e)}"
            )

        self._model_object = ort.InferenceSession(model_filepath)

    def execute_model(self, batch):
        pass
