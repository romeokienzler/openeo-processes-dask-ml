from typing import Literal, Union
from abc import ABC, abstractmethod

task_enum = Literal[
    "regression",
    "classification",
    "scene-classification",
    "detection",
    "object-detection",
    "segmentation",
    "semantic-segmentation",
    "instance-segmentation",
    "panoptic-segmentation",
    "similarity-search",
    "generative",
    "image-captioning",
    "super-resolution"
]

framework_enum = Literal[
    "PyTorch",
    "TensorFlow",
    "scikit-learn",
    "Hugging Face",
    "Keras",
    "ONNX",
    "rgeo",
    "spatialRF",
    "JAX",
    "FLAX",
    "MXNet",
    "Caffe",
    "PyMC",
    "Weka",
    "Paddle"
]  # keep in mind that these are only recommendations

accelerator_enum = Literal[
    "amd64",
    "cuda",
    "xla",
    "amd-rocm",
    "intel-ipex-cpu",
    "intel-ipex-gpu",
    "macos-arm"
]  # keep in mind that these are only recommendations

dimension_name_enum = Literal[
    "batch",
    "channel",
    "time",
    "height",
    "width",
    "depth",
    "token",
    "class",
    "score",
    "confidence"
]  # recommended values only!

data_type_enum = Literal[
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
    "cint16",
    "cint32",
    "cfloat32",
    "cfloat64",
    "other"  # e.g. boolean, string, higher-precision number
]  # values taken from https://github.com/radiantearth/stac-spec/blob/master/commons/common-metadata.md#data-types


class MLModelBand:
    def __init__(
            self,
            name: str,
            format: str = None,
            expression=None
    ):
        self.name = name
        self.format = format
        self.expression = expression


class MLModelInputStructure:
    def __init__(
            self,
            shape: list[int],
            dim_order: list[dimension_name_enum],
            data_type: data_type_enum
    ):
        self.shape = shape
        self.dim_order = dim_order
        self.data_type = data_type


class MLModelOutputStructure:
    def __init__(
            self,
            shape: list[int],
            dim_order: list[data_type_enum],
            data_type: data_type_enum
    ):
        self.shape = shape
        self.dim_order = dim_order
        self.data_type = data_type


class MLModelInput:
    def __init__(
            self,
            name: str,
            bands: Union[list[str], list[MLModelBand], list[dict]],
            input: Union[MLModelInputStructure, dict],
            description: str = None,
            value_scaling: str = None,  # todo: value scaling object
            resize_type: str = None,  # todo: resize enum
            pre_processing_function: str = None  # todo: preprocessing expression
    ):

        if value_scaling is not None:
            raise NotImplementedError("value_scaling currently not implemented")
        if resize_type is not None:
            raise NotImplementedError("resize_type currently not implemented")
        if pre_processing_function is not None:
            raise NotImplementedError("pre_processing_function currently not implemented.")

        # resolve dicts to objects
        if type(input) is list:
            input = MLModelInputStructure(**input)

        # a list of strings is given for bands parameter
        if isinstance(bands, list) and all(isinstance(item, str) for item in a):
            bands = [MLModelBand(band_value) for band_value in bands]

        # a list of dicts is given as bands parameter
        if isinstance(bands, list) and all(isinstance(item, dict) for item in a):
            bands = [MLModelBand(**band_value) for band_value in bands]

        self.name = name
        self.bands = bands
        self.input = input
        self.description = description
        self.value_scaling = value_scaling
        self.resize_type = resize_type
        self.pre_processing_function = pre_processing_function


class MLModelOutput:
    def __init__(
            self,
            name: str,
            tasks: list[task_enum],
            result: Union[MLModelOutputStructure, dict],
            description: str = None,
            classification_classes: int = None,  # todo: classification extension
            post_processing_function: str = None   # todo preprocessing expression
    ):

        if classification_classes is not None:
            raise NotImplementedError("classification_classes currently not supported")
        if post_processing_function is not None:
            raise NotImplementedError("post_processing_function currently not supported")

        # resolve dict to object
        if type(result) is dict:
            result = MLModelOutputStructure(**result)

        self.name = name
        self.tasks = tasks
        self.results = result
        self.description = description
        self.classificaiton_classes = classification_classes
        self.post_processing_function = post_processing_function


class MLModel(ABC):
    def __init__(
            self,
            name: str,
            architecture: str,
            task: list[task_enum],
            input: Union[MLModelInput, dict],
            output: Union[MLModelOutput, dict],
            framework: framework_enum = None,
            framework_version: str = None,
            memory_size: int = None,
            total_parameters: int = None,
            pretrained: bool = None,
            pretrained_source: str = None,
            batch_size_suggestion: int = None,
            accelerator: accelerator_enum = None,
            accelerator_constrained: bool = None,
            accelerator_summary: str = None,
            accelerator_count: int = None,
            hyperparameters: dict = None,  # open JSON object
            **kwargs
    ):
        # resolve dict input to objects
        if type(input) is dict:
            input = MLModelInput(**input)
        if type(output) is dict:
            output = MLModelOutput(**output)

        self.name = name
        self.architecture = architecture
        self.task = task
        self.input = input
        self.output = output
        self.framework = framework
        self.framework_version = framework_version
        self.memory_size = memory_size
        self.total_parameters = total_parameters
        self.pretrained = pretrained
        self.pretrained_source = pretrained_source
        self.batch_size_suggestion = batch_size_suggestion
        self.accelerator = accelerator
        self.accelerator_constrained = accelerator_constrained
        self.accelerator_summary = accelerator_summary
        self.accelerator_count = accelerator_count
        self.hyperparameters = hyperparameters

    @abstractmethod
    def run_model(self):
        pass


class ONNXModel(MLModel):
    def run_model(self):
        pass


if __name__ == "__main__":
    a = [{1:1, 2:1}, {1:4, 4:2}]
    if isinstance(a, list) and all(isinstance(item, dict) for item in a):
        print("yes")
    else:
        print("No")