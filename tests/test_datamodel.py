import pytest
from openeo_processes_dask_ml.process_implementations.data_model import MLModelBand
from openeo_processes_dask_ml.process_implementations.data_model import MLModelInputStructure, MLModelResultStructure
from openeo_processes_dask_ml.process_implementations.data_model import MLModelInput, MLModelOutput


class TestMLModelBand:
    data = {"name": "B04", "format": "asdf"}

    def test_dict_construction(self):
        obj = MLModelBand(**self.data)
        assert obj.name == self.data["name"]
        assert obj.format == self.data["format"]


class TestMLModelInputStructure:
    data = {
        "shape": [-1, 3, 64, 64],
        "dim_order": ["batch", "channel", "width", "height"],
        "data_type": "int8"
    }

    def test_dict_construction(self):
        obj = MLModelInputStructure(**self.data)
        assert obj.shape == self.data["shape"]
        assert obj.dim_order == self.data["dim_order"]
        assert obj.data_type == self.data["data_type"]


class TestMLModelResultStructure:
    data = {
        "shape": [-1, 1, 64, 64],
        "dim_order": ["batch", "class", "width", "height"],
        "data_type": "int8"
    }

    def test_dict_construction(self):
        obj = MLModelResultStructure(**self.data)
        assert obj.shape == self.data["shape"]
        assert obj.dim_order == self.data["dim_order"]
        assert obj.data_type == self.data["data_type"]


class TestMLModelInput:
    data = {
        "name": "ForestClassifier",
        "description": "A model to classify forests"
    }

    def test_dict_construction_bands_variation(self):
        input_structure = MLModelInputStructure(**TestMLModelInputStructure.data)

        # pass list of strings as bands parameter
        bands = ["B02", "B03", "B04"]
        data = {**self.data, "input": input_structure, "bands": bands}
        obj = MLModelInput(**data)
        assert obj.name == data["name"]
        assert obj.description == data["description"]
        assert obj.bands[0].name == bands[0]
        assert obj.input.shape == data["input"].shape

        # pass list of objects as bands parameter
        bands = [MLModelBand("B02"), MLModelBand("B03"), MLModelBand("B04")]
        data["bands"] = bands
        obj = MLModelInput(**data)
        assert obj.bands[0].name == bands[0].name

        # pass lsit of dicts as bands parameter
        bands = [{"name": "B02", "format": "asdf"},
                 {"name": "B03", "format": "asdf"},
                 {"name": "B04", "format": "asdf"}]
        data["bands"] = bands
        obj = MLModelInput(**data)
        assert obj.bands[0].name == bands[0]["name"]

    def test_dict_construction_input_variation(self):
        bands = ["B02", "B03", "B04"]

        # pass object as input-parameter
        input_structure = MLModelInputStructure(**TestMLModelInputStructure.data)
        data = {**self.data, "bands": bands, "input": input_structure}
        obj = MLModelInput(**data)
        assert obj.input.shape == TestMLModelInputStructure.data["shape"]
        assert obj.input.dim_order == TestMLModelInputStructure.data["dim_order"]
        assert obj.input.data_type == TestMLModelInputStructure.data["data_type"]

        # pass dict as input-parameter
        data["input"] = TestMLModelInputStructure.data
        obj = MLModelInput(**data)
        assert obj.input.shape == TestMLModelInputStructure.data["shape"]
        assert obj.input.dim_order == TestMLModelInputStructure.data["dim_order"]
        assert obj.input.data_type == TestMLModelInputStructure.data["data_type"]


class TestMLModelOutput:
    data = {
        "name": "classification_result",
        "tasks": ["classification", "segmentation"],
        "description": "This is the description"
    }

    def test_dict_construction_result_variation(self):

        # pass object as result parameter
        result_dict = TestMLModelResultStructure.data
        result_obj = MLModelResultStructure(**result_dict)
        data = {**self.data, "result": result_obj}
        obj = MLModelOutput(**data)
        assert obj.name == data["name"]
        assert obj.tasks == data["tasks"]
        assert obj.description == data["description"]
        assert obj.result.shape == result_dict["shape"]

        # pass dict as result parameter
        data["result"] = result_dict
        obj = MLModelOutput(**data)
        assert obj.result.shape == result_dict["shape"]
        assert obj.result.dim_order == result_dict["dim_order"]
