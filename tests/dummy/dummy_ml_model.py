from openeo_processes_dask_ml.process_implementations.data_model import MLModel


class DummyMLModel(MLModel):
    # Only for testing purposes

    def create_object(self):
        pass

    def run_model(self):
        pass

    def preprocess_datacube(self):
        pass

    def postprocess_datacube(self):
        pass