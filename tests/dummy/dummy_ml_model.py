from openeo_processes_dask_ml.process_implementations.data_model import MLModel


class DummyMLModel(MLModel):
    # Only for testing purposes

    def create_model_object(self, filepath: str):
        pass

    def execute_model(self, batch):
        pass