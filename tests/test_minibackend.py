from openeo_pg_parser_networkx.process_registry import DEFAULT_NAMESPACE

import minibackend


def test_import():
    registry = minibackend.process_registry
    registered_processes = registry.store[DEFAULT_NAMESPACE].keys()

    # custom process implementations
    assert "load_collection" in registered_processes
    assert "save_result" in registered_processes

    # exemplary process from openeo-processes-dask
    assert "load_stac" in registered_processes

    # openeo-processes-dask-ml processes
    assert "load_ml_model" in registered_processes
    assert "ml_predict" in registered_processes
