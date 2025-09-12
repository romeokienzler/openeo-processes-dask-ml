from openeo_pg_parser_networkx import OpenEOProcessGraph

from .openeo_minibackend import process_registry


def execute_graph(path: str):
    parsed_graph = OpenEOProcessGraph.from_file(path)
    pg_callable = parsed_graph.to_callable(process_registry=process_registry)
    r = pg_callable()
    return r


def execute_graph_dict(graph: dict):
    parsed_graph = OpenEOProcessGraph(graph)
    pg_callable = parsed_graph.to_callable(process_registry=process_registry)
    r = pg_callable()
    return r
