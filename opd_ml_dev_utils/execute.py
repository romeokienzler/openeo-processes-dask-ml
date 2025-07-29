"""
Execute a process graph with ML processes using the provided minibackend
"""

import os

from minibackend import execute_graph

if os.getcwd().split("/")[-1] == "opd_ml_dev_utils":
    os.chdir("..")

GRAPH_PATH = "opd_ml_dev_utils/graph.json"
result = execute_graph(GRAPH_PATH)
print(result)
