import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import utils.graphs as generator

from utils.util import load_data


graphs, _ = load_data(
    dataset="data/test-random-500-100-200-v0.125-v1-0.025.txt",
    degree_as_node_label=False)

n_graphs = len(graphs)
n_nodes = 0.
n_edges = 0.
n_ones = 0.
for graph_container in graphs:
    graph = graph_container.graph
    n_nodes += graph.number_of_nodes()
    n_edges += graph.number_of_edges()

    n_ones += sum(graph_container.node_labels)

print(f"Average number of nodes: {n_nodes/float(n_graphs)}")
print(f"Average number of edges: {n_edges/float(n_graphs)}")
print(f"Average number of 1s - graphs: {n_ones/float(n_graphs)}")
print(f"Average number of 1s - nodes: {n_ones/float(n_nodes)}")
