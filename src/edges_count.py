import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import utils.graphs as generator
import numpy as np

from utils.util import load_data


graphs, _ = load_data(
    dataset="data/test-line-special-500-100-200.txt",
    degree_as_node_label=False)

n_graphs = len(graphs)
n_nodes = 0.
n_edges = 0.
n_ones = 0.

n_greens = 0.

avg_biggest_component = 0.
avg_connected_components = 0.

for graph_container in graphs:
    graph = graph_container.graph
    n_nodes += graph.number_of_nodes()
    n_edges += graph.number_of_edges()

    n_ones += sum(graph_container.node_labels)

    n_greens += sum(np.array([color for color in np.argmax(
        graph_container.node_features, axis=1)]) == 0)

    avg_biggest_component += max([len(component)
                                  for component in nx.connected_components(graph)])
    avg_connected_components += nx.number_connected_components(graph)

print(f"Average number of nodes: {n_nodes/float(n_graphs)}")
print(f"Average number of edges: {n_edges/float(n_graphs)}")
print(f"Average number of 1s - graphs: {n_ones/float(n_graphs)}")
print(f"Average number of 1s - nodes: {n_ones/float(n_nodes)}")

print(f"Average number of greens: {n_greens/float(n_graphs)}")

print(
    f"Average connected components: {avg_connected_components/float(n_graphs)}")
print(f"Biggest Component: {avg_biggest_component/float(n_graphs)}")
