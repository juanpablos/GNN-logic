import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from torch_geometric.utils import to_networkx

from util import load_data

graphs, _ = load_data(
    dataset="../data/formula3/asd-random-erdos.txt",
    degree_as_node_label=False)

n_graphs = len(graphs)
n_nodes = 0.
n_edges = 0.

n_colors = graphs[0].x.size()[1]
average_color_count = dict.fromkeys(range(n_colors), 0.)
average_color_count_1s = dict.fromkeys(range(n_colors), 0.)

avg_biggest_component = 0.
avg_connected_components = 0.

diameter = []

for graph_container in graphs:
    n_nodes += graph_container.num_nodes
    n_edges += graph_container.num_edges / 2.

    colors = torch.max(graph_container.x, dim=1)[1].squeeze().tolist()
    labels = graph_container.node_labels.tolist()

    for color, label in zip(colors, labels):
        average_color_count[color] += 1

        if label > 0:
            average_color_count_1s[color] += 1

    graph = to_networkx(graph_container).to_undirected()
    avg_biggest_component += max([len(component)
                                  for component in nx.connected_components(graph)])
    avg_connected_components += nx.number_connected_components(graph)

    diameter.append(nx.diameter(graph))

total_ones = sum(average_color_count_1s.values())

print(f"Average number of nodes: {n_nodes/float(n_graphs)}")
print(f"Average number of edges: {n_edges/float(n_graphs)}")
print(f"Average number of 1s - graphs: {total_ones/float(n_graphs)}")
print(f"Average number of 1s - nodes: {total_ones/float(n_nodes)}")

print(
    f"Average connected components: {avg_connected_components/float(n_graphs)}")
print(f"Biggest Component: {avg_biggest_component/float(n_graphs)}")

print(f"Diameter {np.mean(diameter)}")

for color in average_color_count:
    print(
        f"Color {color} - {average_color_count_1s[color]}/{average_color_count[color]} are 1s ({average_color_count_1s[color]/average_color_count[color]})\n\t{average_color_count[color]}/{n_nodes} color proportion ({average_color_count[color]/n_nodes})")
