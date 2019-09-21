import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.special import binom
from torch_geometric.utils import to_networkx

from util import load_data

graphs, _ = load_data(
    dataset="../data/formula3/asd-random-erdos.txt",
    degree_as_node_label=False)

n = 0
graph = to_networkx(graphs[n]).to_undirected()
node_labels = graphs[n].node_labels.numpy()


colors = graphs[n].x.numpy()
print(graphs[n].graph_label.item())

color_map = {0: "green", 1: "red", 2: "blue", 3: "yellow", 4: "purple"}
node_colors = [color for color in np.argmax(colors, axis=1)]
node_colors_name = [color_map[color] for color in node_colors]

nx.set_node_attributes(graph, dict(zip(graph, node_colors)), name="color")

labels = dict(zip(graph, node_labels))

fig, ax = plt.subplots()
nx.draw(
    graph,
    with_labels=True,
    ax=ax,
    node_color=node_colors_name,
    labels=labels)
ax.set_title(
    f"{graph.number_of_nodes()} nodes - {graph.number_of_edges()} edges")
plt.tight_layout()
plt.show()
