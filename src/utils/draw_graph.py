import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from scipy.special import binom

from .util import load_data

graphs, _ = load_data(
    dataset="../data/test/test-random-500-50-100-2-0.01.txt",
    degree_as_node_label=False)

n = 0
graph = graphs[n].graph
colors = graphs[n].node_features.numpy()
print(graphs[n].graph_label)

color_map = {0: "green", 1: "red", 2: "blue", 3: "yellow", 4: "purple"}
node_colors = [color_map[color] for color in np.argmax(colors, axis=1)]

fig, ax = plt.subplots()
nx.draw(graph, with_labels=True, ax=ax, node_color=node_colors)
ax.set_title(
    f"{graph.number_of_nodes()} nodes - {graph.number_of_edges()} edges")
plt.tight_layout()
plt.show()
