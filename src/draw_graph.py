import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import utils.graphs as generator
from scipy.special import binom
from utils.util import load_data

# generator.test_dataset(name="random",
#                        seed=None,
#                        n_colors=5,
#                        number_of_graphs=100,
#                        n_min=50,
#                        n_max=100,
#                        random_degrees=True,
#                        min_degree=0,
#                        max_degree=2,
#                        no_green=False,
#                        special_line=True,
#                        split_line=None)

graphs, _ = load_data(
    dataset="data/test/test-random-500-50-100-2-0.01.txt",
    degree_as_node_label=False)

n = 400
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
