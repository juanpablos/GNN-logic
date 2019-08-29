import random
from typing import Callable, List, Optional

import networkx as nx
import numpy as np


def __generate_graph(n: int,
                     generator_fn: Callable[...,
                                            nx.Graph],
                     min_nodes: int,
                     max_nodes: int,
                     random_state: int = 0,
                     **kwargs) -> List[nx.Graph]:
    graph_list = []
    for _ in range(n):
        n_nodes = random.randint(min_nodes, max_nodes)
        graph = generator_fn(n=n_nodes, seed=random_state, **kwargs)
        graph_list.append(graph)

    return graph_list


def generator(distribution: Optional[List[float]],
              n: int,
              min_nodes: int,
              max_nodes: int,
              file_output: str,
              structure_fn: Optional[Callable[...,
                                              nx.Graph]] = None,
              file_input: Optional[str] = None,
              n_colors: int = 10,
              random_state: int = 0,
              **kwargs):
    if distribution is None:
        distribution = [1 / n_colors] * n_colors

    if structure_fn is not None:
        graph_list = __generate_graph(
            n,
            structure_fn,
            in_nodes=min_nodes,
            max_nodes=max_nodes,
            random_state=random_state,
            **kwargs)
    elif file_input is not None:
        # TODO: read from file
        graph_list: List[nx.Graph] = []
        pass
    else:
        raise ValueError()

    possible_labels = list(range(n_colors))
    for graph in graph_list:
        n_nodes = len(graph)
        node_colors = np.random.choice(
            possible_labels, size=n_nodes, replace=True, p=distribution)

        nx.set_node_attributes(graph, dict(
            zip(graph, node_colors)), name="color")

        # TODO: graph label
        graph.graph["label"] = 0

    assert len(graph_list) == n
    __write_graphs(graph_list, filename=file_output)

    return graph_list


def __write_graphs(graphs: List[nx.Graph], filename: str = "file.txt") -> None:
    with open(filename, 'w') as f:
        # write number of graphs
        f.write(f"{len(graphs)}\n")

        for graph in graphs:
            n_nodes = graph.number_of_nodes()
            label = graph.graph["label"]
            # write N_nodes in graph_i and label_i
            f.write(f"{n_nodes} {label}\n")

            # write nodes
            for node in graph.nodes(data="color"):
                node_index, label = node
                edges = " ".join(map(str, list(graph[node_index].keys())))

                f.write(f"{label} {len(graph[node_index])} {edges}\n")


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    g_l = generator(
        distribution=None,
        n=5,
        min_nodes=3,
        max_nodes=10,
        structure_fn=nx.erdos_renyi_graph,
        n_colors=10,
        random_state=seed,
        p=0.5,
        file_output="testing.txt")

    for g in g_l:
        print(g.nodes(data="color"))
        print(g.edges)
        print("--" * 10)
