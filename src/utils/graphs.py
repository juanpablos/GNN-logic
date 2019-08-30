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


def __write_graphs(graphs: List[nx.Graph],
                   filename: str = "file.txt",
                   write_features: Optional[List[str]] = None) -> None:
    with open(filename, 'w') as f:
        # write number of graphs
        f.write(f"{len(graphs)}\n")

        for graph in graphs:
            n_nodes = graph.number_of_nodes()
            label = graph.graph["label"]
            # write N_nodes in graph_i and label_i
            f.write(f"{n_nodes} {label}\n")

            # write nodes
            for node in graph.nodes(data=True):
                node_index, node_attributes = node
                edges = " ".join(map(str, list(graph[node_index].keys())))
                n_edges = len(graph[node_index])

                # writing type 1 graph
                if write_features is None:
                    f.write(
                        f"{node_attributes['color']} {n_edges} {edges}\n")

                # writing type 2 graph
                else:
                    n_features = len(write_features)
                    features = " ".join([node_attributes[feature]
                                         for feature in write_features])
                    assert n_features == len(features)
                    f.write(
                        f"{n_features} {features} {node_attributes['label']} {n_edges}, {edges}")


def __graph_file_reader(filename: str,
                        read_node_label: bool = False) -> List[nx.Graph]:
    graph_list: List[nx.Graph] = []
    with open(filename, 'r') as f:
        # ! only accept format 1, described in readme.
        # first line -> number of graphs
        n_graphs = int(f.readline().strip())
        for _ in range(n_graphs):
            # number of nodes , graph label
            n_nodes, graph_label = map(int, f.readline().strip().split(" "))
            # creates the graph and with its label
            graph = nx.Graph(label=graph_label)
            # adds all nodes
            graph.add_nodes_from(range(n_nodes))
            for node_id in range(n_nodes):
                # node label , number of edges, neighbours
                node_row = list(map(int, f.readline().strip().split(" ")))
                # we may ignore the node label as we are adding our own later
                if read_node_label:
                    node_label = node_row[0]
                    # * we assume the label of the node represents its color
                    graph.node[node_id]["color"] = node_label
                n_edges = node_row[1]
                if n_edges > 0:
                    edges = [(node_id, other_node)
                             for other_node in node_row[2:]]
                    graph.add_edges_from(edges)

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
              **kwargs) -> None:
    if distribution is None:
        distribution = [1. / n_colors] * n_colors

    if structure_fn is not None:
        graph_list = __generate_graph(
            n,
            structure_fn,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            random_state=random_state,
            **kwargs)
    elif file_input is not None:
        graph_list = __graph_file_reader(
            filename=file_input, read_node_label=False)
    else:
        raise ValueError(
            "Must indicate a graph generator function or a filename with the graph structure")

    possible_labels = list(range(n_colors))
    for graph in graph_list:
        n_nodes = len(graph)
        node_colors = np.random.choice(
            possible_labels, size=n_nodes, replace=True, p=distribution)

        nx.set_node_attributes(graph, dict(
            zip(graph, node_colors)), name="color")

        # TODO: graph label
        graph.graph["label"] = 0

    __write_graphs(graph_list, filename=file_output)


def tagger(input_file: str, output_file: str,
           formula: Callable[[int, List[int]], bool]) -> None:
    """Make labels for all nodes based on their color

    Arguments:
        input_file {str} -- name of the file to tag
        output_file {str} -- name of the file to write
        formula {Callable[[int, List[int]], bool]} -- function that tags each node. Arguments must be `color of the node` and `list of all node colors`. It returns a boolean meaning if the node satisfies the condition or not.
    """
    graph_list = __graph_file_reader(input_file, read_node_label=True)

    for graph in graph_list:
        node_colors = [node[1] for node in graph.nodes(data="color")]

        for node_id in graph:
            label = formula(node_colors[node_id], node_colors)
            graph.node[node_id]["label"] = label

    __write_graphs(graph_list, filename=output_file, write_features=["color"])


def tagget_fn(node_feature: int, node_collection_features: List[int]) -> bool:
    return node_feature == 0 and any(
        feature == 1 for feature in node_collection_features)


if __name__ == "__main__":
    seed = 0
    random.seed(seed)
    np.random.seed(seed)

    g_l = generator(
        distribution=None,
        n=150,
        min_nodes=3,
        max_nodes=10,
        # structure_fn=nx.erdos_renyi_graph,
        n_colors=10,
        random_state=seed,
        p=0.1,  # random.random(),
        file_output="testing2.txt",
        file_input="MUTAG.txt")
