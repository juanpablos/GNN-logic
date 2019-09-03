import random
from typing import Callable, List, Optional, Tuple

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
    print("Start generating graphs")
    for i in range(n):
        print(f"{i}/{n} graphs generated")
        n_nodes = random.randint(min_nodes, max_nodes)
        graph = generator_fn(n=n_nodes, seed=random_state, **kwargs)
        graph_list.append(graph)

    print("Finish generating graphs")
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
                    features = " ".join([str(node_attributes[feature])
                                         for feature in write_features])
                    assert n_features == len(features)
                    f.write(
                        f"{n_features} {features} {node_attributes['label']} {n_edges} {edges}\n")


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

    if distribution is not None:
        assert len(distribution) == n_colors

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

    print("Coloring graphs")
    possible_labels = list(range(n_colors))
    n = len(graph_list)
    for i, graph in enumerate(graph_list):
        print(f"{i}/{n} graphs colored")
        n_nodes = len(graph)
        node_colors = np.random.choice(
            possible_labels, size=n_nodes, replace=True, p=distribution)

        nx.set_node_attributes(graph, dict(
            zip(graph, node_colors)), name="color")

        # placeholder
        graph.graph["label"] = 0

    __write_graphs(graph_list, filename=file_output)


def tagger(input_file: str, output_file: str, formula: Callable[[
           List[int]], Tuple[List[bool], int]]) -> None:
    """Make labels for all nodes based on their color

    Arguments:
        input_file {str} -- name of the file to tag
        output_file {str} -- name of the file to write
        formula {Callable[[List[int]], Tuple[np.array[bool], int]]} -- function that tags each node. Arguments must be `color of the nodes`. It returns a list of labels for each node and a boolean meaning if the graph has a node that satisfies the condition.
    """
    print("Start tagging graphs")
    print("-- reading")
    graph_list = __graph_file_reader(input_file, read_node_label=True)

    n = len(graph_list)
    total_nodes = 0
    total_tagged = 0

    for i, graph in enumerate(graph_list):
        print(f"{i}/{n} graphs tagged")
        node_colors = [node[1] for node in graph.nodes(data="color")]

        labels, graph_label = formula(node_colors)
        graph.graph["label"] = graph_label

        total_nodes += len(labels)
        total_tagged += sum(labels)

        for node_id in graph:
            graph.node[node_id]["label"] = labels[node_id]
    print("-- finished tagging")
    print("-- writting")

    print(f"{total_tagged}/{total_nodes} nodes were tagged 1 ({total_tagged/total_nodes})")

    __write_graphs(graph_list, filename=output_file, write_features=["color"])


def tagger_fn(node_features: List[int]) -> Tuple[List[bool], int]:
    features = np.array(node_features)
    # green
    existence_condition = np.any(features == 0)
    if existence_condition:
        # red
        individual_condition = (features == 1).astype(int)
        # return if each node is a red node or not
        return individual_condition, 1
    # no existance condition -> all nodes are 0
    return np.zeros(features.shape[0]).astype(int), 0


if __name__ == "__main__":
    seed = 10
    random.seed(seed)
    np.random.seed(seed)

    n_graphs = 100
    n_nodes = 1000

    number_of_colors = 10
    red_prob = 0.5
    green_prob = 0.01

    rest = (1. - green_prob - red_prob) / (number_of_colors - 2)

    prob = [green_prob, red_prob] + [rest] * (number_of_colors - 2)

    g_l = generator(
        distribution=prob,
        n=n_graphs,
        min_nodes=int(n_nodes / 10),
        max_nodes=n_nodes,
        structure_fn=nx.fast_gnp_random_graph,
        n_colors=number_of_colors,
        random_state=seed,
        p=0.1,  # random.random(),
        # file_input="MUTAG.txt",
        file_output="test.txt")
    tagger(
        input_file="test.txt",
        output_file="test_labeled.txt",
        formula=tagger_fn)
