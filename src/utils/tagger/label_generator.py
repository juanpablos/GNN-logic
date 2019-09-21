from functools import partial
from typing import Callable, List, Tuple, Dict

import networkx as nx
import numpy as np


def __red_exist_green(graph: nx.Graph,
                      n_green=1,
                      **kwargs) -> Tuple[List[int], int]:

    node_features = [node[1] for node in graph.nodes(data="color")]

    features = np.array(node_features)
    # green
    existence_condition = np.sum(features == 0) >= n_green
    if existence_condition:
        # red
        individual_condition = (features == 1).astype(int)
        # return if each node is a red node or not
        return individual_condition, 1
    # no existance condition -> all nodes are 0
    return np.zeros(features.shape[0]).astype(int), 0


def __map_colors(graph, nodes):
    colors = {}
    for node in nodes:
        color = graph.node[node]['color']
        if color not in colors:
            colors.setdefault(color, 0)

        colors[color] += 1

    return colors


def __color_no_connected_color(graph: nx.Graph,
                               local_prop: List[int] = None,
                               global_prop: List[int] = None,
                               global_constraint: Dict[int, Tuple[int, int]] = None,
                               condition: str = "and",
                               **kwargs) -> Tuple[List[int], int]:

    if condition == "and":
        condition = all
    elif condition == "or":
        condition = any
    else:
        raise ValueError()

    if local_prop is None:
        local_prop = []

    if global_prop is None:
        # searching for green
        global_prop = []

    if global_constraint is None:
        global_constraint = {}

    for color in global_prop:
        if color not in global_constraint:
            global_constraint[color] = (1, 1)

    # count graph colors
    graph_colors = __map_colors(graph, graph)
    for cons in global_constraint:
        if cons not in graph_colors:
            graph_colors[cons] = 0

    labels = []
    for node in graph:

        # if the node is of the local property color, or not searching for that
        if graph.node[node]['color'] in local_prop or len(local_prop) == 0:

            # color count of my neighbors
            neighbor_color_map = __map_colors(graph, graph.neighbors(node))

            # colors left in graph that are not my neighbors
            left_in_graph = {}
            for color, count in graph_colors.items():
                neighbor_color_count = neighbor_color_map.get(color, 0)

                left_in_graph[color] = count - neighbor_color_count

            # check if there are N_i nodes with color i in the graph
            # that are not my neighbors
            # if the number of nodes for each color is bigger than what
            # we need, the node satisfies the property
            # depends on the condition - AND or OR
            if len(global_constraint) != 0:
                satisfies = condition(
                    [
                        max_count >= left_in_graph[color] >= min_count
                        for color, (min_count, max_count)
                        in global_constraint.items()
                    ]
                )
            else:
                # not searching for global
                satisfies = True

            if satisfies:
                labels.append(1)
            else:
                labels.append(0)

        else:
            labels.append(0)

    return np.array(labels).astype(int), int(any(l > 0 for l in labels))


def __neighbor_nested_property(graph: nx.Graph,
                               nested: List[str],
                               local_prop_nested: List[List[int]],
                               constraint_nested: List[Tuple[int, int]],
                               self_satisfy_nested: List[bool],
                               **kwargs) -> Tuple[List[int], int]:

    assert isinstance(nested, list)
    assert isinstance(local_prop_nested, list)
    assert isinstance(constraint_nested, list)
    assert isinstance(self_satisfy_nested, list)
    assert len(nested) == len(local_prop_nested) == len(
        constraint_nested) == len(self_satisfy_nested)

    # not empty
    if nested:
        fn = nested[0]
        local_prop = local_prop_nested[0]
        constraint_min, constraint_max = constraint_nested[0]
        self_satisfy = self_satisfy_nested[0]
    else:
        raise ValueError("Can't loop forever")

    if local_prop is None:
        local_prop = []

    # calculate the other property
    inner_labels, _ = tagger_dispatch(fn,
                                      nested=nested[1:],
                                      local_prop_nested=local_prop_nested[1:],
                                      constraint_nested=constraint_nested[1:],
                                      self_satisfy_nested=self_satisfy_nested[1:],
                                      **kwargs)(graph)
    num_1s = sum(inner_labels)

    graph = nx.convert_node_labels_to_integers(
        graph, label_attribute="old-name")

    labels = []
    for node in graph:
        node_satisfy = True
        if self_satisfy:
            node_satisfy = bool(inner_labels[node])

        if node_satisfy and (
            graph.node[node]['color'] in local_prop or
                len(local_prop) == 0):
            neighbors = graph.neighbors(node)
            neigh_1s = sum(inner_labels[i] for i in neighbors)

            if constraint_max >= (num_1s - neigh_1s) >= constraint_min:
                labels.append(1)
            else:
                labels.append(0)
        else:
            labels.append(0)

    return np.array(labels).astype(int), int(any(l > 0 for l in labels))


class Tagger():
    def __init__(self, formula: str, **kwargs):
        self.tagger = tagger_dispatch(formula, **kwargs)

    def __call__(self, graph: nx.Graph):
        graph, graph_label, num_nodes, num_ones = self.__tagging_logic(
            graph=graph, formula=self.tagger)
        return graph, num_nodes, num_ones, graph_label

    def __tagging_logic(self, graph, formula):

        labels, graph_label = formula(graph)
        graph.graph["label"] = graph_label

        for node_id, node_name in enumerate(graph):
            graph.node[node_name]["label"] = labels[node_id]

        return graph, graph_label, len(labels), sum(labels)


def tagger_dispatch(tagger,
                    **kwargs) -> Callable[[nx.Graph],
                                          Tuple[List[int], int]
                                          ]:
    options = {
        "formula1": partial(__red_exist_green, **kwargs),
        "formula2": partial(__red_exist_green, **kwargs),
        "formula3": partial(__color_no_connected_color, **kwargs),
        "formula4": partial(__neighbor_nested_property, **kwargs),
    }
    if tagger not in options:
        raise ValueError()
    return options[tagger]
