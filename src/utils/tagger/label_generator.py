import random
from functools import partial
from itertools import cycle
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

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
                               local_prop=None,
                               global_prop=None,
                               global_constraint=None,
                               condition="and",
                               **kwargs) -> Tuple[List[int], int]:

    if condition == "and":
        condition = all
    elif condition == "or":
        condition = any
    else:
        raise ValueError()

    if local_prop is None:
        # i am red
        local_prop = [1]

    if global_prop is None:
        # searching for green
        global_prop = [0]

    if global_constraint is None:
        global_constraint = {}

    for color in global_prop:
        if color not in global_constraint:
            global_constraint[color] = 1

    # count graph colors
    graph_colors = __map_colors(graph, graph)
    for cons in global_constraint:
        if cons not in graph_colors:
            graph_colors[cons] = 0

    labels = []
    for node in graph:

        # if the node is of the local property color
        if graph.node[node]['color'] in local_prop:

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
            satisfies = condition(
                [left_in_graph[color] >= count_const for color, count_const in global_constraint.items()])

            if satisfies:
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
    }
    if tagger not in options:
        raise ValueError()
    return options[tagger]
