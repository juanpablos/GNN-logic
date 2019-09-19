import random
from functools import partial
from itertools import cycle
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
from utils.generator import *
from utils.tagger import *
from utils.coloring import *

import networkx as nx
import numpy as np


def write_graphs(number_graphs: int,
                 graphs: Generator[nx.Graph, None, None],
                 tagger: Tagger,
                 filename: str = "file.txt",
                 write_features: List[str] = None) -> None:

    total_nodes = 0
    total_1s = 0
    total_graph_1s = 0

    with open(filename, 'w') as f:
        # write number of graphs
        f.write(f"{number_graphs}\n")

        for i, graph in enumerate(graphs, start=1):
            print(f"{i}/{number_graphs} graphs writen")

            graph, num_nodes, num_ones, graph_label = tagger(graph=graph)

            graph = nx.convert_node_labels_to_integers(graph)

            total_nodes += num_nodes
            total_1s += num_ones
            total_graph_1s += graph_label

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

    print(f"{total_1s}/{total_nodes} nodes were tagged 1 ({float(total_1s)/total_nodes})")
    print(f"{total_graph_1s}/{number_graphs} graphs were tagged 1 ({float(total_graph_1s)/number_graphs})")


def generate_dataset(filename,
                     number_graphs,
                     generator_fn,
                     n_nodes,
                     structure_fn,
                     formula,
                     seed=None,
                     number_colors=10,
                     graph_split=None,
                     color_distributions=None,
                     **kwargs):
    """
    generator_fn = empty|degree|line|random|cycle
    structure_fn = line|cycle|normal|centroid
    formula = formula{1|2|3}

    greens: Tuple[int, int]
        min and max greens

    graph_split -> [0.1, 0.3, 0.6]
    color_distributions -> {0:None, 1:[...], 2:[...]}

    kwargs:
        graph_generator:
            create_centroids: bool, default False
            centroid_only_green: bool, default True

            graph_degrees:
                degrees: List[int], default None
                variable_degree: bool, default False

            random_graph:
                p: float
                    prob of an edge in erdos
                m: int, default None
                    number of edges in "barabasi"
                    number of edges in "erdos" will be m*n_nodes
                name = "erdos"|"barabasi", default "erdos"

            cycle_graph:
                pair: bool, default True

        color_generator:
            special_line: bool
            force_color: Dict[int, Dict[int, int]]
                mapping: split -> color -> number
            force_color_position: Dict[int, Dict[int, int]]
                not yet implemented

        tag_generator:
            red_exist_green:
                n_green: int, default 1
                    number of greens to search

            color_no_connected_color:
                local_prop, default [1]
                    possible local properties
                global_prop, default [0]
                    search for global properties
                global_constraint, default [0: 1]
                    number of global properties to search
                condition, default "and"
                    if all global properties must be satisfied or only one
    """

    random.seed(seed)
    np.random.seed(seed)
    kwargs["seed"] = seed

    min_nodes, max_nodes = n_nodes

    tagger = Tagger(formula, **kwargs)
    generator = graph_generator(generator_fn=generator_fn,
                                min_nodes=min_nodes,
                                max_nodes=max_nodes,
                                **kwargs)
    color_graphs = color_generator(graph_generator=generator,
                                   number_graphs=number_graphs,
                                   min_nodes=min_nodes,
                                   max_nodes=max_nodes,
                                   graph_split=graph_split,
                                   color_distributions=color_distributions,
                                   structure_fn=structure_fn,
                                   n_colors=number_colors,
                                   **kwargs)

    if "cycle" in filename:
        file_name = f"data/{formula}/{filename}-{number_graphs}-{min_nodes}-{max_nodes}-{kwargs['m']}.txt"
    else:
        file_name = f"data/{formula}/{filename}-{number_graphs}-{min_nodes}-{max_nodes}.txt"

    write_graphs(number_graphs=number_graphs,
                 graphs=color_graphs,
                 tagger=tagger,
                 filename=file_name,
                 write_features=["color"])


if __name__ == "__main__":
    # TODO: implement manual limit to number of nodes with each color
    """
    formula1 -> x in G, red(x) and exist_N y in G, such that green(y)
    formula2 -> x in G, R_1(x) and
        (exist_N_1 y_1 in G, such that G_1(y_1) AND|OR
         exist_N_2 y_2 in G, such that G_2(y_2) AND|OR ...)
    """

    _tagger_fn = "formula3"
    _name = "barabasi"
    _data_name = f"random-{_name}"
    _m = 1

    generate_dataset(f"train-{_data_name}",
                     number_graphs=100,
                     # empty|degree|line|random|cycle
                     generator_fn=_data_name.split("-")[0],
                     n_nodes=(50, 100),
                     # line|cycle|normal|centroid
                     structure_fn="centroid",
                     # formula{1|2|3}
                     formula=_tagger_fn,
                     seed=None,
                     number_colors=5,
                     # global, tuple
                     greens=(10, 20),
                     # random
                     name=_name,
                     m=_m,
                     # centroid
                     create_centroids=True,
                     centroids=(2, 2),
                     nodes_per_centroid=(20, 20),
                     centroid_connectivity=0.5,
                     centroid_extra=None,  # {},
                     centroid_only_green=True,
                     # tagger
                     # formula 1
                     n_green=1,
                     # formula 3
                     local_prop=[1],
                     global_prop=[0],
                     global_constraint={0: 1},
                     condition="and")

    # test_dataset(
    #     name=_data_name,
    #     tagger_fn=_tagger_fn,
    #     seed=None,
    #     n_colors=5,
    #     number_of_graphs=500,
    #     n_min=50,
    #     n_max=100,
    #     random_degrees=True,
    #     min_degree=0,
    #     max_degree=2,
    #     no_green=False,
    #     special_line=True,
    #     edges=0.025,
    #     split_line=_split_line,
    #     m=_m,
    #     force_green=3,
    #     two_color=True,
    #     # tagger
    #     # formula 1
    #     n_green=1,
    #     # formula 3
    #     local_prop=[1],
    #     global_prop=[0],
    #     global_constraint={0: 1},
    #     condition="or")

    # test_dataset(
    #     name=_data_name,
    #     tagger_fn=_tagger_fn,
    #     seed=None,
    #     n_colors=5,
    #     number_of_graphs=500,
    #     n_min=100,
    #     n_max=200,
    #     random_degrees=True,
    #     min_degree=0,
    #     max_degree=_prop,
    #     no_green=False,
    #     special_line=True,
    #     edges=0.025,
    #     split_line=_split_line,
    #     m=_m,
    #     force_green=3,
    #     two_color=True,
    #     # tagger
    #     # formula 1
    #     n_green=1,
    #     # formula 3
    #     local_prop=[1],
    #     global_prop=[0],
    #     global_constraint={0: 1},
    #     condition="or")
