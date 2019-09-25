import random
from functools import partial
from itertools import cycle
from typing import Dict, Generator, List, Tuple

import networkx as nx
import numpy as np

from utils.coloring import *
from utils.generator import *
from utils.tagger import *


def write_graphs(number_graphs: int,
                 graphs: Generator[nx.Graph, None, None],
                 tagger: Tagger,
                 filename: str = "file.txt",
                 write_features: List[str] = None) -> None:

    total_nodes = 0
    total_1s = 0
    total_graph_1s = 0
    all_1s = 0
    all_0s = 0
    avg_1s_not_all_1s = []
    not_all_1s_size = []

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

            _all_1s = num_ones == len(graph)
            _all_0s = num_ones == 0

            all_1s += int(_all_1s)
            all_0s += int(_all_0s)
            if not _all_1s and not _all_0s:
                avg_1s_not_all_1s.append(num_ones)

            if not _all_1s and not _all_0s:
                not_all_1s_size.append(num_nodes)

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

    avg_1s_not_all_1s = np.array(avg_1s_not_all_1s, dtype=np.float)
    not_all_1s_size = np.array(not_all_1s_size, dtype=np.float)

    print(f"{total_1s}/{total_nodes} nodes were tagged 1 ({float(total_1s)/total_nodes})")
    print(f"{total_graph_1s}/{number_graphs} graphs were tagged 1 ({float(total_graph_1s)/number_graphs})")
    print(f"{all_1s}/{number_graphs} graphs with all 1 ({float(all_1s)/number_graphs})")
    print(f"{all_0s}/{number_graphs} graphs with all 0 ({float(all_0s)/number_graphs})")
    # print(f"{all_0s+all_1s}/{number_graphs} graphs with all 0 or all 1 ({float(all_0s+all_1s)/number_graphs})")
    print(f"{number_graphs-all_0s-all_1s}/{number_graphs} graphs with 0 and 1 ({float(number_graphs-all_0s-all_1s)/number_graphs})")

    if number_graphs - all_0s - all_1s > 0:
        # average number of ones per graph in graph with not all 1s
        # average size of graphs with not all 1s

        temp = avg_1s_not_all_1s / not_all_1s_size

        print(f"{np.mean(avg_1s_not_all_1s)}/{np.mean(not_all_1s_size)} avg 1s in not all 1s ({np.mean(temp)}, nodes +- {np.std(avg_1s_not_all_1s)})")


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

            grid_graph:
                grid_n: int
                grid_m: int
                periodic: bool
                diagonal: bool

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
                local_prop: List[int], default []
                    possible local properties
                global_prop: List[int], default []
                    search for global properties
                global_constraint: Dict[int, int], default {}
                    number of global properties to search
                condition: str, "and"|"or", default "and"
                    if all global properties must be satisfied or only one

            nested_property:
                nested: str, formula{1|2|3}
                    another property
                local_prop_nested: List[int], default []
                nested_constraint: int, default 1
                    how many properties must be satisfied that are not from neighbors
                self_satisfy: bool, default True
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
    elif "asd" in filename:
        file_name = f"data/{formula}/{filename}.txt"
    else:
        file_name = f"data/{formula}/{filename}-{number_graphs}-{min_nodes}-{max_nodes}.txt"

    write_graphs(number_graphs=number_graphs,
                 graphs=color_graphs,
                 tagger=tagger,
                 filename=file_name,
                 write_features=["color"])


if __name__ == "__main__":
    """
    formula1 -> x in G, red(x) and exist_N y in G, such that green(y)
    formula3 -> x in G, R_1(x) and
        (exist_N_1 y_1 in G, such that G_1(y_1) AND|OR
         exist_N_2 y_2 in G, such that G_2(y_2) AND|OR ...)
    formula4 -> x in G, R_1(x) and Exists N nodes that are not in Neigh(x) that satisfiy property Y
    Formula4 represents the nested property.
    """

    _tagger_fn = "formula4"
    _name = "erdos"
    _data_name = f"random-{_name}"
    _m = 8

    generate_dataset(f"test2-{_data_name}",
                     number_graphs=250,
                     # empty|degree|line|random|cycle
                     generator_fn=_data_name.split("-")[0],
                     n_nodes=(51, 60),
                     # line|cycle|normal|centroid
                     structure_fn="normal",
                     # formula{1|2|3}
                     formula=_tagger_fn,
                     seed=None,
                     number_colors=5,
                     # global, tuple
                     greens=(9, 15),
                     # random
                     name=_name,
                     m=_m,
                     # grid
                     grid_n=7,
                     grid_m=7,
                     periodic=False,
                     diagonal=True,
                     # centroid
                     create_centroids=False,
                     centroids=(2, 2),
                     nodes_per_centroid=(20, 20),
                     centroid_connectivity=0.5,
                     centroid_extra=None,  # {},
                     centroid_only_green=True,
                     # tagger
                     # formula 1
                     n_green=1,
                     # formula 3
                     local_prop=[],
                     global_prop=[0],
                     global_constraint={0: (8, 10)},
                     condition="and",
                     # formula 4
                     # for each element, it is a nested call
                     nested=[
                         "formula4",
                         "formula3"
                     ],
                     local_prop_nested=[
                         [],
                         []
                     ],
                     constraint_nested=[
                         (10, 30),
                         (10, 20)
                     ],
                     self_satisfy_nested=[
                         False,
                         False
                     ])
