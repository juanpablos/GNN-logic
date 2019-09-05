import random
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import networkx as nx
import numpy as np


def __generate_empty_graph(n_nodes: int, ** kwargs) -> nx.Graph:
    return nx.empty_graph(n=n_nodes)


def __generate_graph_by_degree(
        seed: int,
        degrees: List[int] = None,
        *,
        use_random: bool = False,
        n_nodes: int = None,
        min_degree: int = 0,
        max_degree: int = 0,
        **kwargs) -> nx.Graph:

    degree_sequence = degrees

    if use_random:
        if n_nodes is not None:
            graph = None

            proposed_n_nodes = n_nodes
            while True:
                try:
                    _degrees = range(min_degree, max_degree + 1, 1)
                    degree_sequence = random.choices(
                        _degrees, k=proposed_n_nodes)

                    graph = nx.random_degree_sequence_graph(
                        sequence=degree_sequence, seed=seed)
                except nx.NetworkXUnfeasible:
                    proposed_n_nodes += 1
                    continue

                print(f"Took {proposed_n_nodes-n_nodes} tries")
                return graph
        else:
            raise ValueError()

    else:
        if degree_sequence is None or len(degree_sequence) == 0:
            raise ValueError()

        return nx.random_degree_sequence_graph(
            sequence=degree_sequence, seed=seed)


def __generate_line_graph(n_nodes: int, **kwargs) -> nx.Graph:
    return nx.path_graph(n=n_nodes)


def __generate_random_graph(
        n_nodes: int,
        p: float,
        seed: int,
        **kwargs) -> nx.Graph:
    return nx.fast_gnp_random_graph(n=n_nodes, p=p, seed=seed)


def __generate_graph(n_graphs: int,
                     generator_fn: str,
                     min_nodes: int,
                     max_nodes: int,
                     random_state: int = 0,
                     variable_degree: bool = False,
                     **kwargs) -> Generator[nx.Graph, None, None]:

    fn = None
    if generator_fn == "empty":
        fn = __generate_empty_graph

    elif generator_fn == "degree":
        fn = __generate_graph_by_degree

    elif generator_fn == "line":
        fn = __generate_line_graph

    elif generator_fn == "random":
        fn = __generate_random_graph

    else:
        raise ValueError()

    print("Start generating graphs")

    for i in range(n_graphs):
        print(f"{i}/{n_graphs} graphs generated")

        n_nodes = random.randint(min_nodes, max_nodes)
        yield fn(n_nodes=n_nodes, seed=random_state, use_random=variable_degree, **kwargs)

    print("Finish generating graphs")


def __graph_file_reader(filename: str,
                        read_node_label: bool) -> Generator[Union[int,
                                                                  nx.Graph],
                                                            None,
                                                            None]:
    with open(filename, 'r') as f:
        # ! only accept format 1, described in readme.
        # first line -> number of graphs
        n_graphs = int(f.readline().strip())

        yield n_graphs

        for i in range(n_graphs):
            print(f"{i}/{n_graphs} graphs read")
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

            yield graph


def write_graphs(graphs: Generator[Union[int, nx.Graph], None, None],
                 filename: str = "file.txt",
                 write_features: Optional[List[str]] = None) -> None:
    with open(filename, 'w') as f:
        # write number of graphs
        f.write(f"{next(graphs)}\n")

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


def generator(graph_distribution: List[float],
              node_distribution_1: List[float],
              node_distribution_2: List[float],
              number_graphs: int,
              min_nodes: int,
              max_nodes: int,
              structure_fn: str = None,
              variable_degree: bool = False,
              file_input: Optional[str] = None,
              n_colors: int = 10,
              force_color: Dict[int, int] = None,
              force_color_position: Dict[int, int] = None,
              random_state: int = 0,
              **kwargs) -> Generator[Union[int, nx.Graph], None, None]:

    if structure_fn is not None:
        graph_generator = __generate_graph(
            n_graphs=number_graphs,
            generator_fn=structure_fn,
            min_nodes=min_nodes,
            max_nodes=max_nodes,
            random_state=random_state,
            variable_degree=variable_degree,
            **kwargs)
        n_graphs = number_graphs

    elif file_input is not None:
        graph_generator = __graph_file_reader(
            filename=file_input, read_node_label=False)
        n_graphs = next(graph_generator)
    else:
        raise ValueError(
            "Must indicate a graph generator function or a filename with the graph structure")

    print("Coloring graphs")
    possible_colors = list(range(n_colors))

    # no green, green is 0
    # al least N greens in partition 2, in defined in `force_color`
    partition_1 = graph_distribution[0] * n_graphs

    yield number_graphs

    for i, graph in enumerate(graph_generator):
        print(f"{i}/{n_graphs} graphs colored")
        n_nodes = len(graph)

        if i < partition_1:
            # no green
            colors = possible_colors[1:]

            node_colors = np.random.choice(
                colors,
                size=n_nodes,
                replace=True,
                p=node_distribution_1)

        else:
            forced_colors = []
            for (color, times) in force_color.items():
                for c in ([color] * times):
                    forced_colors.append(c)

            node_colors = np.random.choice(
                possible_colors,
                size=n_nodes - len(forced_colors),
                replace=True,
                p=node_distribution_2).tolist() + forced_colors

            np.random.shuffle(node_colors)

            if force_color_position is not None:
                # TODO: only work for 2 colors
                (c_1, p_1), (c_2, p_2) = force_color_position.items()

                # search for the color index
                c1_pos = np.where(node_colors == c_1)
                c2_pos = np.where(node_colors == c_2)

                # swap the values
                node_colors[[p_1, c1_pos]] = node_colors[[c1_pos, p_1]]
                node_colors[[p_2, c2_pos]] = node_colors[[c2_pos, p_2]]

        nx.set_node_attributes(graph, dict(
            zip(graph, node_colors)), name="color")

        # placeholder
        graph.graph["label"] = 0

        yield graph


def tagger(input_file: str,
           formula: Callable[[List[int]],
                             Tuple[List[bool],
                                   int]]) -> Generator[Union[int,
                                                             nx.Graph],
                                                       None,
                                                       None]:
    """Make labels for all nodes based on their color

    Arguments:
        input_file {str} -- name of the file to tag
        formula {Callable[[List[int]], Tuple[np.array[bool], int]]} -- function that tags each node. Arguments must be `color of the nodes`. It returns a list of labels for each node and a boolean meaning if the graph has a node that satisfies the condition.
    """
    print("Start tagging graphs")
    print("-- reading")

    reader = __graph_file_reader(input_file, read_node_label=True)

    n_graphs = next(reader)
    yield n_graphs

    total_nodes = 0
    total_tagged = 0
    total_property = 0

    for i, graph in enumerate(reader):
        print(f"{i}/{n_graphs} graphs tagged")
        node_colors = [node[1] for node in graph.nodes(data="color")]

        labels, graph_label = formula(node_colors)

        graph.graph["label"] = graph_label
        total_property += graph_label

        total_nodes += len(labels)
        total_tagged += sum(labels)

        for node_id in graph:
            graph.node[node_id]["label"] = labels[node_id]

        yield graph
    print("-- finished tagging")
    print("-- writting")

    print(f"{total_tagged}/{total_nodes} nodes were tagged 1 ({total_tagged/total_nodes})")
    print(f"{total_property}/{n_graphs} graphs were tagged 1 ({total_property/n_graphs})")


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


def train_dataset(
        name,
        seed,
        n_colors,
        number_of_graphs,
        n_min,
        n_max,
        random_degrees,
        no_green=False,
        **kwargs):
    random.seed(seed)
    np.random.seed(seed)

    # 1/2 of the graphs do not have green
    # the other 1/2 have at least force_color[0] greens
    graph_distribution = [0.5, 0.5]

    # on the second graph split, force 1 green (0) in each graph
    force_color = {0: 1}
    force_pos = {}

    # 1/2 red (1), 0.5/4 the others
    red_prob = 0.5

    green_prob = 0
    others = (1. - red_prob - green_prob) / (n_colors - 2)
    node_distribution_1 = [red_prob] + [others] * (n_colors - 2)

    if not no_green:
        green_prob = (1. - red_prob) / (n_colors - 1)
    others = (1. - red_prob - green_prob) / (n_colors - 2)
    node_distribution_2 = [green_prob,
                           red_prob] + [others] * (n_colors - 2)

    graph_generator = generator(
        graph_distribution=graph_distribution,
        node_distribution_1=node_distribution_1,
        node_distribution_2=node_distribution_2,
        number_graphs=number_of_graphs,
        min_nodes=n_min,
        max_nodes=n_max,
        structure_fn=name.split("-")[0],
        variable_degree=random_degrees,
        n_colors=n_colors,
        # file_input="MUTAG.txt",
        random_state=seed,
        force_color=force_color,
        force_color_position=None,
        p=0.3,
        **kwargs)  # random.random()

    i = random.randint(0, 1000)
    write_graphs(graph_generator, filename=f"temp{i}.txt")

    label_generator = tagger(input_file=f"temp{i}.txt", formula=tagger_fn)
    write_graphs(
        label_generator,
        filename=f"../data/train-{name}-{number_of_graphs}-{n_min}-{n_max}-v{green_prob}-v{force_color[0]}.txt",
        write_features=["color"])


def test_dataset(
        name,
        seed,
        n_colors,
        number_of_graphs,
        n_min,
        n_max,
        random_degrees,
        no_green=False,
        **kwargs):
    random.seed(seed)
    np.random.seed(seed)

    # 1/2 of the graphs do not have green
    # the other 1/2 have at least force_color[0] greens
    graph_distribution = [0.5, 0.5]

    # on the second graph split, force 1 green (0) in each graph
    force_color = {0: 1}
    force_pos = {}

    # 1/2 red (1), 0.5/4 the others
    red_prob = 0.5

    green_prob = 0
    others = (1. - red_prob - green_prob) / (n_colors - 2)
    node_distribution_1 = [red_prob] + [others] * (n_colors - 2)

    if not no_green:
        green_prob = (1. - red_prob) / (n_colors - 1)
    others = (1. - red_prob - green_prob) / (n_colors - 2)
    node_distribution_2 = [green_prob,
                           red_prob] + [others] * (n_colors - 2)

    graph_generator = generator(
        graph_distribution=graph_distribution,
        node_distribution_1=node_distribution_1,
        node_distribution_2=node_distribution_2,
        number_graphs=number_of_graphs,
        min_nodes=n_min,
        max_nodes=n_max,
        structure_fn=name.split("-")[0],
        variable_degree=random_degrees,
        n_colors=n_colors,
        # file_input="MUTAG.txt",
        random_state=seed,
        force_color=force_color,
        force_color_position=None,
        p=0.3,
        **kwargs)  # random.random()

    i = random.randint(0, 1000)
    write_graphs(graph_generator, filename=f"temp{i}.txt")

    label_generator = tagger(input_file=f"temp{i}.txt", formula=tagger_fn)
    write_graphs(
        label_generator,
        filename=f"../data/test-{name}-{number_of_graphs}-{n_min}-{n_max}-v{green_prob}-v{force_color[0]}.txt",
        write_features=["color"])


if __name__ == "__main__":

    # train_dataset(
    #     name="degree-0y2",
    #     seed=33,
    #     n_colors=5,
    #     number_of_graphs=5000,
    #     n_min=10,
    #     n_max=50,
    #     random_degrees=True,
    #     min_degree=0,
    #     max_degree=2,
    #     no_green=True)

    test_dataset(
        name="degree-0y2",
        seed=888,
        n_colors=5,
        number_of_graphs=100,
        n_min=100,
        n_max=120,
        random_degrees=True,
        min_degree=0,
        max_degree=2,
        no_green=False)
