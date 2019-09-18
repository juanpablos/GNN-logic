import random
from functools import partial
from itertools import cycle
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

# TODO: Urgent Refactor


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
        force_proportion=None,
        **kwargs) -> nx.Graph:

    if force_proportion is not None:
        return nx.gnm_random_graph(
            n_nodes, n_nodes * force_proportion, seed=seed)

    return nx.fast_gnp_random_graph(n=n_nodes, p=p, seed=seed)


def __generate_cycle_graph(n_nodes: int, **kwargs):
    nodes_in_graph = n_nodes
    if n_nodes % 2 != 0:
        nodes_in_graph += 1

    return nx.cycle_graph(n=nodes_in_graph)


def __generate_graph(n_graphs: int,
                     generator_fn: str,
                     min_nodes: int,
                     max_nodes: int,
                     random_state: int = 0,
                     variable_degree: bool = False,
                     **kwargs) -> Generator[nx.Graph, None, None]:

    fn = None
    _n_graphs = n_graphs
    if generator_fn == "empty":
        fn = __generate_empty_graph

    elif generator_fn == "degree":
        fn = __generate_graph_by_degree

    elif generator_fn == "line":
        fn = __generate_line_graph

    elif generator_fn == "random":
        fn = __generate_random_graph

    elif generator_fn == "cycle":
        # TODO: generate only unique cycles?
        fn = __generate_cycle_graph

    else:
        raise ValueError()

    yield _n_graphs

    print("Start generating graphs")

    for i in range(_n_graphs):
        print(f"{i}/{_n_graphs} graphs generated")

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


def __split_line(graph, i, split_line, partition, colors, distribution):
    n_nodes = len(graph)

    _to_remove = split_line["split"]
    if len(split_line["split"]) == 0:
        _to_remove = [random.randint(1, (n_nodes - 1))]
    elif isinstance(split_line["split"][0], float):
        _to_remove = []
        for _split in split_line["split"]:
            _to_remove.append(int(_split * (n_nodes - 1)))
    elif isinstance(split_line["split"][0], int):
        _to_remove = np.random.choice(
            range(1, n_nodes - 1),  # no green
            size=split_line["split"][0] - 1,
            replace=False,
            p=None).tolist()

    edges = list(graph.edges)
    for _split in _to_remove:
        graph.remove_edge(*edges[_split])

    sub_graphs = [graph.subgraph(c) for c in nx.connected_components(graph)]

    if i < partition:
        node_colors = np.random.choice(
            colors[1:],  # no green
            size=n_nodes,
            replace=True,
            p=distribution)
    else:
        # part 1: one green
        # random -> only one green

        if random.random() < 0.3:
            part_1 = np.random.choice(
                colors[:1] + colors[2:],
                size=len(sub_graphs[0]) - 1,
                replace=True,
                p=None).tolist() + [0]
        else:
            part_1 = np.random.choice(
                colors[2:],
                size=len(sub_graphs[0]) - 1,
                replace=True,
                p=None).tolist() + [0]

        # part 2: other side, all red
        part_2 = np.random.choice(
            [1],
            size=len(sub_graphs[-1]),
            replace=True,
            p=None).tolist()

        # part 3: all other random
        nodes_left = sum([len(g) for g in sub_graphs[1:-1]])
        part_3 = np.random.choice(
            colors,
            size=nodes_left,
            replace=True,
            p=None).tolist()

        node_colors = part_1 + part_3 + part_2

    return graph, node_colors


def __coloring_logic(
    *,
    graph,
    split_line,
    number,
    partition,
    possible_colors,
    node_distribution_1,
    node_distribution_2,
    structure_fn,
    force_color,
        force_color_position):

    n_nodes = len(graph)

    if split_line is None:
        if number < partition:
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

            node_colors = np.array(node_colors)

            np.random.shuffle(node_colors)

            if force_color_position is not None:
                # TODO: only work for 2 colors
                (c_1, p_1), (c_2, p_2) = force_color_position.items()
                # search for the color index
                for times in range(force_color[c_1]):
                    arr = node_colors[times:]

                    c1_pos = np.where(arr == c_1)[0][0]
                    arr[c1_pos], arr[p_1] = arr[p_1], arr[c1_pos]

                for times in range(force_color[c_2]):
                    arr = node_colors
                    if times != 0:
                        arr = arr[:-times]

                    c2_pos = np.where(arr == c_2)[0][0]
                    arr[c2_pos], arr[p_2] = arr[p_2], arr[c2_pos]

    elif split_line is not None and structure_fn == "line":
        graph, node_colors = __split_line(
            graph, number, split_line, partition, possible_colors, node_distribution_1)

    else:
        raise ValueError()

    nx.set_node_attributes(graph, dict(
        zip(graph, node_colors)), name="color")

    # placeholder
    graph.graph["label"] = 0

    return graph


def __special_line(
        graph,
        colors,
        number,
        n_graphs,
        only_extreme=False,
        **kwargs):

    class_1 = int(n_graphs * 0.3)
    class_2 = int(n_graphs * 0.5) + class_1
    class_3 = int(n_graphs * 0.1) + class_2
    class_4 = int(n_graphs * 0.1) + class_3

    n_nodes = len(graph)
    first_half = int(n_nodes * 0.5)

    if number < class_1:
        # class 1 (30%), no green
        # first half no green, other uniform. Second half 80% red no greens

        first_half_colors = np.random.choice(
            colors[1:],
            size=first_half,
            replace=True,
            p=None).tolist()

        red = 0.8
        others = (1. - red) / (len(colors) - 2)
        _colors = [red] + [others] * (len(colors) - 2)
        second_half_colors = np.random.choice(
            colors[1:],
            size=n_nodes - first_half,
            replace=True,
            p=_colors).tolist()

        use_colors = first_half_colors + second_half_colors

    elif number < class_2:
        # class 2 (50%)
        # first half green on first positions, other uniform. Second half
        # 80% red no greens

        # 20% of first 50% have 80% green
        greens = int(first_half * 0.2)
        green = 0.8
        others = (1. - green) / (len(colors) - 2)
        green_colors = [0] + np.random.choice(
            colors,
            size=greens - 1,
            replace=True,
            p=[green, 0.] + [others] * (len(colors) - 2)).tolist()

        others = 1. / (len(colors) - 2)
        # the other 30% is uniform with no greens and not reds
        first_half_colors = np.random.choice(
            colors,
            size=first_half - greens,
            replace=True,
            p=[0., 0.] + [others] * (len(colors) - 2)).tolist()

        red = 0.9
        others = (1. - red) / (len(colors) - 2)
        _colors = [red] + [others] * (len(colors) - 2)
        second_half_colors = np.random.choice(
            colors[1:],
            size=n_nodes - first_half,
            replace=True,
            p=_colors).tolist()

        use_colors = green_colors + first_half_colors + second_half_colors

    elif number < class_3:
        # class 3 (10%)
        # uniform, no green

        use_colors = np.random.choice(
            colors[1:],
            size=n_nodes,
            replace=True,
            p=None).tolist()

    else:
        # class 4 (10%)
        # uniform

        use_colors = np.random.choice(
            colors,
            size=n_nodes,
            replace=True,
            p=None).tolist()

    nx.set_node_attributes(graph, dict(
        zip(graph, use_colors)), name="color")

    graph.graph["label"] = 0

    return graph


def __cycle_graphs(
        graph,
        colors,
        number,
        n_graphs,
        *,
        two_color=True,
        color_alternate=False,
        **kwargs):

    n_nodes = len(graph)

    if two_color:
        use_colors = cycle([0, 1])

        if color_alternate:
            use_colors = cycle([1, 0])

    else:
        raise NotImplementedError()

    nx.set_node_attributes(graph, dict(zip(graph, use_colors)), name="color")
    graph.graph["label"] = 0

    return graph


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
              split_line=None,
              special_line=False,
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

    elif file_input is not None:
        graph_generator = __graph_file_reader(
            filename=file_input, read_node_label=False)
    else:
        raise ValueError(
            "Must indicate a graph generator function or a filename with the graph structure")

    print("Coloring graphs")
    n_graphs = next(graph_generator)
    possible_colors = list(range(n_colors))

    # no green, green is 0
    # al least N greens in partition 2, in defined in `force_color`
    partition_1 = graph_distribution[0] * n_graphs

    yield number_graphs

    for i, graph in enumerate(graph_generator):
        print(f"{i}/{n_graphs} graphs colored")

        if special_line and structure_fn == "line":
            graph = __special_line(
                graph=graph,
                colors=possible_colors,
                number=i,
                n_graphs=n_graphs,
                **kwargs)

        elif structure_fn == "cycle":
            graph = __cycle_graphs(
                graph=graph,
                colors=possible_colors,
                number=i,
                n_graphs=n_graphs,
                color_alternate=bool(random.getrandbits(1)),
                **kwargs)

        else:
            graph = __coloring_logic(
                graph=graph,
                split_line=split_line,
                number=i,
                partition=partition_1,
                possible_colors=possible_colors,
                node_distribution_1=node_distribution_1,
                node_distribution_2=node_distribution_2,
                structure_fn=structure_fn,
                force_color=force_color,
                force_color_position=force_color_position)

        yield graph


def __tagging_logic(graph, formula):

    labels, graph_label = formula(graph)

    graph.graph["label"] = graph_label

    for node_id in graph:
        graph.node[node_id]["label"] = labels[node_id]

    return graph, graph_label, len(labels), sum(labels)


def tagger(input_file: str,
           formula: Callable[[List[nx.Graph]],
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

        graph, graph_label, num_nodes, num_ones = __tagging_logic(
            graph=graph, formula=formula)

        total_property += graph_label
        total_nodes += num_nodes
        total_tagged += num_ones

        yield graph
    print("-- finished tagging")
    print("-- writting")

    print(f"{total_tagged}/{total_nodes} nodes were tagged 1 ({total_tagged/total_nodes})")
    print(f"{total_property}/{n_graphs} graphs were tagged 1 ({total_property/n_graphs})")


def __red_exist_green(graph: nx.Graph, n_green=1,
                      **kwargs) -> Tuple[List[bool], int]:

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


def __color_no_connected_green(graph: nx.Graph,
                               local_prop=None,
                               global_prop=None,
                               global_constraint=None,
                               condition="and",
                               **kwargs) -> Tuple[List[bool],
                                                  int]:

    if condition  == "and":
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
            neighbor_features = [graph.node[node]['color']
                                 for neighbor in graph.neighbors(node)]

            # color count of my neighbors
            neighbor_color_map = __map_colors(graph, neighbor_features)

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


def tagger_dispatch(tagger, **kwargs):
    options = {
        "formula1": partial(__red_exist_green, **kwargs),
        "formula2": partial(__red_exist_green, **kwargs),
        "formula3": partial(__color_no_connected_green, **kwargs),
    }
    if tagger not in options:
        raise ValueError()
    return options[tagger]


def online_generator(
        node_distribution,
        min_nodes,
        max_nodes,
        structure_fn,
        variable_degree,
        n_colors,
        force_color,
        force_color_position,
        random_state,
        split_line,
        formula,
        **kwargs):

    fn = None
    if structure_fn == "empty":
        fn = __generate_empty_graph

    elif structure_fn == "degree":
        fn = __generate_graph_by_degree

    elif structure_fn == "line":
        fn = __generate_line_graph

    elif structure_fn == "random":
        fn = __generate_random_graph

    else:
        raise ValueError()

    n_nodes = random.randint(min_nodes, max_nodes)
    graph = fn(
        n_nodes=n_nodes,
        seed=random_state,
        use_random=variable_degree,
        **kwargs)

    graph = __coloring_logic(
        graph=graph,
        split_line=split_line,
        number=1,
        partition=0,
        possible_colors=range(n_colors),
        node_distribution_1=None,
        node_distribution_2=node_distribution,
        structure_fn=structure_fn,
        force_color=force_color,
        force_color_position=force_color_position)

    graph = __tagging_logic(graph, formula)

    return graph


def train_dataset(
        name,
        seed,
        n_colors,
        number_of_graphs,
        n_min,
        n_max,
        random_degrees,
        edges,
        tagger_fn,
        no_green=False,
        force_green=None,
        **kwargs):
    random.seed(seed)
    np.random.seed(seed)

    # 1/2 of the graphs do not have green
    # the other 1/2 have at least force_color[0] greens
    # graph_distribution = [0.5, 0.5]
    graph_distribution = [0, 0.5]

    # on the second graph split, force 1 green (0) in each graph
    #force_color = {0: 1, 1: 1}
    # force_color = {0: 1}
    force_color = {}
    #force_pos = {0: 0, 1: -1}
    force_pos = None

    # # 1/2 red (1), 0.5/4 the others
    # red_prob = 0.5
    # #red_prob = 1. / n_colors

    # green_prob = 0
    # #green_prob = 1. / n_colors
    # others = (1. - red_prob - green_prob) / (n_colors - 2)
    # node_distribution_1 = [red_prob] + [others] * (n_colors - 2)

    # #red_prob = 0.9
    # if not no_green:
    #     green_prob = (1. - red_prob) / (n_colors - 1)

    # if force_green is not None:
    #     green_prob = (force_green - 1) / float(n_max)

    # others = (1. - red_prob - green_prob) / (n_colors - 2)
    # node_distribution_2 = [green_prob,
    #                        red_prob] + [others] * (n_colors - 2)

    node_distribution_1=[]
    node_distribution_2=[1./n_colors]*n_colors

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
        force_color_position=force_pos,
        p=edges,
        **kwargs)  # random.random()

    i = random.randint(0, 1000)
    write_graphs(graph_generator, filename=f"temp{i}.txt")

    label_generator = tagger(
        input_file=f"temp{i}.txt",
        formula=tagger_dispatch(tagger_fn, **kwargs))

    if "cycle" in name:
        filename = f"../data/{tagger_fn}/train-{name}-{number_of_graphs}-{n_min}-{n_max}.txt"
    else:
        filename = f"../data/{tagger_fn}/train-{name}-{number_of_graphs}-{n_min}-{n_max}-{kwargs['force_proportion']}.txt"

    write_graphs(
        label_generator,
        # filename=f"../data/train-{name}-{number_of_graphs}-{n_min}-{n_max}-v{green_prob}-v{force_color[0]}-{edges}.txt",
        filename=filename,
        write_features=["color"])


def test_dataset(
        name,
        seed,
        n_colors,
        number_of_graphs,
        n_min,
        n_max,
        random_degrees,
        edges,
        tagger_fn,
        no_green=False,
        force_green=None,
        **kwargs):
    random.seed(seed)
    np.random.seed(seed)

    # 1/2 of the graphs do not have green
    # the other 1/2 have at least force_color[0] greens
    # graph_distribution = [0.5, 0.5]
    graph_distribution = [0, 0.5]

    # on the second graph split, force 1 green (0) in each graph
    #force_color = {0: 1, 1: 1}
    # force_color = {0: 1}
    force_color = {}
    #force_pos = {0: 0, 1: -1}
    force_pos = None

    # # 1/2 red (1), 0.5/4 the others
    # red_prob = 0.5
    # #red_prob = 1. / n_colors

    # green_prob = 0
    # #green_prob = 1. / n_colors
    # others = (1. - red_prob - green_prob) / (n_colors - 2)
    # node_distribution_1 = [red_prob] + [others] * (n_colors - 2)

    # #red_prob = 0.9
    # if not no_green:
    #     green_prob = (1. - red_prob) / (n_colors - 1)

    # if force_green is not None:
    #     green_prob = (force_green - 1) / float(n_max)

    # others = (1. - red_prob - green_prob) / (n_colors - 2)
    # node_distribution_2 = [green_prob,
    #                        red_prob] + [others] * (n_colors - 2)

    node_distribution_1 = []
    node_distribution_2 = [1. / n_colors] * n_colors

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
        force_color_position=force_pos,
        p=edges,
        **kwargs)  # random.random()

    i = random.randint(0, 1000)
    write_graphs(graph_generator, filename=f"temp{i}.txt")

    label_generator = tagger(
        input_file=f"temp{i}.txt",
        formula=tagger_dispatch(tagger_fn, **kwargs))

    if "cycle" in name:
        filename = f"../data/{tagger_fn}/test-{name}-{number_of_graphs}-{n_min}-{n_max}.txt"
    else:
        filename = f"../data/{tagger_fn}/test-{name}-{number_of_graphs}-{n_min}-{n_max}-{kwargs['force_proportion']}.txt"

    write_graphs(
        label_generator,
        # filename=f"../data/test-{name}-{number_of_graphs}-{n_min}-{n_max}-v{green_prob}-v{force_color[0]}-{edges}.txt",
        filename=filename,
        write_features=["color"])


if __name__ == "__main__":
    # TODO: implement manual limit to number of nodes with each color
    """
    formula1 -> x in G, red(x) and exist y in G, such that green(y)

    formula2 -> x in G, green(x) and exist^50 y_i in G, such that red(y_i)
    formula3 -> x in G, red(x) and exist y in G, such that green(y) and edge(x,y)
    """

    # if int -> indices
    #_split_line = {"split": [10]}
    _split_line = None
    _tagger_fn = "formula3"
    _data_name = "random"
    _prop =3

    # only_extreme=True|False

    train_dataset(
        name=_data_name,
        tagger_fn=_tagger_fn,
        seed=None,
        n_colors=5,
        number_of_graphs=5000,
        n_min=75,
        n_max=75,
        random_degrees=True,
        min_degree=0,
        max_degree=2,
        no_green=False,
        special_line=True,
        edges=0.025,
        split_line=_split_line,
        force_proportion=_prop,
        force_green=3,
        two_color=True,
        # tagger
        # formula 1
        n_green=1,
        # formula 3
        local_prop=[1,2],
        global_prop=[0],
        global_constraint={0: 10, 3:10},
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
    #     force_proportion=_prop,
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
    #     force_proportion=_prop,
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
