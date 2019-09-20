import random
from itertools import cycle
from typing import Dict, Generator, List, Tuple

import networkx as nx
import numpy as np


class ColorDistributionSplit():
    def __init__(
            self,
            number_graphs,
            graph_split,
            colors,
            color_distributions,
            force_color,
            force_color_position,
            greens):

        splits = np.array(graph_split) * number_graphs
        self.graph_splits = np.cumsum(splits)
        self.current_split = 0

        self.colors = colors
        self.color_distributions = color_distributions if color_distributions is not None else {}
        self.force_color = force_color if force_color is not None else {}
        self.force_color_position = force_color_position

        self.limit_greens = False
        if greens is not None:
            self.min_greens, self.max_greens = greens
            self.limit_greens = True

    def __call__(self, index, num_nodes):
        assert isinstance(index, int)
        if index > self.graph_splits[self.current_split]:
            self.current_split += 1

        forced_colors = []
        if self.current_split in self.force_color:
            for c, t in self.force_color[self.current_split].items():
                forced_colors.extend([c] * t)

        color_distribution = None
        if self.current_split in self.color_distributions:
            color_distribution = self.color_distributions[self.current_split]

        if self.limit_greens:
            n_greens = random.randint(self.min_greens, self.max_greens)
            greens = [0] * n_greens

            if color_distribution is not None:
                color_distribution = color_distribution[1:]

            colors = np.random.choice(
                self.colors[1:],
                size=num_nodes - n_greens,
                replace=True,
                p=color_distribution).tolist() + greens

        else:
            colors = np.random.choice(
                self.colors,
                size=num_nodes - len(forced_colors),
                replace=True,
                p=color_distribution).tolist() + forced_colors

        colors = np.array(colors)
        np.random.shuffle(colors)

        # if self.current_split in self.force_color_position:
        #     raise NotImplementedError()

        return colors


def __split_line(graph: nx.Graph,
                 i: int,
                 split_line: Dict[str,
                                  List[int]],
                 partition: int,
                 colors: List[int],
                 distribution: List[float]) -> Tuple[nx.Graph,
                                                     np.ndarray]:
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


def __coloring_logic(graph: nx.Graph,
                     color_dispatch: ColorDistributionSplit,
                     iteration: int,
                     **kwargs):

    colors = color_dispatch(index=iteration, num_nodes=len(graph))
    nx.set_node_attributes(graph, dict(zip(graph, colors)), name="color")

    return graph


def __special_line(
        graph: nx.Graph,
        colors: List[int],
        number: int,
        n_graphs: int,
        only_extreme: bool = False,
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

    nx.set_node_attributes(graph, dict(zip(graph, use_colors)), name="color")

    return graph


def __cycle_graphs(
        graph: nx.Graph,
        colors: List[int],
        *,
        two_color: bool = True,
        color_alternate: bool = False,
        **kwargs):

    n_nodes = len(graph)

    if two_color:
        use_colors = cycle([0, 1])

        if color_alternate:
            use_colors = cycle([1, 0])

    else:
        raise NotImplementedError()

    nx.set_node_attributes(graph, dict(zip(graph, use_colors)), name="color")

    return graph


def __centroid_graphs(
        graph: nx.Graph,
        colors: List[int],
        **kwargs):

    only_green_centroid = kwargs.get("centroid_only_green", True)

    use_colors = colors
    if only_green_centroid:
        use_colors = use_colors[1:]

    # TODO: support node proportion
    node_colors = np.random.choice(
        use_colors,
        size=len(graph),
        replace=True,
        p=None)

    nx.set_node_attributes(graph, dict(zip(graph, node_colors)), name="color")
    central_nodes = filter(lambda name: name.split("-")[1] == "0", graph)

    nx.set_node_attributes(
        graph, dict.fromkeys(
            central_nodes, 0), name="color")

    return graph


def color_generator(graph_generator: Generator[nx.Graph, None, None],
                    number_graphs: int,
                    min_nodes: int,
                    max_nodes: int,
                    structure_fn: str,
                    verbose=False,
                    graph_split: List[float] = None,
                    color_distributions: Dict[int, List[float]] = None,
                    n_colors: int = 10,
                    seed: int = None,
                    special_line: bool = False,
                    force_color: Dict[int, Dict[int, int]] = None,
                    force_color_position: Dict[int, Dict[int, int]] = None,
                    greens: Tuple[int, int] = None,
                    **kwargs) -> Generator[nx.Graph, None, None]:

    if graph_split is not None:
        assert sum(graph_split) - 1 < 1e4
    else:
        graph_split = [1.]

    if color_distributions is not None:
        assert len(graph_split) == len(color_distributions)
        assert all([n_colors == len(c) for c in color_distributions.values()])

    possible_colors = list(range(n_colors))

    distribution_dispatch = ColorDistributionSplit(
        number_graphs=number_graphs,
        graph_split=graph_split,
        colors=possible_colors,
        color_distributions=color_distributions,
        force_color=force_color,
        force_color_position=force_color_position,
        greens=greens)

    for i in range(1, number_graphs + 1):
        graph = next(graph_generator)

        if special_line and structure_fn == "line":
            graph = __special_line(
                graph=graph,
                colors=possible_colors,
                number=i,
                n_graphs=number_graphs,
                **kwargs)

        elif structure_fn == "cycle":
            graph = __cycle_graphs(
                graph=graph,
                colors=possible_colors,
                color_alternate=bool(random.getrandbits(1)),
                **kwargs)

        elif structure_fn == "centroid":
            graph = __centroid_graphs(
                graph=graph,
                colors=possible_colors,
                **kwargs)

        elif structure_fn == "normal":
            graph = __coloring_logic(graph=graph,
                                     color_dispatch=distribution_dispatch,
                                     iteration=i,
                                     **kwargs)

        else:
            raise ValueError()

        if verbose:
            print(f"{i}/{number_graphs} graphs colored")
        yield graph
