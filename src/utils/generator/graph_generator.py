import random
from functools import partial
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union

import networkx as nx
import numpy as np

# TODO: Urgent Refactor


def __generate_empty_graph(n_nodes: int, **kwargs) -> nx.Graph:
    return nx.empty_graph(n=n_nodes)


def __generate_graph_by_degree(
        degrees: List[int] = None,
        *,
        seed: int = None,
        variable_degree: bool = False,
        n_nodes: int = None,
        min_degree: int = 0,
        max_degree: int = 0,
        **kwargs) -> nx.Graph:

    degree_sequence = degrees

    if variable_degree:
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
        p: float = None,
        m: int = None,
        seed: int = None,
        name="erdos",
        **kwargs) -> nx.Graph:

    assert name in ["erdos", "barabasi"]
    if name == "barabasi":
        assert m is not None
    if name == "erdos":
        assert p is not None

    if name == "erdos":
        if m is not None:
            return nx.gnm_random_graph(n=n_nodes, m=n_nodes * m, seed=seed)

        return nx.fast_gnp_random_graph(n=n_nodes, p=p, seed=seed)

    elif name == "barabasi":
        return nx.barabasi_albert_graph(n=n_nodes, m=m, seed=seed)

    else:
        raise ValueError()


def __generate_cycle_graph(n_nodes: int, pair=True, **kwargs):
    nodes_in_graph = n_nodes
    if pair and n_nodes % 2 != 0:
        nodes_in_graph += 1

    return nx.cycle_graph(n=nodes_in_graph)


def graph_generator(generator_fn: str,
                    min_nodes: int,
                    max_nodes: int,
                    **kwargs) -> Generator[nx.Graph, None, None]:

    fn = None
    if generator_fn == "empty":
        fn = __generate_empty_graph

    elif generator_fn == "degree":
        fn = __generate_graph_by_degree

    elif generator_fn == "line":
        fn = __generate_line_graph

    elif generator_fn == "random":
        fn = partial(__generate_random_graph, **kwargs)

    elif generator_fn == "cycle":
        fn = __generate_cycle_graph
    else:
        raise ValueError()

    print("Start generating graphs")

    while True:
        n_nodes = random.randint(min_nodes, max_nodes)
        yield fn(n_nodes=n_nodes, **kwargs)
