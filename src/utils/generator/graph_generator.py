import itertools
import random
from functools import partial
from typing import Any, Dict, Generator, List, Tuple

import networkx as nx


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
        assert p is not None or m is not None

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


def __generate_star_graph(n_nodes=None, **kwargs):
    return nx.star_graph(n_nodes)


def __generate_grid_graph(
        grid_n,
        grid_m,
        periodic=False,
        diagonal=False,
        **kwargs):

    graph = nx.grid_2d_graph(n=grid_n, m=grid_m, periodic=periodic)

    if diagonal:
        graph.add_edges_from([
            ((x, y), (x + 1, y + 1))
            for x in range(grid_m - 1)
            for y in range(grid_n - 1)
        ] + [
            ((x + 1, y), (x, y + 1))
            for x in range(grid_m - 1)
            for y in range(grid_n - 1)
        ])

        if periodic:
            graph.add_edges_from([
                ((0, y), (grid_m - 1, y + 1))
                for y in range(grid_n - 1)
            ] + [
                ((0, y + 1), (grid_m - 1, y))
                for y in range(grid_n)
            ] + [
                ((x, 0), (x + 1, grid_n - 1))
                for x in range(grid_m)
            ] + [
                ((x + 1, 0), (x, grid_n - 1))
                for x in range(grid_m)
            ])

    return graph


def __create_centroids(gen_fun,
                       centroids: Tuple[int, int],
                       nodes_per_centroid: Tuple[int, int],
                       centroid_connectivity: float,
                       centroid_extra: Dict[str, Any] = None,
                       n_nodes=None,
                       **kwargs):

    if centroid_extra:
        raise NotImplementedError()

    graphs = [nx.null_graph()]
    n_centroids = random.randint(*centroids)
    for _ in range(n_centroids):
        n_nodes_centroid = random.randint(*nodes_per_centroid)
        graphs.append(gen_fun(n_nodes_centroid, **kwargs))

    graph = nx.union_all(
        graphs,
        rename=map(
            lambda i: str(i)
            + "-",
            range(
                len(graphs))))

    central_nodes = filter(lambda name: name.split("-")[1] == "0", graph)
    centroid_edges = itertools.combinations(central_nodes, 2)

    for node_1, node_2 in centroid_edges:
        if random.random() < centroid_connectivity:
            graph.add_edge(node_1, node_2)

    return graph


def graph_generator(generator_fn: str,
                    min_nodes: int,
                    max_nodes: int,
                    create_centroids: bool = False,
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

    elif generator_fn == "star":
        fn = __generate_star_graph

    elif generator_fn == "grid":
        fn = __generate_grid_graph
    else:
        raise ValueError()

    if create_centroids:
        fn = partial(__create_centroids, gen_fun=fn)

    print("Start generating graphs")

    while True:
        n_nodes = random.randint(min_nodes, max_nodes)
        yield fn(n_nodes=n_nodes, **kwargs)
