from typing import Dict, List, Tuple, Union

import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from torch.jit.annotations import TensorType


class S2VGraph(object):
    def __init__(
            self,
            graph: nx.Graph,
            graph_label: int,
            node_features: List[List[int]],
            node_labels: List[int],
            neighbours: List[List[int]],
            max_neighbour: int
    ):
        """
            graph: a networkx graph
            graph_label: an integer graph label
            node_labels: a list of integer node tags
            node_features: a torch float tensor, one-hot representation of the tag that is used as input to neural nets
            edge_mat: a torch long tensor, contain edge list, will be used to create torch sparse tensor
            neighbors: list of neighbors (without self-loop)
        """
        self.graph: nx.Graph = graph
        self.graph_label: int = graph_label
        self.node_features: Union[List[List[int]], TensorType] = node_features
        self.node_labels: List[int] = node_labels
        self.neighbors: List[List[int]] = neighbours
        self.max_neighbour: max_neighbour

        self.edge_mat: TensorType = None

        self.aux_node_label_mapping: Dict[int, int] = None

# TODO: label indices are set by appearance order. Should implement fixed
# ! classes such that they actually represent their index.


def load_data(dataset: str,
              degree_as_node_label: bool = False,
              graph_type: int = 2,) -> Tuple[List[S2VGraph],
                                             Tuple[int, int, int]]:

    if graph_type == 1:
        raise NotImplementedError()

    print('Loading data...')
    graph_list: List[S2VGraph] = []
    graph_label_dict: Dict[int, int] = {}
    # {real_label : index}
    node_labels: Dict[int, int] = {}
    node_features: Dict[int, int] = {}

    with open(dataset, 'r') as in_file:
        n_graphs = int(in_file.readline().strip())
        for _ in range(n_graphs):
            n_nodes, graph_label = map(
                int, in_file.readline().strip().split(" "))
            # register graph label (not really important)
            if graph_label not in graph_label_dict:
                # index the graph_label
                graph_label_dict[graph_label] = len(graph_label_dict)

            graph = nx.Graph()
            _nodes_label: List[int] = []
            max_neighbours: int = 0
            _nodes_features: List[List[int]] = []
            neighbour_collection = []

            # --- READING GRAPH ----
            # for each node in the graph
            for node_id in range(n_nodes):
                # add the index (starts at 0 to n_nodes-1)
                graph.add_node(node_id)

                # n_features, [features], node label, n_edges, [neighbours]
                # TODO: only work for categorical node features
                node_row = list(
                    map(int, in_file.readline().strip().split(" ")))

                # ---- FEATURES ----
                # first comes the number of features
                n_features = node_row[0]
                # if n_features > max_features:
                #     max_features = n_features
                features = []
                if n_features != 0:
                    # get all features, column 1 to n_features-1
                    features = node_row[1:n_features + 1]
                # we need the index of each featur
                _features = []
                for f in features:
                    if f not in node_features:
                        # index it
                        node_features[f] = len(node_features)
                    _features.append(node_features[f])

                _nodes_features.append(_features)
                # ---- /FEATURES ----

                # ---- LABELS ----
                node_label = node_row[n_features + 1]
                if node_label not in node_labels:
                    node_labels[node_label] = len(node_labels)
                # append the node label, not the index. This is for
                # compatibility after when using `degree_as_node_label`
                _nodes_label.append(node_label)
                # ---- /LABELS ----

                # get the number of neighbours
                n_neighbours = node_row[n_features + 2]
                if n_neighbours > max_neighbours:
                    max_neighbours = n_neighbours

                # ---- EDGES ----
                # get the rest, the neighbours
                neighbours = node_row[n_features + 3:]
                # register connections
                # * we are assuming the graph comes well formatted and is an undirected graph
                for neighbour in neighbours:
                    graph.add_edge(node_id, neighbour)

                neighbour_collection.append(neighbours)

                # ---- /EDGES ----
            # --- /READING GRAPH ----

            # adds the graph structure and the graph label
            graph_list.append(
                S2VGraph(
                    graph=graph,
                    graph_label=graph_label_dict[graph_label],
                    node_features=_nodes_features,
                    node_labels=_nodes_label,
                    neighbours=neighbour_collection,
                    max_neighbour=max_neighbours))

    # add labels and edge_mat
    # for each of the graphs
    for graph in graph_list:

        # create edges to make matrix
        edges = [list(pair) for pair in graph.graph.edges()]
        # reciprocal
        edges.extend([[i, j] for j, i in edges])

        # generate the edge mapping with 2 lists.
        # matrix is (2,2xE),
        # 2-> node in - node out
        # 2xE, double the number of edges
        graph.edge_mat = torch.LongTensor(edges).transpose(0, 1)

        # * in case the nodes do not have a label (placeholder for format), assign the node degree as label
        if degree_as_node_label:
            graph.node_labels = list(dict(graph.graph.degree).values())

    # Extracting unique tag labels
    if degree_as_node_label:
        # * used when degree_as_node_label=True, useless otherwise
        tagset = set()
        for graph in graph_list:
            tagset = tagset.union(set(graph.node_features))
        tagset = list(tagset)
        # when label=degree it is needed to index value
        label2index = {tagset[i]: i for i in range(len(tagset))}

        for graph in graph_list:
            graph.aux_node_label_mapping = label2index
            indexed_features = []
            for features in graph.node_features:
                indexed_features.append([label2index[i] for i in features])
            graph.node_features = indexed_features

    # creates the node feature matrix
    for graph in graph_list:
        # matrix (n_nodes, n_features)
        feature_classes = len(node_features)
        features = torch.tensor(graph.node_features)
        # * 1-hot encoding for each feature
        # ! only supports 1-hot
        graph.node_features = torch.nn.functional.one_hot(
            features, feature_classes).squeeze()

    print(f"#Graphs: {len(graph_list)}")
    print(f"#Graphs Labels: {len(graph_label_dict)}")
    print(f"#Node Properties: {len(node_features)}")
    print(f"#Node Labels: {len(node_labels)}")

    return graph_list, \
        (len(graph_label_dict), len(node_features), len(node_labels))


def separate_data(graph_list: List[S2VGraph], seed: int,
                  test_size: float = 0.2) -> Tuple[List[S2VGraph],
                                                   List[S2VGraph]]:

    return train_test_split(
        graph_list,
        random_state=seed,
        test_size=test_size,
        shuffle=True)


if __name__ == "__main__":
    load_data(dataset="colors1_labeled.txt")
