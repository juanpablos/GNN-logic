from typing import Dict, List, Tuple, Union, Set
from torch_geometric.data import Data
import networkx as nx
import numpy as np
import torch
from sklearn.model_selection import train_test_split


def load_data(dataset: str,
              degree_as_node_label: bool = False,
              graph_type: int = 2,
              undirected=True) -> Tuple[List[Data],
                                        Tuple[int, int, int]]:

    if graph_type == 1:
        raise NotImplementedError()

    print('Loading data...')
    graph_list: List[Data] = []

    unique_graph_label: Set[int] = {}
    unique_node_labels: Set[int] = {}
    unique_node_features: Set[int] = {}

    with open(dataset, 'r') as in_file:
        n_graphs = int(in_file.readline().strip())
        for _ in range(n_graphs):
            n_nodes, graph_label = map(
                int, in_file.readline().strip().split(" "))

            # register graph label (not really important)
            unique_graph_label.add(graph_label)

            graph: nx.Graph = nx.Graph()
            node_labels: List[int] = []
            node_features: List[List[int]] = []

            # --- READING GRAPH ----
            # for each node in the graph
            for node_id in range(n_nodes):
                # add the index (starts at 0 to n_nodes-1)
                graph.add_node(node_id)

                # n_features, [features], node label, n_edges, [neighbors]
                # * only work for categorical node features
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

                unique_node_features.update(features)

                node_features.append(features)
                # ---- /FEATURES ----

                # ---- LABELS ----
                # TODO: support multiple labels
                node_label = node_row[n_features + 1]
                unique_node_labels.add(node_label)

                node_labels.append(node_label)
                # ---- /LABELS ----

                # ---- EDGES ----
                # get the rest, the neighbours
                neighbors = node_row[n_features + 3:]
                # register connections
                for neighbor in neighbors:
                    graph.add_edge(node_id, neighbor)

                    if undirected:
                        graph.add_edge(neighbor, node_id)

                # ---- /EDGES ----
            # --- /READING GRAPH ----

            feature_classes = len(unique_node_features)
            features = torch.tensor(graph.node_features)

            x = torch.nn.functional.one_hot(
                features.squeeze(), feature_classes)
            edges = torch.tensor(list(graph.edges), dtype=torch.long)
            node_labels = torch.tensor(node_labels)

            graph_list.append(
                Data(
                    x=x,
                    edge_index=edges.t().contiguous(),
                    node_labels=node_labels,
                    graph_label=graph_label
                ))

    print(f"#Graphs: {len(graph_list)}")
    print(f"#Graphs Labels: {len(unique_graph_label)}")
    print(f"#Node Properties: {len(unique_node_features)}")
    print(f"#Node Labels: {len(unique_node_labels)}")

    return graph_list, \
        (len(unique_graph_label),
         len(unique_node_features),
         len(unique_node_labels))


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
