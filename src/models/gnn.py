
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import MLP


class GNN(nn.Module):
    def __init__(
            self,
            num_layers: int,
            num_mlp_layers: int,
            input_dim: int,  # node representations, (n_features)
            hidden_dim: int,
            output_dim: int,
            final_dropout: float,
            learn_eps: bool,
            combine_type: str,
            aggregate_type: str,
            readout_type: str,
            recursive_weighting: bool,
            task: str,
            device: torch.device
    ):

        super(GNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers
        self.learn_eps = learn_eps
        # ? should we support this? Center node weighting
        self.eps = nn.Parameter(torch.zeros(self.num_layers - 1))

        self.recursive_weighting = recursive_weighting
        self.task = task

        self.combine = self.__get_combine_fn(combine_type)
        self.readout = self.__get_readout_fn(readout_type)
        self.aggregate = self.__get_aggregate_fn(aggregate_type)

        self.node_preprocess = self.__preprocess_neighbors_maxpool if aggregate_type == "max" else self.__preprocess_neighbors_sumavgpool

        self.compute_layer = self._next_layer_eps if learn_eps else self._next_layer

        # Linear function that maps the hidden representation
        # for each network layer.
        # if recursive_weighting=False only use the last representation for the
        # readout
        self.linear_predictions = torch.nn.ModuleList()
        if recursive_weighting:
            for layer in range(num_layers):
                if layer == 0:
                    self.linear_predictions.append(
                        nn.Linear(input_dim, output_dim))
                else:
                    self.linear_predictions.append(
                        nn.Linear(hidden_dim, output_dim))
        else:
            self.linear_predictions.append(
                nn.Linear(input_dim, output_dim))

        self.V = torch.nn.ModuleList()
        self.A = torch.nn.ModuleList()
        self.R = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.V.append(nn.Linear(input_dim, input_dim))
            self.A.append(nn.Linear(input_dim, input_dim))
            self.R.append(nn.Linear(input_dim, input_dim))
        # ? self.b = torch.nn.ModuleList()

    def __get_combine_fn(self, combine_type):
        # return a funtion that takes 3 parameters
        # the hidden representetion of the node: x1
        # the aggregated representation of its neighbors: x2
        # the readpout of the whole graph: x3
        # returns a tensor of the same dimensions of x1|x2
        options = {
            "sum": partial(self.__functional_combine, function="sum"),
            "average": partial(self.__functional_combine, function="average"),
            "max": partial(self.__functional_combine, function="max"),
            "trainable": self.__trainable_combine}
        if combine_type not in options:
            raise ValueError()
        return options[combine_type]

    def __get_aggregate_fn(self, aggregate_type):
        # this should return a function that takes the hidden representation of each node and the neighbors for each node.
        # It should return a new representation, composed by all
        # neighbours of the node. (it does not combine it with the current
        # node, that is the combine function. This function just aggregates the
        # neighbor nodes and computes a new representation).
        options = {
            "sum": partial(self.__node_sumavgpool, average=False),
            "average": partial(self.__node_sumavgpool, average=True),
            "max": self.__node_maxpool}
        if aggregate_type not in options:
            raise ValueError()
        return options[aggregate_type]

    def __get_readout_fn(self, readout_type):
        # returns a funtion that performs a full graph aggregation
        # takes all node's representations as input and returns a single
        # vector of size (1, n_features), that is the combination of
        # the hidden representations of all nodes in the graph
        options = {
            "sum": partial(self.__graph_sumavgpool, average=False),
            "average": partial(self.__graph_sumavgpool, average=True),
            "max": self.__graph_maxpool}
        if readout_type not in options:
            raise ValueError()
        return options[readout_type]

    def __node_maxpool(self, h, aux_data):
        # aux_data must be a padded neighbor list for each node in the batch,
        # each node must be indexed by their graph index.

        # Return has shape (nodes, features)
        # just use this to assign to the -1 neighbors -> padding
        # TODO: this count as combine?
        dummy, _ = torch.min(h, dim=0)
        # append the min to assign as -1 padding
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        # take the representation for each node's neighbors. Assign the min to
        # the -1 padding then take the max for each node (pool the neighbors)
        pooled_rep, _ = torch.max(h_with_dummy[aux_data], dim=1)
        return pooled_rep

    def __graph_maxpool(self, h):
        # h must be a matrix with all node representations.
        # shape (n_batch_nodes, feature_classes)
        # Return has shape (1, features)
        pooled_rep, _ = torch.max(h, dim=0, keepdim=True)
        return pooled_rep

    def __node_sumavgpool(self, h, aux_data, average=False):
        # aux_data must be a adjancency matrix of dims
        # (number of nodes, number of nodes), number of nodes in the current
        # graph batch, indexed by their graph index
        # * because h is an stacked matrix of features for nodes in the current batch, mutiplying with the adjancency matrix is the same as adding them.
        # Return has shape (nodes, features)

        pooled_rep = torch.spmm(aux_data, h)

        if average:
            # If average pooling
            degree = torch.spmm(
                aux_data, torch.ones((aux_data.shape[0], 1)).to(self.device))
            pooled_rep = pooled_rep / degree

        return pooled_rep

    def __graph_sumavgpool(self, h, average=False):
        # h must be a matrix with all node representations.
        # shape (n_batch_nodes, feature_classes)
        # Return has shape (1, features)
        if average:
            return torch.mean(h, dim=0, keepdim=True)
        else:
            return torch.sum(h, dim=0, keepdim=True)

    def __functional_combine(self, x1, x2, x3, function="max", **kwargs):
        # x1: node representations, shape (nodes, features)
        # x2: node aggregations, shape (nodes, features)
        # x3: graph readout, shape (1, features)
        # TODO: allow for weighted sum/mean
        # same memory allocations, only references
        expanded_x3 = x3.expand(x1.size())
        combined = torch.cat(
            [x1.unsqueeze(0), x2.unsqueeze(0), expanded_x3.unsqueeze(0)])
        if function == "max":
            combined, _ = torch.max(combined, dim=0)
            return combined
        elif function == "sum":
            return torch.sum(combined, dim=0)
        elif function == "average":
            return torch.mean(combined, dim=0)
        else:
            raise ValueError()

    def __trainable_combine(
            self,
            x1,
            x2,
            x3,
            layer,
            activation="relu",
            **kwargs):
        inner = self.V[layer](x1) + self.A[layer](x2) + self.R[layer](x3)
        # ? + self.b[layer].unsqueeze(dim=0)

        # TODO: dropout?
        # TODO: remove this relu and put it in _next_layer?
        if activation == "relu":
            return F.relu(inner)
        else:
            raise NotImplementedError()

    def __preprocess_neighbors_maxpool(self, batch_graph):
        # create padded_neighbor_list in concatenated graph

        # compute the maximum number of neighbors within the graphs in the
        # current minibatch
        max_deg = max([graph.max_neighbor for graph in batch_graph])

        padded_neighbor_list = []
        start_idx = [0]

        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.graph))
            padded_neighbors = []
            for j in range(len(graph.neighbors)):
                # add off-set values to the neighbor indices
                pad = [n + start_idx[i] for n in graph.neighbors[j]]
                # padding, dummy data is assumed to be stored in -1
                pad.extend([-1] * (max_deg - len(pad)))

                # TODO: does this count as combine?
                # Add center nodes in the maxpooling if learn_eps is False,
                # i.e., aggregate center nodes and neighbor nodes altogether.
                if not self.learn_eps:
                    pad.append(j + start_idx[i])

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list)

    def __preprocess_neighbors_sumavgpool(self, batch_graph):
        # create block diagonal sparse matrix

        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.graph))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

        # TODO: does this count as combine?
        # Add self-loops in the adjacency matrix if learn_eps is False, i.e.,
        # aggregate center nodes and neighbor nodes altogether.
        if not self.learn_eps:
            num_node = start_idx[-1]
            self_loop_edge = torch.LongTensor(
                [range(num_node), range(num_node)])
            elem = torch.ones(num_node)
            Adj_block_idx = torch.cat([Adj_block_idx, self_loop_edge], 1)
            Adj_block_elem = torch.cat([Adj_block_elem, elem], 0)

        Adj_block = torch.sparse.FloatTensor(
            Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device)

    def _next_layer_eps(self, h, layer, aux_data):
        raise NotImplementedError()

    def _next_layer(self, h, layer, aux_data):
        raise NotImplementedError()

    def forward(self, batch_graph):
        # Stack node features -> result is a matrix of size (nodes, features)
        # nodes -> nodes in the batch
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).type(
            torch.FloatTensor).to(self.device)

        aux_data = self.node_preprocess(batch_graph)
        # list of hidden representation at each layer (including input)
        # TODO: recursive_weighting
        # hidden_rep = [X_concat]
        h = X_concat
        for layer in range(self.num_layers):
            h = self.compute_layer(h=h, layer=layer, aux_data=aux_data)
            # hidden_rep.append(h)

        if self.task == "node":
            if self.recursive_weighting:
                # TODO: how should we hadle this case?
                # ? weight every layer and then sum?
                raise NotImplementedError()
            else:
                # TODO: activation function? sigmoid?
                return F.sigmoid(self.linear_predictions[-1](h))

        elif self.task == "graph":
            raise NotImplementedError()
        else:
            raise ValueError()
