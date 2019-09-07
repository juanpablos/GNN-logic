
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init

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
            combine_type: str,
            aggregate_type: str,
            readout_type: str,
            mlp_aggregate: str,
            recursive_weighting: bool,
            task: str,
            input_factor: int,
            device: torch.device
    ):

        super(GNN, self).__init__()

        self.final_dropout = final_dropout
        self.device = device
        self.num_layers = num_layers

        self.recursive_weighting = recursive_weighting
        self.task = task

        # combine function to use
        self.combine = self.__get_combine_fn(
            combine_type, mlp_aggregate=mlp_aggregate)
        # readout function to use
        self.readout = self.__get_readout_fn(readout_type)
        # aggregate function to use
        self.aggregate = self.__get_aggregate_fn(aggregate_type)

        # preprocess depends on the aggregation type
        self.node_preprocess = self.__preprocess_neighbors_maxpool if aggregate_type == "max" else self.__preprocess_neighbors_sumavgpool

        if input_dim > hidden_dim:
            raise ValueError(
                "Input dim cannot be larger than the hidden dims. Increase the hidden dims to at least input dim size.")
        self.padding = nn.ConstantPad1d((0, hidden_dim - input_dim), value=0)

        # Batch norms applied to the combied representation
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers):
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function that maps the hidden representation for each network
        # layer. if recursive_weighting=False only use the last representation
        # for the readout
        self.linear_predictions = torch.nn.ModuleList()
        if recursive_weighting:
            for layer in range(num_layers):
                self.linear_predictions.append(
                    nn.Linear(hidden_dim, output_dim))
        else:
            self.linear_predictions.append(nn.Linear(hidden_dim, output_dim))

        if combine_type == "trainable":
            self.V = torch.nn.ModuleList()
            self.A = torch.nn.ModuleList()
            for layer in range(num_layers):
                self.V.append(nn.Linear(hidden_dim, hidden_dim))
                self.A.append(nn.Linear(hidden_dim, hidden_dim))

            if input_factor == 3:
                self.R = torch.nn.ModuleList()
                for layer in range(num_layers):
                    self.R.append(nn.Linear(hidden_dim, hidden_dim))

        # If the combine type is MLP
        if combine_type == "mlp":
            self.mlps = torch.nn.ModuleList()
            for layer in range(self.num_layers):
                # * this is needed because mlp_aggregate=concat can mean
                # * (nodes, hidden*2) or a (nodes, hidden*3) matrix.
                # * aggregate=sum|avg|max mean a (nodes, hidden) matrix
                if mlp_aggregate == "concat":
                    self.mlps.append(
                        MLP(num_mlp_layers, hidden_dim * input_factor, hidden_dim, hidden_dim))
                else:
                    self.mlps.append(
                        MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

    def __get_combine_fn(self, combine_type, *, mlp_aggregate=None):
        # return a funtion that takes 3 parameters
        # the hidden representation of the node: x1
        # the aggregated representation of its neighbors: x2
        # the readout of the whole graph: x3
        # returns a tensor of the same dimensions of x1|x2
        # * return dimension (nodes, hidden)
        options = {
            "sum": partial(self.__functional_combine, function="sum"),
            "average": partial(self.__functional_combine, function="average"),
            "max": partial(self.__functional_combine, function="max"),
            "trainable": self.__trainable_combine,
            "mlp": partial(self.__mlp_combine, aggregate=mlp_aggregate)}
        if combine_type not in options:
            raise ValueError()
        return options[combine_type]

    def __get_aggregate_fn(self, aggregate_type):
        # this should return a function that takes the hidden representation of each node and the neighbors for each node.
        # It should return a new representation, composed by all neighbours of the node.
        # (it does not combine it with the current node, that is the combine function. This function just aggregates the neighbor nodes and computes a new representation).
        # * return dimension (nodes, hidden)
        options = {
            "sum": partial(self.__node_sumavgpool, average=False),
            "average": partial(self.__node_sumavgpool, average=True),
            "max": self.__node_maxpool}
        if aggregate_type not in options:
            raise ValueError()
        return options[aggregate_type]

    def __get_readout_fn(self, readout_type):
        # returns a funtion that performs a full graph aggregation.
        # takes all node's representations as input and returns a single
        # vector representation, that is the combination of
        # the hidden representations of all nodes in the graph
        # * return dimension (1, hidden)
        options = {
            "sum": partial(self.__graph_sumavgpool, average=False),
            "average": partial(self.__graph_sumavgpool, average=True),
            "max": self.__graph_maxpool}
        if readout_type not in options:
            raise ValueError()
        return options[readout_type]

    def __node_maxpool(self, h, aux_data):
        # aux_data must be a padded neighbor list for each node in the batch,
        # each node must be indexed by their graph index. Padding must be -1
        # * return dimension (nodes, hidden)
        # just use this to assign to the -1 neighbors -> padding
        dummy, _ = torch.min(h, dim=0)
        # append the min to assign as -1 padding
        h_with_dummy = torch.cat([h, dummy.reshape((1, -1)).to(self.device)])
        # take the representation for each node's neighbors. Assign the min to
        # the -1 padding then take the max for each node (pool the neighbors)
        if aux_data.nelement() == 0:
            pooled_rep = h
        else:
            pooled_rep, _ = torch.max(h_with_dummy[aux_data], dim=1)
        return pooled_rep

    def __graph_maxpool(self, h, indices):
        # h must be a matrix with all node representations.
        # shape (nodes, hidden)
        # * return dimension (nodes, hidden)

        pooled_hidden = []

        for index in range(1, len(indices)):
            start = indices[index - 1]
            end = indices[index]
            graph = h[start:end]

            pooled_rep, _ = torch.max(graph, dim=0, keepdim=True)
            pooled_rep = pooled_rep.expand(graph.size())

            pooled_hidden.append(pooled_rep)

        return torch.cat(pooled_hidden, dim=0).to(self.device)

    def __node_sumavgpool(self, h, aux_data, average=False):
        # aux_data must be a adjancency matrix of dims
        # (nodes, nodes), number of nodes in the current
        # graph batch, indexed by their graph index
        # * return dimension (nodes, hidden)

        # because h is an stacked matrix of hidden features for nodes in the
        # current batch, mutiplying with the adjancency matrix is the same as
        # adding them.
        pooled_rep = torch.spmm(aux_data, h)

        if average:
            # If average pooling
            degree = torch.spmm(
                aux_data, torch.ones((aux_data.shape[0], 1)).to(self.device))
            degree[degree == 0.0] = 1

            pooled_rep = pooled_rep / degree

        return pooled_rep

    def __graph_sumavgpool(self, h, indices, average=False):
        # h must be a matrix with all node representations.
        # shape (nodes, hidden)
        # * return dimension (nodes, hidden)
        pooled_hidden = []

        for index in range(1, len(indices)):
            start = indices[index - 1]
            end = indices[index]
            graph = h[start:end]

            if average:
                pooled_rep = torch.mean(graph, dim=0, keepdim=True)
            else:
                pooled_rep = torch.sum(graph, dim=0, keepdim=True)

            pooled_rep = pooled_rep.expand(graph.size())

            pooled_hidden.append(pooled_rep)

        return torch.cat(pooled_hidden, dim=0).to(self.device)

    def __functional_combine(
            self,
            x1,
            x2,
            x3=None,
            *,
            function="max",
            **kwargs):
        # x1: node representations, shape (nodes, hidden)
        # x2: node aggregations, shape (nodes, hidden)
        # - x3: graph readout, shape (1, hidden)

        if x3 is None:
            combined = torch.cat([x1.unsqueeze(0), x2.unsqueeze(0)])
        else:
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

    def __trainable_combine(self, x1, x2, x3=None, *, layer, **kwargs):
        # x1: node representations, shape (nodes, hidden)
        # x2: node aggregations, shape (nodes, hidden)
        # - x3: graph readout, shape (1, hidden)

        if x3 is None:
            return self.V[layer](x1) + self.A[layer](x2)
        else:
            return self.V[layer](x1) + self.A[layer](x2) + self.R[layer](x3)

    def __mlp_combine(self, x1, x2, x3=None, *, layer, aggregate, **kwargs):
        # x1: node representations, shape (nodes, hidden)
        # x2: node aggregations, shape (nodes, hidden)
        # - x3: graph readout, shape (1, hidden)

        if aggregate == "concat":
            if x3 is None:
                combined = torch.cat([x1, x2], dim=1)
            else:
                broad_x3 = x3.expand(x1.size())
                combined = torch.cat([x1, x2, broad_x3], dim=1)
        else:
            combined = self.__functional_combine(
                x1=x1, x2=x2, x3=x3, function=aggregate)

        # * combined is (nodes, feathiddenures) matrix if aggregate=sum|avg|max
        # * and (nodes, hidden*(2|3)) when aggregate=concat
        h = self.mlps[layer](combined)
        return h

    def __preprocess_neighbors_maxpool(self, batch_graph):
        # TODO: preprocess outside the network. Dataloader
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

                # * dont include central nodes, combine is done separately

                padded_neighbors.append(pad)
            padded_neighbor_list.extend(padded_neighbors)

        return torch.LongTensor(padded_neighbor_list), start_idx

    def __preprocess_neighbors_sumavgpool(self, batch_graph):
        # TODO: preprocess outside the network. Dataloader
        # create block diagonal sparse matrix
        edge_mat_list = []
        start_idx = [0]
        for i, graph in enumerate(batch_graph):
            start_idx.append(start_idx[i] + len(graph.graph))
            edge_mat_list.append(graph.edge_mat + start_idx[i])
        Adj_block_idx = torch.cat(edge_mat_list, 1)
        Adj_block_elem = torch.ones(Adj_block_idx.shape[1])

       # * dont include central nodes, combine is done separately

        Adj_block = torch.sparse.FloatTensor(
            Adj_block_idx, Adj_block_elem, torch.Size([start_idx[-1], start_idx[-1]]))

        return Adj_block.to(self.device), start_idx

    def compute_layer(self, h, layer, aux_data):
        raise NotImplementedError()

    class delete:
        def __init__(self):
            self.n = 0

        def __call__(self):
            self.n += 1
            return self.n

    d = delete()

    def forward(self, batch_graph):
        # Stack node features -> result is a matrix of size (nodes, features)
        # then add paddind with 0 to fit hidden_dims
        # * (nodes, features)
        X_concat = torch.cat([graph.node_features for graph in batch_graph], 0).type(
            torch.FloatTensor).to(self.device)
        # * (nodes, hidden), padded
        X_concat = self.padding(X_concat)

        aux_data, graph_indices = self.node_preprocess(batch_graph)

        h = X_concat
        for layer in range(self.num_layers):
            combined_rep = self.compute_layer(
                h=h, layer=layer, aux_data=aux_data, indices=graph_indices)

            h = self.batch_norms[layer](combined_rep)
            # hidden_rep.append(h)

        if self.task == "node":
            if self.recursive_weighting:
                # TODO: how should we hadle this case?
                # ? weight every layer and then sum?
                raise NotImplementedError()
            else:
                return self.linear_predictions[-1](h)

        elif self.task == "graph":
            raise NotImplementedError()
        else:
            raise ValueError()
