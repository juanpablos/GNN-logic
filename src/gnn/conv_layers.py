import torch.nn as nn
import torch
import torch_geometric.nn as geom_nn
from torch_geometric.nn.conv import MessagePassing
from .mlp import MLP


class ACRConv(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            aggregate_type: str,
            readout_type: str,
            combine_type: str,
            num_mlp_layers: int,
            **kwargs):

        assert aggregate_type in ["add", "mean", "max"]
        assert combine_type in ["simple", "mlp"]
        assert readout_type in ["add", "mean", "max"]

        super(ACRConv, self).__init__(aggr=aggregate_type, **kwargs)

        if combine_type == "mlp":
            self.mlp1 = MLP(
                num_layers=num_mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim)
            self.mlp2 = MLP(
                num_layers=num_mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim)
            self.mlp3 = MLP(
                num_layers=num_mlp_layers,
                input_dim=output_dim,
                hidden_dim=output_dim,
                output_dim=output_dim)

            self.combine = self.__combine_mlp
        else:
            self.V = nn.Linear(input_dim, output_dim)
            self.A = nn.Linear(input_dim, output_dim)
            self.R = nn.Linear(input_dim, output_dim)

            self.combine = self.__combine_simple

        self.readout = self.__get_readout_fn(readout_type)

    def __get_readout_fn(self, readout_type):
        options = {
            "add": geom_nn.global_add_pool,
            "mean": geom_nn.global_mean_pool,
            "max": geom_nn.global_max_pool
        }
        if readout_type not in options:
            raise ValueError()
        return options[readout_type]

    def forward(self, h, edge_index, batch):

        # this give a (batch_size, features) tensor
        readout = self.readout(x=h, batch=batch)
        # this give a (nodes, features) tensor
        readout = readout[batch]

        return self.propagate(
            edge_index=edge_index,
            h=h,
            readout=readout)

    def message(self, h_j):
        return h_j

    def __combine_mlp(self, aggr, h, readout):
        return self.mlp1(h) + self.mlp2(aggr) + self.mlp3(readout)

    def __combine_simple(self, aggr, h, readout):
        return self.V(h) + self.A(aggr) + self.R(readout)

    def update(self, aggr, h, readout):
        return self.combine(aggr=aggr, h=h, readout=readout)


class ACConv(MessagePassing):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            aggregate_type: str,
            combine_type: str,
            num_mlp_layers: int,
            **kwargs):

        assert aggregate_type in ["add", "mean", "max"]
        assert combine_type in ["simple", "mlp"]

        if combine_type == "mlp":
            raise NotImplementedError()

        super(ACConv, self).__init__(aggr=aggregate_type, **kwargs)

        self.V = nn.Linear(input_dim, output_dim)
        self.A = nn.Linear(input_dim, output_dim)

    def forward(self, h, edge_index, batch):
        return self.propagate(
            edge_index=edge_index,
            h=h)

    def message(self, h_j):
        return h_j

    def update(self, aggr, h):
        return self.V(h) + self.A(aggr)
