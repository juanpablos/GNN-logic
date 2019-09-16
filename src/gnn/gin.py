import torch
import torch.nn as nn
from torch_geometric.nn.conv import GINConv


class GIN(torch.nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            num_layers: int,
            num_mlp_layers: int,
            task: str,
            **kwargs
    ):
        super(GIN, self).__init__()

        self.num_layers = num_layers
        self.task = task

        self.bigger_input = input_dim > hidden_dim

        if not self.bigger_input:
            self.padding = nn.ConstantPad1d(
                (0, hidden_dim - input_dim), value=0)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            if layer == 0 and self.bigger_input:
                _nn = nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))
            else:
                _nn = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, hidden_dim))
            self.convs.append(GINConv(nn=_nn))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linear_prediction = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):

        h = x
        if not self.bigger_input:
            h = self.padding(h)

        for layer in range(self.num_layers):
            h = self.convs[layer](x=h, edge_index=edge_index)
            h = torch.relu(h)
            h = self.batch_norms[layer](h)

        if self.task == "node":
            return self.linear_prediction(h)

        else:
            raise NotImplementedError()
