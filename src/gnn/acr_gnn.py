import torch
from gnn.conv_layers import ACRConv
import torch.nn as nn


class ACRGNN(torch.nn.Module):

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            output_dim: int,
            combine_type: str,
            aggregate_type: str,
            readout_type: str,
            num_layers: int,
            num_mlp_layers: int,
            task: str,
            **kwargs
    ):
        super(ACRGNN, self).__init__()

        self.num_layers = num_layers
        self.task = task

        self.padding = nn.ConstantPad1d((0, hidden_dim - input_dim), value=0)

        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(self.num_layers):
            self.convs.append(nn.ACRConv(hidden_dim=hidden_dim,
                                         aggregate_type=aggregate_type,
                                         readout_type=readout_type,
                                         combine_type=combine_type,
                                         num_mlp_layers=num_mlp_layers))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.linear_prediction = nn.Linear(hidden_dim, output_dim)

    def forward(self, x, edge_index, batch):
        h = self.padding(x)
        for layer in range(self.num_layers):
            h = self.convs[layer](h=h, edge_index=edge_index, batch=batch)
            h = torch.relu(h)
            h = self.batch_norms[layer](h)

        if self.task == "node":
            return self.linear_prediction(h)

        else:
            raise NotImplementedError()
