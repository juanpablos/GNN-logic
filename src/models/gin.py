import torch
import torch.nn as nn
import torch.nn.functional as F

from .gnn import GNN
from .mlp import MLP


class GIN(GNN):
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
        super(GIN, self).__init__(num_layers=num_layers,
                                  num_mlp_layers=num_mlp_layers,
                                  input_dim=input_dim,
                                  hidden_dim=hidden_dim,
                                  output_dim=output_dim,
                                  final_dropout=final_dropout,
                                  learn_eps=learn_eps,
                                  combine_type=combine_type,
                                  aggregate_type=aggregate_type,
                                  readout_type=readout_type,
                                  recursive_weighting=recursive_weighting,
                                  task=task,
                                  device=device)
        # List of MLPs
        self.mlps = torch.nn.ModuleList()
        # List of batchnorms applied to the output of MLP
        # (input of the final prediction linear layer)
        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.num_layers - 1):
            if layer == 0:
                self.mlps.append(
                    MLP(num_mlp_layers, input_dim, hidden_dim, hidden_dim))
            else:
                self.mlps.append(
                    MLP(num_mlp_layers, hidden_dim, hidden_dim, hidden_dim))

            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

    def __next_layer(self, h, layer, aux_data):
        # pooling neighboring nodes and center nodes altogether

        aggregated = self.aggregate(h=h, aux_data=aux_data)
        # representation of neighboring and center nodes
        pooled_rep = self.mlps[layer](aggregated)
        h = self.batch_norms[layer](pooled_rep)

        # non-linearity
        h = F.relu(h)
        return h
