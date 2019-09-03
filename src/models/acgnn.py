import torch
import torch.nn.functional as F

from .gnn import GNN


class ACGNN(GNN):
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
            device: torch.device
    ):
        super(ACGNN, self).__init__(num_layers=num_layers,
                                    num_mlp_layers=num_mlp_layers,
                                    input_dim=input_dim,
                                    hidden_dim=hidden_dim,
                                    output_dim=output_dim,
                                    final_dropout=final_dropout,
                                    combine_type=combine_type,
                                    aggregate_type=aggregate_type,
                                    readout_type=readout_type,
                                    mlp_aggregate=mlp_aggregate,
                                    recursive_weighting=recursive_weighting,
                                    task=task,
                                    input_factor=2,
                                    device=device)

    def compute_layer(self, h, layer, aux_data):
        # pooling neighboring nodes and center nodes altogether
        aggregated = self.aggregate(h=h, aux_data=aux_data)
        h = self.combine(x1=h, x2=aggregated, layer=layer)
        return torch.relu(h)
