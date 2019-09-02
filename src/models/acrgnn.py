import torch

from .gnn import GNN


class ACRGNN(GNN):
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
            recursive_weighting: bool,
            task: str,
            device: torch.device
    ):
        super(ACRGNN, self).__init__(num_layers=num_layers,
                                     num_mlp_layers=num_mlp_layers,
                                     input_dim=input_dim,
                                     hidden_dim=hidden_dim,
                                     output_dim=output_dim,
                                     final_dropout=final_dropout,
                                     combine_type=combine_type,
                                     aggregate_type=aggregate_type,
                                     readout_type=readout_type,
                                     recursive_weighting=recursive_weighting,
                                     task=task,
                                     device=device)

    def compute_layer(self, h, layer, aux_data):
        # pooling neighboring nodes and center nodes altogether
        aggregated = self.aggregate(h=h, aux_data=aux_data)
        readout = self.readout(h=h)
        h = self.combine(x1=h, x2=aggregated, x3=readout, layer=layer)
        return h
