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
                                    device=device)

    def compute_layer(self, h, layer, aux_data):
        # pooling neighboring nodes and center nodes altogether
        aggregated = self.aggregate(h=h, aux_data=aux_data)
        h = self.combine(x1=h, x2=aggregated, layer=layer)
        return torch.relu(h)

    def _GNN__functional_combine(self, x1, x2, function="max", **kwargs):
        # x1: node representations, shape (nodes, features)
        # x2: node aggregations, shape (nodes, features)

        # TODO: allow for weighted sum/mean
        combined = torch.cat([x1.unsqueeze(0), x2.unsqueeze(0)])
        if function == "max":
            combined, _ = torch.max(combined, dim=0)
            return combined
        elif function == "sum":
            return torch.sum(combined, dim=0)
        elif function == "average":
            return torch.mean(combined, dim=0)
        else:
            raise ValueError()

    def _GNN__trainable_combine(self, x1, x2, layer, **kwargs):
        # ? + self.b[layer].unsqueeze(dim=0)
        h = self.V[layer](x1) + self.A[layer](x2)
        return h
