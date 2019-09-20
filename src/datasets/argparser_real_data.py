import argparse


def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--random_splits', type=bool, default=False)
    parser.add_argument('--runs', type=int, default=100)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--early_stopping', type=int, default=10)
    parser.add_argument('--hidden_dim', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--normalize_features', type=bool, default=True)

    parser.add_argument(
        '--num_layers',
        type=int,
        default=5,
        help='number of layers INCLUDING the input one (default: 5)')
    parser.add_argument(
        '--num_mlp_layers',
        type=int,
        default=2,
        help='number of layers for MLP EXCLUDING the input one (default: 2). 1 means linear model.')
    parser.add_argument(
        '--readout',
        type=str,
        default="max",
        choices=[
            "add",
            "mean",
            "max"],
        help='Pooling for over all nodes in a graph: add, mean or max')
    parser.add_argument(
        '--aggregate',
        type=str,
        default="add",
        choices=[
            "add",
            "mean",
            "max"],
        help='Pooling for over neighboring nodes: add, mean or max')
    parser.add_argument(
        '--combine',
        type=str,
        default="simple",
        choices=[
            "simple",
            "mlp"],
        help='Reduction of the aggregation: simple or mlp')
    parser.add_argument(
        '--mlp_combine_agg',
        type=str,
        default="add",
        choices=[
            "add",
            "mean",
            "max",
            "concat"],
        help='Aggregate function to use inside mlp combine')
    parser.add_argument('--filename', type=str, default="training.log",
                        help='output file')
    parser.add_argument(
        '--network',
        type=str,
        default="acrgnn",
        choices=[
            "acrgnn",
            "acgnn",
            "gin"],
        help='Type of GNN to use. a=Aggregate, c=Combine, r=Readout')
    parser.add_argument(
        '--task_type',
        type=str,
        default="node",
        choices=[
            "node",
            "graph"],
        help='Task to solve, `node` or `graph` classification')
    return parser
