import argparse


def argument_parser():
    # Training settings
    # Note: Hyper-parameters need to be tuned in order to obtain results
    # reported in the paper.
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str,
                        help='data to read')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='input batch size for training (default: 32)')
    parser.add_argument(
        '--iters_per_epoch',
        type=int,
        default=50,
        help='number of iterations per each epoch (default: 50)')
    parser.add_argument('--epochs', type=int, default=350,
                        help='number of epochs to train (default: 350)')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='learning rate (default: 0.01)')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='random seed (default: 0)')
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
    parser.add_argument('--hidden_dim', type=int, default=64,
                        help='number of hidden units (default: 64)')
    parser.add_argument('--final_dropout', type=float, default=0.5,
                        help='final layer dropout (default: 0.5)')
    parser.add_argument(
        '--readout',
        type=str,
        default="sum",
        choices=[
            "sum",
            "average",
            "max"],
        help='Pooling for over all nodes in a graph: sum or average')
    parser.add_argument(
        '--aggregate',
        type=str,
        default="sum",
        choices=[
            "sum",
            "average",
            "max"],
        help='Pooling for over neighboring nodes: sum, average or max')
    parser.add_argument(
        '--combine',
        type=str,
        default="trainable",
        choices=[
            "sum",
            "average",
            "max",
            "trainable"],
        help='Reduction of the aggregation: sum, average, max or trainable')
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
        '--recursive_weighting',
        action="store_true",
        help='Whether the prediction operation should be performed based on all layers instead of just on the last one.')
    parser.add_argument(
        '--task_type',
        type=str,
        default="node",
        choices=[
            "node",
            "graph"],
        help='Task to solve, `node` or `graph` classification')
    parser.add_argument(
        '--degree_as_label',
        action="store_true",
        help='If there are no labels, use the node degre as label.')
    return parser
