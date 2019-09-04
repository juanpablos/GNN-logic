from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from networkx.classes.function import nodes
from tqdm import tqdm

from models import *
from utils.argparser import argument_parser
from utils.util import load_data, separate_data


def train(
        args,
        model,
        device,
        train_graphs,
        optimizer,
        criterion,
        scheduler,
        epoch) -> float:
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:

        try:
            batch_graph = np.random.choice(
                train_graphs, size=args.batch_size, replace=False)
        except ValueError:
            batch_graph = np.random.choice(
                train_graphs, size=args.batch_size, replace=True)
        # batches_nodes -> all nodes in the batch
        # (sum(n_nodes(graph), classes), for graph in batch
        output = model(batch_graph)

        # get the real node labels (nodes) vector
        # (sum(n_nodes(graph)), for graph in batch
        labels = []
        for graph in batch_graph:
            labels.extend(graph.node_labels)
        labels = torch.tensor(
            labels, dtype=torch.long).unsqueeze(
            dim=1).to(device)
        labels = torch.zeros_like(output).scatter_(1, labels, 1.).to(device)

        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            loss.backward()
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        loss_accum += loss.detach().cpu().numpy()

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


# pass data to model with minibatch during testing to avoid memory
# overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=64):
    def chunks(iterable, n):
        for i in range(0, len(iterable), n):
            yield iterable[i:i + n]

    model.eval()
    output = []
    n_nodes = []
    labels = []
    for graph_batch in chunks(graphs, minibatch_size):
        output.append(model(graph_batch).detach())
        for graph in graph_batch:
            n_nodes.append(len(graph.graph))
            labels.extend(graph.node_labels)

    return torch.cat(output, dim=0), n_nodes, np.cumsum(n_nodes), labels


def test(args, model, device, train_graphs, test_graphs, epoch, run_test):
    model.eval()

    # --- train
    output, n_nodes, indices, labels = pass_data_iteratively(
        model, train_graphs)
    output = torch.sigmoid(output)
    _, predicted_labels = output.max(dim=1)

    # equals both vectors, prediction == label
    results = np.equal(predicted_labels.cpu(), labels).numpy()

    # micro average -> mean between all nodes
    train_micro_avg = np.mean(results)

    # split node equality by graph, we dont need the last value of `indices`
    macro_split = np.split(results, indices[:-1])
    # macro average -> mean between the mean of nodes for each graph
    train_macro_avg = np.mean([np.mean(graph) for graph in macro_split])

    # * Debug
    # with open("debug.txt", 'w') as f:
    #     f.write("PREDICTED\n")
    #     for i in range(len(predicted_labels)):
    #         f.write(f"{str(predicted_labels[i])}\n{str(labels[i])}")
    #         f.write("\n\n")
    # import sys
    # sys.exit()

    # --- test
    test_micro_avg, test_macro_avg = -1, -1
    if not args.no_test:
        if run_test:
            output, n_nodes, indices, labels = pass_data_iteratively(
                model, test_graphs)
            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            # equals both vectors, prediction == label
            results = np.equal(predicted_labels.cpu(), labels).numpy()

            # micro average -> mean between all nodes
            test_micro_avg = np.mean(results)

            # split node equality by graph, we dont need the last value of
            # `indices`
            macro_split = np.split(results, indices[:-1])
            # macro average -> mean between the mean of nodes for each graph
            test_macro_avg = np.mean([np.mean(graph) for graph in macro_split])

    print(
        f"Train accuracy: micro: {train_micro_avg}\tmacro: {train_macro_avg}")
    print(f"Test accuracy: micro: {test_micro_avg}\tmacro: {test_macro_avg}")

    return train_micro_avg, train_macro_avg, test_micro_avg, test_macro_avg


def main(args, data_train=None, data_test=None, n_classes=None):
    # set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    if data_train is None:
        # list of graphs, (number of label classes for the graph, number of
        # feature classes for nodes, number of label classes for the nodes)
        train_graphs, (n_graph_classes, n_node_features, n_node_labels) = load_data(
            dataset=args.data_train, degree_as_node_label=args.degree_as_label)
    else:
        print("Using preloaded data")
        train_graphs = data_train

    if args.task_type == "node":
        if n_classes is None:
            num_classes = n_node_labels
        else:
            print("Using preloaded data")
            num_classes = n_classes
    else:
        raise NotImplementedError()

    if data_test is None:
        if args.data_test is not None:
            test_graphs, _ = load_data(
                dataset=args.data_test, degree_as_node_label=args.degree_as_label)
        else:
            if args.no_test:
                train_graphs, test_graphs = train_graphs, None
            else:
                train_graphs, test_graphs = separate_data(
                    train_graphs, args.seed)
    else:
        print("Using preloaded data")
        test_graphs = data_test

    if args.network == "acgnn":
        _model = ACGNN
    elif args.network == "acrgnn":
        _model = ACRGNN
    elif args.network == "gin":
        _model = GIN
    else:
        raise ValueError()

    model = _model(
        num_layers=args.num_layers,
        num_mlp_layers=args.num_mlp_layers,
        input_dim=train_graphs[0].node_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        final_dropout=args.final_dropout,
        combine_type=args.combine,
        aggregate_type=args.aggregate,
        readout_type=args.readout,
        mlp_aggregate=args.mlp_combine_agg,
        recursive_weighting=args.recursive_weighting,
        task=args.task_type,
        device=device).to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if not args.filename == "":
        with open(args.filename, 'w') as f:
            f.write("Epoch,Loss,train_micro,train_macro,test_micro,test_macro\n")

    # `epoch` is only for printing purposes
    for epoch in range(1, args.epochs + 1):

        avg_loss = train(
            args=args,
            model=model,
            device=device,
            train_graphs=train_graphs,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            epoch=epoch)

        # run the test set only every 20 epochs
        if epoch % args.test_every == 0:
            train_micro, train_macro, test_micro, test_macro = test(
                args=args, model=model, device=device, train_graphs=train_graphs, test_graphs=test_graphs, epoch=epoch, run_test=True)
        else:
            train_micro, train_macro, test_micro, test_macro = test(
                args=args, model=model, device=device, train_graphs=train_graphs, test_graphs=test_graphs, epoch=epoch, run_test=False)

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write(
                    f"{epoch},{avg_loss:.4f},{train_micro:.4f},{train_macro:.4f},{test_micro:.4f},{test_macro:.4f}\n")


if __name__ == '__main__':
    combine = {"trainable": "T", "mlp": "MLP"}
    aggregation = {"sum": "S", "average": "A", "max": "M"}
    # aggregation = {"max": "M"}
    readout = {"sum": "S", "average": "A", "max": "M"}
    mlp_agg = {"sum": "S"}  # , "average": "A", "max": "M", "concat": "C"}
    network = {"acrgnn"}
    datasets = ["T3"]
    hidden = [64]
    batch = [32]

    print("Start running")
    for dataset in datasets:
        print(f"Start for dataset {dataset}")

        train_graphs, (_, _, n_node_labels) = load_data(
            dataset=f"utils/{dataset}-1.txt", degree_as_node_label=False)

        test_graphs, _ = load_data(
            dataset=f"utils/{dataset}-2.txt", degree_as_node_label=False)

        for net in network:
            for agg, agg_abr in aggregation.items():
                for read, read_abr in readout.items():
                    for comb, comb_abr in combine.items():
                        for h in hidden:
                            for b in batch:
                                if comb == "mlp":
                                    for agg_mlp, agg_mlp_abr in mlp_agg.items():
                                        print(
                                            f"Running for {net}-{agg}-{read}-{comb}-{h}-{agg_mlp}-{b} in {dataset}")
                                        args = argument_parser().parse_args(
                                            [
                                                f"--readout={read}",
                                                f"--aggregate={agg}",
                                                f"--combine={comb}",
                                                f"--network={net}",
                                                f"--mlp_combine_agg={agg_mlp}",
                                                f"--filename={dataset}-{net}-agg{agg_abr}-read{read_abr}-comb{comb_abr}-h{h}-mlp{agg_mlp_abr}-b{b}.log",
                                                "--epochs=100",
                                                # "--no_test",
                                                f"--batch_size={b}",
                                                "--test_every=1",
                                                f"--hidden_dim={h}"
                                            ])
                                        main(
                                            args,
                                            data_train=train_graphs,
                                            data_test=test_graphs,
                                            n_classes=n_node_labels)
                                else:
                                    print(
                                        f"Running for {net}-{agg}-{read}-{comb}-{h}-{b} in {dataset}")
                                    args = argument_parser().parse_args(
                                        [
                                            f"--readout={read}",
                                            f"--aggregate={agg}",
                                            f"--combine={comb}",
                                            f"--network={net}",
                                            f"--mlp_combine_agg=sum",
                                            f"--filename={dataset}-{net}-agg{agg_abr}-read{read_abr}-comb{comb_abr}-h{h}-mlpX-b{b}.log",
                                            "--epochs=100",
                                            # "--no_test",
                                            f"--batch_size={b}",
                                            "--test_every=1",
                                            f"--hidden_dim={h}"
                                        ])
                                    main(
                                        args,
                                        data_train=train_graphs,
                                        data_test=test_graphs,
                                        n_classes=n_node_labels)
