import logging
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from models import *
from utils.argparser import argument_parser
from utils.graphs import online_generator
from utils.util import load_data, separate_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    filename="logging/logger.log")


def __batch_generator(configuration, batch_size):
    ...


def train(
        args,
        model,
        device,
        train_graphs,
        optimizer,
        criterion,
        scheduler,
        epoch,
        online=None) -> float:
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:

        # batch_graph = train_graphs
        if online is None:
            try:
                batch_graph = np.random.choice(
                    train_graphs, size=args.batch_size, replace=False)
            except ValueError:
                batch_graph = np.random.choice(
                    train_graphs, size=args.batch_size, replace=True)
        else:
            batch_graph = __batch_generator(online, args.batch_size)

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
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

        loss_accum += loss.detach().cpu().numpy()

        # report
        pbar.set_description('epoch: %d' % (epoch))

    average_loss = loss_accum / total_iters
    print("loss training: %f" % (average_loss))

    return average_loss


# pass data to model with minibatch during testing to avoid memory
# overflow (does not perform backpropagation)
def pass_data_iteratively(model, graphs, minibatch_size=128):
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


def test(
        args,
        model,
        device,
        train_graphs,
        test_graphs,
        epoch,
        run_test,
        criterion,
        another_test=None):
    model.eval()

    with torch.no_grad():

        # --- train
        output, n_nodes, indices, labels = pass_data_iteratively(
            model, train_graphs)
        output = torch.sigmoid(output)
        _, predicted_labels = output.max(dim=1)

        ######

        pred_zeros = np.sum(predicted_labels.cpu().numpy() == 0)
        pred_ones = np.sum(predicted_labels.cpu().numpy() == 1)
        real_zeros = np.sum(np.array(labels) == 0)
        real_ones = np.sum(np.array(labels) == 1)
        logging.info(f"Epoch {epoch} - Train")
        logging.info(f"Predicted 0s: {pred_zeros}/{real_zeros}")
        logging.info(f"Predicted 1s: {pred_ones}/{real_ones}")
        ######

        # equals both vectors, prediction == label
        results = np.equal(predicted_labels.cpu(), labels).numpy()

        # micro average -> mean between all nodes
        train_micro_avg = np.mean(results)

        # split node equality by graph, we dont need the last value of
        # `indices`
        macro_split = np.split(results, indices[:-1])
        # macro average -> mean between the mean of nodes for each graph
        train_macro_avg = np.mean([np.mean(graph) for graph in macro_split])

        # --- test
        test_micro_avg, test_macro_avg = -1, -1
        if not args.no_test:
            if run_test:

                output, n_nodes, indices, labels = pass_data_iteratively(
                    model, test_graphs)
                output = torch.sigmoid(output)
                _, predicted_labels = output.max(dim=1)

                # test loss
                _labels = torch.tensor(
                    labels, dtype=torch.long).unsqueeze(dim=1).to(device)
                _labels = torch.zeros_like(output).scatter_(
                    1, _labels, 1.).to(device)

                test1_loss = criterion(output, _labels).detach().cpu().numpy()

                # equals both vectors, prediction == label
                results = np.equal(predicted_labels.cpu(), labels).numpy()

                # micro average -> mean between all nodes
                test_micro_avg = np.mean(results)

                ######
                pred_zeros = np.sum(predicted_labels.cpu().numpy() == 0)
                pred_ones = np.sum(predicted_labels.cpu().numpy() == 1)
                real_zeros = np.sum(np.array(labels) == 0)
                real_ones = np.sum(np.array(labels) == 1)

                logging.info(f"Epoch {epoch} - Test1")
                logging.info(f"Predicted 0s: {pred_zeros}/{real_zeros}")
                logging.info(f"Predicted 1s: {pred_ones}/{real_ones}")
                ######

                # split node equality by graph, we dont need the last value of
                # `indices`
                macro_split = np.split(results, indices[:-1])
                # macro average -> mean between the mean of nodes for each
                # graph
                test_macro_avg = np.mean([np.mean(graph)
                                          for graph in macro_split])

                if another_test is not None:
                    output, n_nodes, indices, labels = pass_data_iteratively(
                        model, another_test)
                    output = torch.sigmoid(output)
                    _, predicted_labels = output.max(dim=1)

                    # test loss
                    _labels = torch.tensor(
                        labels, dtype=torch.long).unsqueeze(
                        dim=1).to(device)
                    _labels = torch.zeros_like(output).scatter_(
                        1, _labels, 1.).to(device)

                    test2_loss = criterion(
                        output, _labels).detach().cpu().numpy()

                    # equals both vectors, prediction == label
                    results = np.equal(predicted_labels.cpu(), labels).numpy()

                    ######
                    pred_zeros = np.sum(predicted_labels.cpu().numpy() == 0)
                    pred_ones = np.sum(predicted_labels.cpu().numpy() == 1)
                    real_zeros = np.sum(np.array(labels) == 0)
                    real_ones = np.sum(np.array(labels) == 1)

                    logging.info(f"Epoch {epoch} - Test2")
                    logging.info(f"Predicted 0s: {pred_zeros}/{real_zeros}")
                    logging.info(f"Predicted 1s: {pred_ones}/{real_ones}")
                    ######

                    # micro average -> mean between all nodes
                    test_another_micro_avg = np.mean(results)

                    # split node equality by graph, we dont need the last value of
                    # `indices`
                    macro_split = np.split(results, indices[:-1])
                    # macro average -> mean between the mean of nodes for each
                    # graph
                    test_another_macro_avg = np.mean(
                        [np.mean(graph) for graph in macro_split])

    print(f"Test1 loss: {test1_loss}")
    print(f"Test2 loss: {test2_loss}")
    print(
        f"Train accuracy: micro: {train_micro_avg}\tmacro: {train_macro_avg}")
    print(f"Test accuracy: micro: {test_micro_avg}\tmacro: {test_macro_avg}")
    if another_test is not None:
        print(
            f"Test accuracy: micro: {test_another_micro_avg}\tmacro: {test_another_macro_avg}")

        return train_micro_avg, train_macro_avg, test_micro_avg, test_macro_avg, test_another_micro_avg, test_another_macro_avg, test1_loss, test2_loss

    return train_micro_avg, train_macro_avg, test_micro_avg, test_macro_avg, - \
        1, -1, test1_loss, -1


def main(
        args,
        data_train=None,
        data_test=None,
        n_classes=None,
        another_test=None,
        save_model=None,
        load_model=None,
        train_model=True):
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
    elif args.network == "gingnn":
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
        device=device)

    if load_model is not None:
        print("Loading Model")
        model.load_state_dict(torch.load(load_model))

    model = model.to(device)

    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if not args.filename == "":
        with open(args.filename, 'w') as f:
            f.write(
                "train_loss,test1_loss,test2_loss,train_micro,train_macro,test1_micro,test1_macro,test2_micro,test2_macro\n")

    if train_model:
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
                train_micro, train_macro, test_micro, test_macro, test_micro2, test_macro2, test1_loss, test2_loss = test(
                    args=args, model=model, device=device, train_graphs=train_graphs, test_graphs=test_graphs, epoch=epoch, run_test=True, another_test=another_test, criterion=criterion)
            else:
                train_micro, train_macro, test_micro, test_macro, test_micro2, test_macro2, test1_loss, test2_loss = test(
                    args=args, model=model, device=device, train_graphs=train_graphs, test_graphs=test_graphs, epoch=epoch, run_test=False, another_test=another_test, criterion=criterion)

            if not args.filename == "":
                with open(args.filename, 'a') as f:
                    f.write(
                        f"{avg_loss:.10f},{test1_loss:.10f},{test2_loss:.10f},{train_micro:.8f},{train_macro:.8f},{test_micro:.8f},{test_macro:.8f},{test_micro2:.8f},{test_macro2:.8f}\n")

        if save_model is not None:
            torch.save(model.state_dict(), save_model)

        return f"{avg_loss:.10f},{test1_loss:.10f},{test2_loss:.10f},{train_micro:.8f},{train_macro:.8f},{test_micro:.8f},{test_macro:.8f},{test_micro2:.8f},{test_macro2:.8f},"

    else:

        train_micro, train_macro, test_micro, test_macro, test_micro2, test_macro2, test1_loss, test2_loss = test(
            args=args, model=model, device=device, train_graphs=train_graphs, test_graphs=test_graphs, epoch=-1, run_test=True, another_test=another_test, criterion=criterion)

        with open(args.filename, 'a') as f:
            f.write(
                f"{test1_loss:.10f},{test2_loss:.10f},{train_micro:.8f},{train_macro:.8f},{test_micro:.8f},{test_macro:.8f},{test_micro2:.8f},{test_macro2:.8f}\n")

        return f"{test1_loss:.10f},{test2_loss:.10f},{train_micro:.8f},{train_macro:.8f},{test_micro:.8f},{test_macro:.8f},{test_micro2:.8f},{test_macro2:.8f},"


if __name__ == '__main__':

    # agg, read, comb
    _networks = [
        [{"average": "A"}, {"average": "A"}, {"trainable": "T"}],
        [{"average": "A"}, {"average": "A"}, {"mlp": "MLP"}],
        [{"average": "A"}, {"max": "M"}, {"trainable": "T"}],
        [{"average": "A"}, {"max": "M"}, {"mlp": "MLP"}],
        [{"average": "A"}, {"sum": "S"}, {"trainable": "T"}],
        [{"average": "A"}, {"sum": "S"}, {"mlp": "MLP"}],

        [{"max": "M"}, {"average": "A"}, {"trainable": "T"}],
        [{"max": "M"}, {"average": "A"}, {"mlp": "MLP"}],
        [{"max": "M"}, {"max": "M"}, {"trainable": "T"}],
        [{"max": "M"}, {"max": "M"}, {"mlp": "MLP"}],
        [{"max": "M"}, {"sum": "S"}, {"trainable": "T"}],
        [{"max": "M"}, {"sum": "S"}, {"mlp": "MLP"}],

        [{"sum": "S"}, {"average": "A"}, {"trainable": "T"}],
        [{"sum": "S"}, {"average": "A"}, {"mlp": "MLP"}],
        [{"sum": "S"}, {"max": "M"}, {"trainable": "T"}],
        [{"sum": "S"}, {"max": "M"}, {"mlp": "MLP"}],
        [{"sum": "S"}, {"sum": "S"}, {"trainable": "T"}],
        [{"sum": "S"}, {"sum": "S"}, {"mlp": "MLP"}],

        # [{"0": "0"}, {"average": "A"}, {"trainable": "T"}],
        # # [{"0": "0"}, {"average": "A"}, {"mlp": "MLP"}],
        # [{"0": "0"}, {"max": "M"}, {"trainable": "T"}],
        # # [{"0": "0"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"0": "0"}, {"sum": "S"}, {"trainable": "T"}],
        # # [{"0": "0"}, {"sum": "S"}, {"mlp": "MLP"}],
    ]

    print("Start running")
    for _key in ["cycle"]:
        for _enum, _set in enumerate([

            [("train-cycle-300-50-150",
              "test-cycle-150-50-150",
              "test-cycle-500-20-180"),
             ]

        ]):

            key = _key
            enum = _enum

            if "mix" in _set[0][0]:
                key = "mix"
                enum = 0

            if "line" in _set[0][0]:
                key = "line-special"
                enum = 0

            for index, (_train, _test1, _test2) in enumerate(_set):

                print(f"Start for dataset {_train}-{_test1}-{_test2}")

                _train_graphs, (_, _, _n_node_labels) = load_data(
                    dataset=f"data/{_train}.txt",
                    degree_as_node_label=False)

                _test_graphs, _ = load_data(
                    dataset=f"data/{_test1}.txt",
                    degree_as_node_label=False)

                _test_graphs2, _ = load_data(
                    dataset=f"data/{_test2}.txt",
                    degree_as_node_label=False)
                # _train_graphs, (_, _, _n_node_labels) = load_data(
                #     dataset=f"test.txt",
                #     degree_as_node_label=False)

                for _net_class in [
                    "ac",
                    "gin",
                    "acr"
                ]:
                    filename = f"logging/{key}-{enum}-{index}.mix"
                    for a, r, c in _networks:
                        (_agg, _agg_abr) = list(a.items())[0]
                        (_read, _read_abr) = list(r.items())[0]
                        (_comb, _comb_abr) = list(c.items())[0]

                        if (_net_class == "ac" or _net_class == "gin") and (
                                _read == "max" or _read == "sum"):
                            continue
                        elif _net_class == "gin" and _comb == "mlp":
                            continue
                        elif (_net_class == "ac" or _net_class == "gin") and _agg == "0":
                            continue

                        for l in [2]:

                            print(a, r, c, _net_class, l)
                            logging.info(f"{key}-{_net_class}-{_read_abr}")
                            logging.info(f"{a}, {r}, {c}, {_net_class}")

                            _args = argument_parser().parse_args(
                                [
                                    f"--readout={_read}",
                                    f"--aggregate={_agg}",
                                    f"--combine={_comb}",
                                    f"--network={_net_class}gnn",
                                    f"--mlp_combine_agg=sum",
                                    f"--filename=logging/{key}-{enum}-{index}-{_net_class}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}.log",
                                    "--epochs=20",
                                    "--iters_per_epoch=50",
                                    # "--no_test",
                                    f"--batch_size=32",
                                    "--test_every=1",
                                    f"--hidden_dim=64",
                                    f"--num_layers={l}"
                                ])

                            line = main(
                                _args,
                                data_train=_train_graphs,
                                data_test=_test_graphs,
                                n_classes=_n_node_labels,
                                another_test=_test_graphs2,
                                # save_model=f"saved_models/MODEL-{_net_class}-{key}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}.pth",
                                train_model=True,
                                # load_model=f"saved_models/h32/MODEL-{_net_class}-{key}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}.pth"
                            )

                            # append results per layer
                            with open(filename, 'a') as f:
                                f.write(line)

                        # next combination
                        with open(filename, 'a') as f:
                            f.write("\n")
