import logging
from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from gnn import *
from utils.argparser import argument_parser
from utils.graphs import online_generator
from utils.util import load_data, separate_data

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.DEBUG,
    format="%(message)s",
    filename="logging/logger.log")


def __loss_aux(output, loss, data, binary_prediction):
    if binary_prediction:
        labels = torch.zeros_like(output).scatter_(
            1, data.node_labels.unsqueeze(1), 1.)
    else:
        raise NotImplementedError()

    return loss(output, labels)


def train(
        model,
        device,
        training_data,
        optimizer,
        criterion,
        scheduler,
        binary_prediction=True) -> float:
    model.train()

    loss_accum = 0.

    for data in tqdm(training_data):
        data = data.to(device)

        output = model(x=data.x,
                       edge_index=data.edge_index,
                       batch=data.batch)

        loss = __loss_aux(
            output=output,
            loss=criterion,
            data=data,
            binary_prediction=binary_prediction)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_accum += loss.detach().cpu().numpy()

    average_loss = loss_accum / len(training_data)

    print(f"Train loss: {average_loss}")

    return average_loss


def __accuracy_aux(node_labels, predicted_labels, batch, device):

    results = torch.eq(
        predicted_labels,
        node_labels).type(
        torch.FloatTensor).to(device)

    # micro average -> mean between all nodes
    micro = torch.sum(results)

    # macro average -> mean between the mean of nodes for each graph
    macro = torch.sum(scatter_mean(results, batch))

    return micro, macro


def test(
        model,
        device,
        criterion,
        epoch,
        train_data,
        test_data1,
        test_data2=None,
        binary_prediction=True):
    model.eval()

    # ----- TRAIN ------
    train_micro_avg = 0.
    train_macro_avg = 0.

    if train_data is not None:
        n_nodes = 0
        n_graphs = 0
        for data in train_data:
            data = data.to(device)

            with torch.no_grad():
                output = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch)

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data.node_labels,
                predicted_labels=predicted_labels,
                batch=data.batch, device=device)

            train_micro_avg += micro.cpu().numpy()
            train_macro_avg += macro.cpu().numpy()
            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        train_micro_avg = train_micro_avg / n_nodes
        train_macro_avg = train_macro_avg / n_graphs

    # ----- /TRAIN ------

    # ----- TEST 1 ------
    test1_micro_avg = 0.
    test1_macro_avg = 0.
    test1_avg_loss = 0.

    if test_data1 is not None:
        n_nodes = 0
        n_graphs = 0
        for data in test_data1:
            data = data.to(device)

            with torch.no_grad():
                output = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch)

            loss = __loss_aux(
                output=output,
                loss=criterion,
                data=data,
                binary_prediction=binary_prediction)

            test1_avg_loss += loss.detach().cpu().numpy()

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data.node_labels,
                predicted_labels=predicted_labels,
                batch=data.batch, device=device)

            test1_micro_avg += micro.cpu().numpy()
            test1_macro_avg += macro.cpu().numpy()
            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        test1_avg_loss = test1_avg_loss / len(test_data1)

        test1_micro_avg = test1_micro_avg / n_nodes
        test1_macro_avg = test1_macro_avg / n_graphs

    # ----- /TEST 1 ------

    # ----- TEST 2 ------
    test2_micro_avg = 0.
    test2_macro_avg = 0.
    test2_avg_loss = 0.

    if test_data2 is not None:
        n_nodes = 0
        n_graphs = 0
        for data in test_data2:
            data = data.to(device)

            with torch.no_grad():
                output = model(
                    x=data.x,
                    edge_index=data.edge_index,
                    batch=data.batch)

            loss = __loss_aux(
                output=output,
                loss=criterion,
                data=data,
                binary_prediction=binary_prediction)

            test2_avg_loss += loss.detach().cpu().numpy()

            output = torch.sigmoid(output)
            _, predicted_labels = output.max(dim=1)

            micro, macro = __accuracy_aux(
                node_labels=data.node_labels,
                predicted_labels=predicted_labels,
                batch=data.batch, device=device)

            test2_micro_avg += micro.cpu().numpy()
            test2_macro_avg += macro.cpu().numpy()
            n_nodes += data.num_nodes
            n_graphs += data.num_graphs

        test2_avg_loss = test2_avg_loss / len(test_data2)

        test2_micro_avg = test2_micro_avg / n_nodes
        test2_macro_avg = test2_macro_avg / n_graphs

        # ----- /TEST 2 ------

    print(
        f"Train accuracy: micro: {train_micro_avg}\tmacro: {train_macro_avg}")
    print(f"Test1 loss: {test1_avg_loss}")
    print(f"Test2 loss: {test2_avg_loss}")
    print(f"Test accuracy: micro: {test1_micro_avg}\tmacro: {test1_macro_avg}")
    print(f"Test accuracy: micro: {test2_micro_avg}\tmacro: {test2_macro_avg}")

    return (train_micro_avg, train_macro_avg), \
        (test1_avg_loss, test1_micro_avg, test1_macro_avg), \
        (test2_avg_loss, test2_micro_avg, test2_macro_avg)


def main(
        args,
        manual,
        train_data=None,
        test1_data=None,
        test2_data=None,
        n_classes=None,
        save_model=None,
        load_model=None,
        train_model=True):
    # set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
        device = torch.device("cuda:" + str(args.device))
    else:
        device = torch.device("cpu")

    if not manual:
        raise NotImplementedError()

    else:
        assert train_data is not None
        assert test1_data is not None
        assert test2_data is not None
        assert n_classes is not None

        # manual settings
        print("Using preloaded data")
        train_graphs = train_data
        test_graphs1 = test1_data
        test_graphs2 = test2_data

        if args.task_type == "node":
            num_classes = n_classes
        else:
            raise NotImplementedError()

    train_loader = DataLoader(
        train_graphs,
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=0)
    test1_loader = DataLoader(
        test_graphs1,
        batch_size=512,
        pin_memory=True,
        num_workers=0)
    test2_loader = DataLoader(
        test_graphs2,
        batch_size=512,
        pin_memory=True,
        num_workers=0)

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
        input_dim=train_graphs[0].num_features,
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

            print(f"Epoch {epoch}/{args.epochs}")

            # TODO: binary prediction
            avg_loss = train(
                model=model,
                device=device,
                training_data=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                binary_prediction=True)

            (train_micro, train_macro), (test1_loss, test1_micro, test1_macro), (test2_loss, test2_micro, test2_macro) = test(
                model=model,
                device=device,
                train_data=train_loader,
                test_data1=test1_loader,
                test_data2=test2_loader,
                epoch=epoch,
                criterion=criterion)

            file_line = f"{avg_loss: .10f}, {test1_loss: .10f}, {test2_loss: .10f}, {train_micro: .8f}, {train_macro: .8f}, {test1_micro: .8f}, {test1_macro: .8f}, {test2_micro: .8f}, {test2_macro: .8f}"

            if not args.filename == "":
                with open(args.filename, 'a') as f:
                    f.write(file_line + "\n")

        if save_model is not None:
            torch.save(model.state_dict(), save_model)

        return file_line + ","

    else:

        (train_micro, train_macro), (test1_loss, test1_micro, test1_macro), (test2_loss, test2_micro, test2_macro) = test(
            model=model, device=device, train_data=train_loader, test_data1=test1_loader, test_data2=test2_loader, epoch=-1, criterion=criterion)

        file_line = f" {-1: .8f}, {test1_loss: .10f}, {test2_loss: .10f}, {train_micro: .8f}, {train_macro: .8f}, {test1_micro: .8f}, {test1_macro: .8f}, {test2_micro: .8f}, {test2_macro: .8f}"

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write(file_line + "\n")

        return file_line + ","


if __name__ == '__main__':

    # agg, read, comb
    _networks = [
        [{"mean": "A"}, {"mean": "A"}, {"simple": "T"}],
        [{"mean": "A"}, {"max": "M"}, {"simple": "T"}],
        [{"mean": "A"}, {"add": "S"}, {"simple": "T"}],

        [{"max": "M"}, {"mean": "A"}, {"simple": "T"}],
        [{"max": "M"}, {"max": "M"}, {"simple": "T"}],
        [{"max": "M"}, {"add": "S"}, {"simple": "T"}],

        [{"add": "S"}, {"mean": "A"}, {"simple": "T"}],
        [{"add": "S"}, {"max": "M"}, {"simple": "T"}],
        [{"add": "S"}, {"add": "S"}, {"simple": "T"}],
    ]

    print("Start running")
    for _key in ["validation"]:
        for _enum, _set in enumerate([

            # [("train-random-5000-50-100-15-0.02",
            #   "test-random-500-50-100-15-0.02",
            #   "test-random-500-100-200-15-0.01"),
            #  ],
            # [("train-random-5000-50-100-1-0.02",
            #   "test-random-500-50-100-1-0.02",
            #   "test-random-500-100-200-1-0.01"),
            #  ],
            # [("train-random-5000-50-100-1.2-0.02",
            #   "test-random-500-50-100-1.2-0.02",
            #   "test-random-500-100-200-1.2-0.01"),
            #  ],
            # [("train-random-5000-50-100-1.5-0.02",
            #   "test-random-500-50-100-1.5-0.02",
            #   "test-random-500-100-200-1.5-0.01"),
            #  ],
            [("train-random-5000-50-100-2-0.02",
              "test-random-500-50-100-2-0.02",
              "test-random-500-100-200-2-0.01"),
             ],
            # [("train-random-20000-50-100-mix-0.02",
            #   "test-random-2000-50-100-mix-0.02",
            #   "test-random-2000-100-200-mix-0.02"),
            #  ],
            # [("train-line-special-5000-50-100",
            #   "test-line-special-500-50-100",
            #   "test-line-special-500-100-200"),
            #  ],
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
                    dataset=f"data/exp/{_train}.txt",
                    degree_as_node_label=False)

                _test_graphs, _ = load_data(
                    dataset=f"data/exp/{_test1}.txt",
                    degree_as_node_label=False)

                _test_graphs2, _ = load_data(
                    dataset=f"data/exp/{_test2}.txt",
                    degree_as_node_label=False)

                for _net_class in [
                    "acgnn",
                    # "gin",
                    # "acrgnn"
                ]:
                    filename = f"logging/delete/{key}-{enum}-{index}.mix"
                    for a, r, c in _networks:
                        (_agg, _agg_abr) = list(a.items())[0]
                        (_read, _read_abr) = list(r.items())[0]
                        (_comb, _comb_abr) = list(c.items())[0]

                        if (_net_class == "acgnn" or _net_class == "gin") and (
                                _read == "max" or _read == "add"):
                            continue
                        elif _net_class == "gin" and _comb == "mlp":
                            continue

                        for l in [5]:

                            print(a, r, c, _net_class, l)
                            logging.info(f"{key}-{_net_class}-{_read_abr}")
                            logging.info(f"{a}, {r}, {c}, {_net_class}")

                            _args = argument_parser().parse_args(
                                [
                                    f"--readout={_read}",
                                    f"--aggregate={_agg}",
                                    f"--combine={_comb}",
                                    f"--network={_net_class}",
                                    f"--mlp_combine_agg=add",
                                    f"--filename=logging/delete/{key}-{enum}-{index}-{_net_class}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}.log",
                                    "--epochs=100",
                                    # "--no_test",
                                    f"--batch_size=32",
                                    "--test_every=1",
                                    f"--hidden_dim=16",
                                    f"--num_layers={l}"
                                ])

                            line = main(
                                _args,
                                manual=True,
                                train_data=_train_graphs,
                                test1_data=_test_graphs,
                                test2_data=_test_graphs2,
                                n_classes=_n_node_labels,
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
