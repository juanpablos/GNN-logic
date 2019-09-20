
import os
import random
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
from torch_scatter import scatter_mean
from tqdm import tqdm

from gnn import *
from utils.argparser import argument_parser
from utils.util import load_data


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

    loss_accum = []

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

        loss_accum.append(loss.detach().cpu().numpy())

    average_loss = np.mean(loss_accum)

    print(f"Train loss: {average_loss}")

    return average_loss, loss_accum


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
    test1_loss = []
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

            test1_loss.append(loss.detach().cpu().numpy())

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

        test1_avg_loss = np.mean(test1_loss)

        test1_micro_avg = test1_micro_avg / n_nodes
        test1_macro_avg = test1_macro_avg / n_graphs

    # ----- /TEST 1 ------

    # ----- TEST 2 ------
    test2_micro_avg = 0.
    test2_macro_avg = 0.
    test2_loss = []
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

            test2_loss.append(loss.detach().cpu().numpy())

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

        test2_avg_loss = np.mean(test2_loss)

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


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def main(
        args,
        manual,
        train_data=None,
        test1_data=None,
        test2_data=None,
        n_classes=None,
        save_model=None,
        load_model=None,
        train_model=True,
        plot=None):
    # set up seeds and gpu device
    seed_everything(args.seed)

    if torch.cuda.is_available():
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

    # np.random.shuffle(train_graphs)
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

            with open(args.filename + ".train", 'w') as f:
                f.write(
                    "train_loss\n")
            with open(args.filename + ".test", 'w') as f:
                f.write(
                    "test1_loss,test2_loss\n")

    if train_model:
        # `epoch` is only for printing purposes
        for epoch in range(1, args.epochs + 1):

            print(f"Epoch {epoch}/{args.epochs}")

            # TODO: binary prediction
            avg_loss, loss_iter = train(
                model=model,
                device=device,
                training_data=train_loader,
                optimizer=optimizer,
                criterion=criterion,
                scheduler=scheduler,
                binary_prediction=True)

            (train_micro, train_macro), (test1_loss, test1_micro, test1_macro), (test2_loss, test2_micro, test2_macro) = test(
                model=model, device=device, train_data=train_loader, test_data1=test1_loader, test_data2=test2_loader, epoch=epoch, criterion=criterion)

            file_line = f"{avg_loss: .10f}, {test1_loss: .10f}, {test2_loss: .10f}, {train_micro: .8f}, {train_macro: .8f}, {test1_micro: .8f}, {test1_macro: .8f}, {test2_micro: .8f}, {test2_macro: .8f}"

            if not args.filename == "":
                with open(args.filename, 'a') as f:
                    f.write(file_line + "\n")

            if not args.filename == "":
                with open(args.filename + ".train", 'a') as f:
                    for l in loss_iter:
                        f.write(f"{l: .15f}\n")

                with open(args.filename + ".test", 'a') as f:
                    f.write(f"{test1_loss: .15f}, {test2_loss: .15f}\n")

        if save_model is not None:
            torch.save(model.state_dict(), save_model)

        if plot is not None:
            iter_losses = np.loadtxt(args.filename + ".train", skiprows=1)
            epoch_t1_losses, epoch_t2_losses = np.loadtxt(
                args.filename + ".test", delimiter=",", skiprows=1).T

            iters = np.arange(len(iter_losses))

            batch = (len(iter_losses) / len(epoch_t1_losses))
            epochs = np.arange(len(epoch_t1_losses)) * batch + batch

            plt.figure(figsize=(16, 10))
            plt.plot(
                iters,
                iter_losses,
                color="#377eb8",
                marker="*",
                linestyle="-",
                label="Train")
            plt.plot(
                epochs,
                epoch_t1_losses,
                color="#ff7f00",
                marker="o",
                linestyle="-",
                label="Test1")
            plt.plot(
                epochs,
                epoch_t2_losses,
                color="#4daf4a",
                marker="x",
                linestyle="-",
                label="Tets2")

            plt.title(
                f"{plot.split('/')[-1].split('.')[0]} - H{args.hidden_dim} - B{args.batch_size} - L{args.num_layers} - Epochs{args.epochs}")

            plt.ylim(bottom=0)
            plt.legend(loc='upper right')
            plt.savefig(plot, dpi=150, bbox_inches='tight')

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
        # [{"mean": "A"}, {"mean": "A"}, {"simple": "T"}],
        # [{"mean": "A"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"mean": "A"}, {"max": "M"}, {"simple": "T"}],
        # [{"mean": "A"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"mean": "A"}, {"add": "S"}, {"simple": "T"}],
        # [{"mean": "A"}, {"add": "S"}, {"mlp": "MLP"}],

        # [{"max": "M"}, {"mean": "A"}, {"simple": "T"}],
        # [{"max": "M"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"max": "M"}, {"max": "M"}, {"simple": "T"}],
        # [{"max": "M"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"max": "M"}, {"add": "S"}, {"simple": "T"}],
        # [{"max": "M"}, {"add": "S"}, {"mlp": "MLP"}],

        [{"add": "S"}, {"mean": "A"}, {"simple": "T"}],
        [{"add": "S"}, {"mean": "A"}, {"mlp": "MLP"}],
        [{"add": "S"}, {"max": "M"}, {"simple": "T"}],
        [{"add": "S"}, {"max": "M"}, {"mlp": "MLP"}],
        [{"add": "S"}, {"add": "S"}, {"simple": "T"}],
        [{"add": "S"}, {"add": "S"}, {"mlp": "MLP"}],
    ]

    h = 64

    print("Start running")
    formula = "formula4"
    for _key in ["nested2"]:
        for _enum, _set in enumerate([
            [(f"{formula}/nested2/train-random-erdos-5000-50-50",
              f"{formula}/nested2/test-random-erdos-500-50-50",
              f"{formula}/nested2/test-random-erdos-500-45-55")
             ],
        ]):

            key = _key
            enum = _enum

            if "mix" in _set[0][0]:
                key = "mix"
                enum = 0

            if "line" in _set[0][0]:
                key = "line-special"
                enum = 0

            if "cycle" in _set[0][0]:
                key = "cycle"
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

                for _net_class in [
                    # "acgnn",
                    # "gin",
                    "acrgnn"
                ]:

                    filename = f"logging/{formula}/{key}-{enum}-{index}.mix"
                    for a, r, c in _networks:
                        (_agg, _agg_abr) = list(a.items())[0]
                        (_read, _read_abr) = list(r.items())[0]
                        (_comb, _comb_abr) = list(c.items())[0]

                        if (_net_class == "acgnn" or _net_class == "gin") and (
                                _read == "max" or _read == "add"):
                            continue
                        elif _net_class == "gin" and _comb == "mlp":
                            continue

                        for l in [1, 2, 3, 4]:

                            print(a, r, c, _net_class, l)

                            run_filename = f"{formula}/{key}-{enum}-{index}-{_net_class}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}"
                            _args = argument_parser().parse_args(
                                [
                                    f"--readout={_read}",
                                    f"--aggregate={_agg}",
                                    f"--combine={_comb}",
                                    f"--network={_net_class}",
                                    f"--mlp_combine_agg=add",
                                    f"--filename=logging/{run_filename}.log",
                                    "--epochs=50",
                                    # "--no_test",
                                    f"--batch_size=128",
                                    "--test_every=1",
                                    f"--hidden_dim={h}",
                                    f"--num_layers={l}"
                                ])

                            line = main(
                                _args,
                                manual=True,
                                train_data=_train_graphs,
                                test1_data=_test_graphs,
                                test2_data=_test_graphs2,
                                n_classes=_n_node_labels,
                                save_model=f"saved_models/{formula}/{key}/MODEL-{_net_class}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}-H{h}.pth",
                                train_model=True,
                                # load_model=f"saved_models/h32/MODEL-{_net_class}-{key}-{enum}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}.pth",
                                plot=f"plots/{run_filename}.png"
                            )

                            # append results per layer
                            with open(filename, 'a') as f:
                                f.write(line)

                        # next combination
                        with open(filename, 'a') as f:
                            f.write("\n")
