
import os
import os.path as osp
import random

import numpy as np
import torch
import torch.nn.functional as F
from sklearn import metrics
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.data import DataLoader
from torch_geometric.datasets import PPI

from gnn import *
from utils.early_stopping import EarlyStopping


def node_test(x, y, multi_label=False):
    if multi_label:
        micro_f1 = metrics.f1_score(
            y.cpu().detach().numpy(),
            (x > 0).cpu().detach().numpy(),
            average='micro')
        node_acc_count = micro_f1 * len(x)
    else:
        y = y.cpu()
        pred = torch.argmax(F.softmax(x, dim=1), dim=1).cpu()
        node_acc_count = metrics.accuracy_score(y,
                                                pred,
                                                normalize=False)

    return node_acc_count


def train(
        model,
        optimizer,
        loader,
        device,
        criterion,
        node_multi_label=True,
        mode="train"):

    if mode == "train":
        model.train()
    else:
        model.eval()

    total_loss = 0
    total_node = 0
    node_acc_count = 0
    data_count = 0

    for data in loader:
        data_count += data.num_graphs
        num_graphs = data.num_graphs

        data = data.to(device)

        optimizer.zero_grad()

        if mode == "train":
            logits = model(x=data.x,
                           edge_index=data.edge_index,
                           batch=data.batch)
        else:
            with torch.no_grad():
                logits = model(x=data.x,
                               edge_index=data.edge_index,
                               batch=data.batch)

        loss = criterion(logits, data.y)

        node_acc_count += node_test(logits,
                                    data.y,
                                    node_multi_label)
        total_node += len(logits)

        total_loss += loss.item() * num_graphs

        if mode == "train":
            loss.backward()
            optimizer.step()

    node_acc = float(node_acc_count) / total_node
    return total_loss / data_count, node_acc


def load_data(path):
    train_dataset = PPI(path, split='train')
    val_dataset = PPI(path, split='val')
    test_dataset = PPI(path, split='test')
    train_loader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0)
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0)
    test_loader = DataLoader(
        test_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=0)

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader


def trainer(
        model,
        logger,
        summary_file,
        train_loader,
        val_loader,
        test_loader,
        device,
        criterion,
        max_epoch=200,
        early_stopping=None):

    lr = 2e-4

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=0)
    scheduler = ReduceLROnPlateau(
        optimizer,
        "min",
        patience=100,
        verbose=True,
        factor=0.5,
        cooldown=30,
        min_lr=lr / 100)

    best_val_loss = float('inf')

    for epoch in range(0, max_epoch):
        train_loss, train_node_acc = train(
            model=model, optimizer=optimizer, loader=train_loader, device=device, mode="train", criterion=criterion)

        val_loss, val_node_acc = train(
            model=model, optimizer=optimizer, loader=val_loader, device=device, mode="val", criterion=criterion)

        test_loss, test_node_acc = train(
            model=model, optimizer=optimizer, loader=test_loader, device=device, mode="test", criterion=criterion)

        logger.write(
            f"{train_loss},{val_loss},{test_loss},{train_node_acc},{val_node_acc},{test_node_acc}\n")

        print(
            f"Epoch: {epoch}/{max_epoch}\nTrain:\t{train_loss}\t{train_node_acc}\nVal:\t{val_loss}\t{val_node_acc}\nTest:\t{test_loss}\t{test_node_acc}")

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping")
            break

        scheduler.step(train_loss)

    with open(summary_file, "a") as f:
        f.write(
            f"{train_loss},{val_loss},{test_loss},{train_node_acc},{val_node_acc},{test_node_acc},")

    return train_node_acc, val_node_acc, test_node_acc


def run_std(runs, file_name, **kwargs):
    train_accs, val_accs, test_accs = [], [], []
    for i in range(runs):
        kwargs["model"].reset_parameters()

        es = EarlyStopping(
            patience=20)

        train_node_acc, val_node_acc, test_node_acc = trainer(
            early_stopping=es, **kwargs)

        train_accs.append(train_node_acc)
        val_accs.append(val_node_acc)
        test_accs.append(test_node_acc)

    with open(file_name, "w") as std_file:
        std_file.write(f"{np.mean(train_accs)}, {np.std(train_accs)}\n")
        std_file.write(f"{np.mean(val_accs)}, {np.std(val_accs)}\n")
        std_file.write(f"{np.mean(test_accs)}, {np.std(test_accs)}\n")


def seed_everything(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    h = 256

    _networks = [
        # [{"mean": "A"}, {"mean": "A"}, {"simple": "T"}],
        # [{"mean": "A"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"mean": "A"}, {"max": "M"}, {"simple": "T"}],
        # [{"mean": "A"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"mean": "A"}, {"add": "S"}, {"simple": "T"}],
        # [{"mean": "A"}, {"add": "S"}, {"mlp": "MLP"}],

        [{"max": "M"}, {"mean": "A"}, {"simple": "T"}],
        # [{"max": "M"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"max": "M"}, {"max": "M"}, {"simple": "T"}],
        # [{"max": "M"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"max": "M"}, {"add": "S"}, {"simple": "T"}],
        # [{"max": "M"}, {"add": "S"}, {"mlp": "MLP"}],

        # [{"add": "S"}, {"mean": "A"}, {"simple": "T"}],
        # [{"add": "S"}, {"mean": "A"}, {"mlp": "MLP"}],
        # [{"add": "S"}, {"max": "M"}, {"simple": "T"}],
        # [{"add": "S"}, {"max": "M"}, {"mlp": "MLP"}],
        # [{"add": "S"}, {"add": "S"}, {"simple": "T"}],
        # [{"add": "S"}, {"add": "S"}, {"mlp": "MLP"}],
    ]

    file_path = "."
    extra_name = "delete"

    for _net_class in [
        # "acgnn",
        # "gin",
        "acrgnn",
        # "acrgnn-single"
    ]:

        filename = f"{file_path}/logging/{extra_name}/ppi.mix"

        for a, r, c in _networks:
            (_agg, _agg_abr) = list(a.items())[0]
            (_read, _read_abr) = list(r.items())[0]
            (_comb, _comb_abr) = list(c.items())[0]

            for comb_layers in [1]:

                if _net_class == "acgnn" and (
                        _read == "max" or _read == "add"):
                    continue
                elif _net_class == "gin" and (_agg == "mean" or _agg == "max" or _comb == "mlp" or _read == "max" or _read == "add"):
                    continue

                if _comb == "mlp" and comb_layers > 1:
                    continue

                for l in [2]:

                    print(a, r, c, _net_class, l, comb_layers)

                    _log_file = f"{file_path}/logging/{extra_name}/ppi-{_net_class}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-cl{comb_layers}-L{l}-h{h}.log"

                    with open(_log_file, "w") as log_file:
                        log_file.write(
                            "train_loss,val_loss,test_loss,train_acc,val_acc,test_acc\n")

                        train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = load_data(
                            f"{file_path}/data/{extra_name}")

                        if _net_class == "acgnn":
                            _model = ACGNN
                        elif _net_class == "acrgnn":
                            _model = ACRGNN
                        elif _net_class == "acrgnn-single":
                            _model = SingleACRGNN
                        elif _net_class == "gin":
                            _model = GIN

                        seed_everything(0)

                        if torch.cuda.is_available():
                            device = torch.device("cuda:0")
                        else:
                            device = torch.device("cpu")

                        model = _model(
                            input_dim=train_dataset.num_features,
                            hidden_dim=h,
                            output_dim=train_dataset.num_classes,
                            num_layers=l,
                            aggregate_type=_agg,
                            readout_type=_read,
                            combine_type=_comb,
                            combine_layers=comb_layers,
                            num_mlp_layers=2,
                            task="node",
                            truncated_fn=None)

                        model = model.to(device)

                        # trainer(
                        #     model=model,
                        #     logger=log_file,
                        #     summary_file=filename,
                        #     train_loader=train_loader,
                        #     val_loader=val_loader,
                        #     test_loader=test_loader,
                        #     device=device,
                        #     criterion=torch.nn.BCEWithLogitsLoss(),
                        #     max_epoch=500,
                        #     early_stopping=es)

                        run_std(runs=10,
                                file_name="logging/ppi/ACR-M-A-SIMPLE-1-2",
                                model=model,
                                logger=log_file,
                                summary_file=filename,
                                train_loader=train_loader,
                                val_loader=val_loader,
                                test_loader=test_loader,
                                device=device,
                                criterion=torch.nn.BCEWithLogitsLoss(),
                                max_epoch=500)

                with open(filename, "a") as f:
                    f.write("\n")
