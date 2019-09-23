import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch_geometric.data import Batch

from datasets.argparser_real_data import argument_parser
from datasets.datasets import get_planetoid_dataset, random_planetoid_splits
from gnn import ACRGNN
from gnn.utils import reset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def run(filename,
        dataset,
        model,
        runs,
        epochs,
        lr,
        weight_decay,
        early_stopping,
        permute_masks=None,
        logger=None):

    val_losses, accs, durations = [], [], []
    for i in range(runs):
        print(f"Run {i+1}/{runs}")

        data = dataset[0]
        if permute_masks is not None:
            data = permute_masks(data, dataset.num_classes)
        data = Batch.from_data_list([data])
        data = data.to(device)

        model = model.to(device)
        reset(model)

        optimizer = Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_start = time.perf_counter()

        best_val_loss = float('inf')
        test_acc = 0
        val_loss_history = []

        for epoch in range(1, epochs + 1):
            train(model, optimizer, data)
            eval_info = evaluate(model, data)
            eval_info['epoch'] = epoch

            if logger is not None:
                logger.write(f"{eval_info}\n")

            if eval_info['val_loss'] < best_val_loss:
                best_val_loss = eval_info['val_loss']
                test_acc = eval_info['test_acc']

            val_loss_history.append(eval_info['val_loss'])
            if early_stopping > 0 and epoch > epochs // 2:
                tmp = torch.tensor(val_loss_history[-(early_stopping + 1):-1])
                if eval_info['val_loss'] > tmp.mean().item():
                    break

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        t_end = time.perf_counter()

        val_losses.append(best_val_loss)
        accs.append(test_acc)
        durations.append(t_end - t_start)

    loss, acc, duration = torch.tensor(
        val_losses), torch.tensor(accs), torch.tensor(durations)

    print('Val Loss: {:.4f}, Test Accuracy: {:.3f} ± {:.3f}, Duration: {:.3f}'.
          format(loss.mean().item(),
                 acc.mean().item(),
                 acc.std().item(),
                 duration.mean().item()))
    with open(filename, 'a') as f:
        f.write(
            f"{loss.mean().item():.8f}, {acc.mean().item():.8f}, {acc.std().item():.8f}, {duration.mean().item():.8f}\n")


def train(model, optimizer, data):
    model.train()
    optimizer.zero_grad()
    out = F.log_softmax(model(x=data.x,
                              edge_index=data.edge_index,
                              batch=data.batch), dim=1)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()


def evaluate(model, data):
    model.eval()

    with torch.no_grad():
        logits = F.log_softmax(model(x=data.x,
                                     edge_index=data.edge_index,
                                     batch=data.batch), dim=1)

    outs = {}
    for key in ['train', 'val', 'test']:
        mask = data['{}_mask'.format(key)]
        loss = F.nll_loss(logits[mask], data.y[mask]).item()
        pred = logits[mask].max(1)[1]
        acc = pred.eq(data.y[mask]).sum().item() / mask.sum().item()

        outs['{}_loss'.format(key)] = loss
        outs['{}_acc'.format(key)] = acc

    return outs


def get_model(args,
              data):
    return ACRGNN(
        num_layers=args.num_layers,
        num_mlp_layers=args.num_mlp_layers,
        input_dim=data.num_features,
        hidden_dim=args.hidden_dim,
        output_dim=data.num_classes,
        combine_type=args.combine,
        aggregate_type=args.aggregate,
        readout_type=args.readout,
        task=args.task_type)


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
    _dataset = "Cora"
    _network = "acrgnn"

    _networks = [
        [{"mean": "A"}, {"mean": "A"}, {"simple": "T"}],
        [{"mean": "A"}, {"mean": "A"}, {"mlp": "MLP"}],
        [{"mean": "A"}, {"max": "M"}, {"simple": "T"}],
        [{"mean": "A"}, {"max": "M"}, {"mlp": "MLP"}],
        [{"mean": "A"}, {"add": "S"}, {"simple": "T"}],
        [{"mean": "A"}, {"add": "S"}, {"mlp": "MLP"}],

        [{"max": "M"}, {"mean": "A"}, {"simple": "T"}],
        [{"max": "M"}, {"mean": "A"}, {"mlp": "MLP"}],
        [{"max": "M"}, {"max": "M"}, {"simple": "T"}],
        [{"max": "M"}, {"max": "M"}, {"mlp": "MLP"}],
        [{"max": "M"}, {"add": "S"}, {"simple": "T"}],
        [{"max": "M"}, {"add": "S"}, {"mlp": "MLP"}],

        [{"add": "S"}, {"mean": "A"}, {"simple": "T"}],
        [{"add": "S"}, {"mean": "A"}, {"mlp": "MLP"}],
        [{"add": "S"}, {"max": "M"}, {"simple": "T"}],
        [{"add": "S"}, {"max": "M"}, {"mlp": "MLP"}],
        [{"add": "S"}, {"add": "S"}, {"simple": "T"}],
        [{"add": "S"}, {"add": "S"}, {"mlp": "MLP"}],
    ]
    for l in [1, 2, 3, 4]:

        run_filename = f"logging/{_dataset}/real-L{l}.log"

        with open(run_filename, 'w') as f:
            f.write("val_loss,test_acc,test_std,duration\n")

        for a, r, c in _networks:
            (_agg, _agg_abr) = list(a.items())[0]
            (_read, _read_abr) = list(r.items())[0]
            (_comb, _comb_abr) = list(c.items())[0]

            print(a, r, c, _network, l)

            _log_file = f"logging/{_dataset}/real-{_network}-agg{_agg_abr}-read{_read_abr}-comb{_comb_abr}-L{l}.log"
            with open(_log_file, "w") as log_file:

                _args = argument_parser().parse_args(
                    [
                        f"--dataset={_dataset}",
                        "--random_splits=False",
                        "--runs=100",
                        f"--readout={_read}",
                        f"--aggregate={_agg}",
                        f"--combine={_comb}",
                        f"--network={_network}",
                        f"--mlp_combine_agg=add",
                        f"--filename=logging/{run_filename}.log",
                        f"--hidden_dim=16",
                        f"--num_layers={l}"
                    ])

                dataset = get_planetoid_dataset(
                    _args.dataset, _args.normalize_features)
                permute_masks = random_planetoid_splits if _args.random_splits else None

                run(run_filename,
                    dataset,
                    get_model(_args, dataset),
                    _args.runs,
                    _args.epochs,
                    _args.lr,
                    _args.weight_decay,
                    _args.early_stopping,
                    permute_masks,
                    logger=log_file)
