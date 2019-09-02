import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from typing import List

from models.acgnn import ACGNN
from utils.argparser import argument_parser
from utils.util import load_data, separate_data


def train(
        args,
        model: nn.Module,
        device: torch.device,
        train_graphs: List[S2VGraph],
        optimizer,
        criterion,
        epoch: int) -> float:
    model.train()

    total_iters = args.iters_per_epoch
    pbar = tqdm(range(total_iters), unit='batch')

    loss_accum = 0
    for pos in pbar:

        batch_graph = np.random.choice(
            train_graphs, size=args.batch_size, replace=False)
        # TODO: output should be (batch_nodes, property)
        # batches_nodes -> all nodes in the batch
        # (sum(n_nodes(graph), classes), for graph in batch
        output = model(batch_graph)

        # get the real node labels (nodes) vector
        # (sum(n_nodes(graph)), for graph in batch
        labels = []
        for graph in batch_graph:
            labels.extend(graph.node_labels)
        labels = torch.tensor(labels, dtype=torch.long).to(device)

        loss = criterion(output, labels)

        # backprop
        if optimizer is not None:
            loss.backward()
            optimizer.step()
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
    model.eval()
    output = []
    idx = np.arange(len(graphs))
    for i in range(0, len(graphs), minibatch_size):
        sampled_idx = idx[i:i + minibatch_size]
        if len(sampled_idx) == 0:
            continue
        output.append(model([graphs[j] for j in sampled_idx]).detach())
    return torch.cat(output, 0)


def test(args, model, device, train_graphs, test_graphs, epoch):
    model.eval()

    output = pass_data_iteratively(model, train_graphs)
    _, predicted_labels = output.max(1, keepdim=True)
    labels = []
    for graph in train_graphs:
        labels.extend(graph.node_labels)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    correct = predicted_labels.eq(
        labels.view_as(predicted_labels)).sum().cpu().item()
    acc_train = correct / float(len(train_graphs))

    # TODO: same
    output = pass_data_iteratively(model, test_graphs)
    _, predicted_labels = output.max(1, keepdim=True)
    labels = []
    for graph in test_graphs:
        labels.extend(graph.node_labels)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    correct = predicted_labels.eq(
        labels.view_as(predicted_labels)).sum().cpu().item()
    acc_test = correct / float(len(test_graphs))

    print("accuracy train: %f test: %f" % (acc_train, acc_test))

    return acc_train, acc_test


def main():
    args = argument_parser().parse_args()

    # set up seeds and gpu device
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda:" + str(args.device)
                          ) if torch.cuda.is_available() else torch.device("cpu")
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # list of graphs, (number of label classes for the graph, number of
    # feature classes for nodes, number of label classes for the nodes)
    graphs, (n_graph_classes, n_node_features, n_node_labels) = load_data(
        args.dataset, args.degree_as_tag)

    if args.task_type == "node":
        num_classes = n_node_labels
    else:
        raise NotImplementedError()

    train_graphs, test_graphs = separate_data(graphs, args.seed)

    if args.network == "acgnn":
        _model = ACGNN
    else:
        raise NotImplementedError()

    model = _model(
        num_layers=args.num_layers,
        num_mlp_layers=args.num_mlp_layers,
        input_dim=train_graphs[0].node_features.shape[1],
        hidden_dim=args.hidden_dim,
        output_dim=num_classes,
        final_dropout=args.final_dropout,
        learn_eps=args.learn_eps,
        combine_type=args.combine,
        aggregate_type=args.aggregate,
        readout_type=args.readout,
        recursive_weighting=args.recursive_weighting,
        task=args.task_type,
        device=device).to(device)

    criterion = nn.CrossEntropyLoss(reduction='sum')
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)

    if not args.filename == "":
        with open(args.filename, 'w') as f:
            f.write("Epoch\tLoss\tTrain\tTest\n")

    # `epoch` is only for printing purposes
    for epoch in range(1, args.epochs + 1):
        scheduler.step()

        avg_loss = train(
            args=args,
            model=model,
            device=device,
            train_graphs=train_graphs,
            optimizer=optimizer,
            criterion=criterion,
            epoch=epoch)
        acc_train, acc_test = test(
            args, model, device, train_graphs, test_graphs, epoch)

        if not args.filename == "":
            with open(args.filename, 'a') as f:
                f.write(
                    f"{epoch}\t{avg_loss:.4f}\t{acc_train:.4f}\t{acc_test:.4f}\n")


if __name__ == '__main__':
    main()
