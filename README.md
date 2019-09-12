# GNN-logic

Code for de paper .....

## Install

Run `pip install -r requirements.txt` to install all dependencies.

## Generate synthetic graphs

The graphs used in the paper are in the zip-file `graphs.zip`. Just unzip them to `src/data`.

To generate new graphs use the utility function `train_dataset` and `test_dataset` in `src/utils/graphs.py`. The experiments shown in the paper use `force_green=3` for all erdos datasets, and change `force_proportion` for 1, 1.2, 1.5 and 2.

For the line graphs change `name="random"` to `name="line-special"`. This will ignore all other settings and create a line with splitted colors.
The line special dataset will consist of 4 graph types described below:

1. Class 1 (30% of dataset): there are no green in this graph type. The first half of the line is composed by uniformly choosing between betweek the other 4 colors. The second half is composed by having 80% red nodes, the other nodes are chosen uniformly for the left 20%.
2. Class 2 (50% of dataset): the first half of the line have 80% green nodes for the first 20% of it (that means that the first 10% of the whole line have 80% green nodes). The other 30% of the first half have no green nor red nodes, the other colors are chosen uniformly. The second half have no greens and 90% reds. The other colors are chosen uniformly.
3. Class 3 (10% of dataset): no greens in this graph type. Choose uniformly the 4 remaining colors for the whole line.
4. Class 4 (10% of dataset): Choose uniformly all colors for the whole line.

## Replicate results

Run the script in `src/main.py`. The results will be printed to console and logged in `src/logging` if different file for each combination. A single file will collect the last epoch for each experiment for each dataset.

Example: `experiment-1-0-acr-aggM-readA-combT-L2` means that it is the experiment for the second dataset in the list, for the `ACR-GNN`, with aggregate MAX, readout AVERAGE and combine SIMPLE, for 2 layers.
