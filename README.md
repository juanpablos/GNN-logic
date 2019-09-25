# GNN-logic

Code for the paper Logical Expressiveness of Graph Neural Networks.

## Install

Run `pip install -r requirements.txt` to install all dependencies.

## Generate synthetic graphs

The graphs used in the paper are in the zip-file `datasets.zip`. Just unzip them to `src/data/datasets`. The script expects 3 folders inside `src/data/datasets` named `p1`, `p2` and `p3`. These folders contains the datasets for the properties <img src="https://latex.codecogs.com/gif.latex?\alpha_1" title="\alpha_1" />, <img src="https://latex.codecogs.com/gif.latex?\alpha_2" title="\alpha_2" /> and <img src="https://latex.codecogs.com/gif.latex?\alpha_3" title="\alpha_3" /> described in the apendix F on DATA FOR THE EXPERIMENT WITH CLASSIFIER <img src="https://latex.codecogs.com/gif.latex?\alpha_i(x)" title="\alpha_i(x)" /> IN EQUATION (6).

To generate new graphs use the script in `src/utils/graphs.py`. There is a small description of the arguments in `generate_dataset`.

## Replicate synthetic results

Run the script in `src/main.py`. The results will be printed to console and logged in `src/logging/results`. A single file will collect the last epoch for each experiment for each dataset.

Example: `p2-0-0-acrgnn-aggS-readM-combMLP-cl1-L2` means:

* `p2`: the property, in this case <img src="https://latex.codecogs.com/gif.latex?\alpha_2" title="\alpha_2" />.
* `acrgnn`: the network being benchmarked, in this case ACR-GNN.
* `aggS`: the aggregation used, can be S=SUM, M=MAX or A=AVG.
* `readM`: the readout used, can be S=SUM, M=MAX or A=AVG.
* `combMLP`: the combine used, can be SIMPLE or MLP. If SIMPLE a ReLU function is used to apply the non-linearity. If MLP, a 2 layer MLP is used, with batch normalization and ReLU activation function in the hidden layer. No activation function is used over the output.
* `cl1`: the number of layers in the MLP used to weight each component (h, agg, readout), refered as `A`, `B` and `C` in the paper, `V`, `A` and `R` in the code. If 0, no weighting is done. If 1, a Linear unit is used. If N, with N>1, a N layer MLP is used, with batch normalization and ReLU activation function in the hidden layer.
* `L2`: the number of layers of the GNN. 2 in this case.

## Replicate PPI results

Run the script in `src/run_ppi.py`. The results will be printed to console and logged in `src/logging/ppi`. A single file will collect the last epoch for each GNN combination.
A file with no extension will be created with the mean of 10 runs for each configuration and the standard deviation.
