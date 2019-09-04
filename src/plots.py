import glob
import os

import matplotlib.pyplot as plt
import pandas as pd

plot_dataset = "T3"  # T1|T2|T3
plot_variable = "loss"  # loss|micro_train|macro_train|micro_test|macro_test

#log_files = glob.glob("./results/V2/*.log")
log_files = glob.glob("./*.log")
dataframes = {}
for file in log_files:
    f = file.split("\\")[-1].split(".")[0].strip()
    dataframes[f] = pd.read_csv(file)

T = {
    "loss": {"agg": {}, "comb": {"mlp": {}}, "read": {}, "h": {}, "batch": {}},
    "micro_train": {"agg": {}, "comb": {"mlp": {}}, "read": {}, "h": {}, "batch": {}},
    "macro_train": {"agg": {}, "comb": {"mlp": {}}, "read": {}, "h": {}, "batch": {}},
    "micro_test": {"agg": {}, "comb": {"mlp": {}}, "read": {}, "h": {}, "batch": {}},
    "macro_test": {"agg": {}, "comb": {"mlp": {}}, "read": {}, "h": {}, "batch": {}}
}

for log, df in dataframes.items():
    dataset, network, agg, read, comb, h, mlp, b = log.split("-")

    if dataset == plot_dataset:
        if agg == "aggS":
            T["loss"]["agg"].setdefault("sum", pd.DataFrame())
            T["loss"]["agg"]["sum"][log] = df.Loss
            T["micro_train"]["agg"].setdefault("sum", pd.DataFrame())
            T["micro_train"]["agg"]["sum"][log] = df.train_micro
            T["macro_train"]["agg"].setdefault("sum", pd.DataFrame())
            T["macro_train"]["agg"]["sum"][log] = df.train_macro
            T["micro_test"]["agg"].setdefault("sum", pd.DataFrame())
            T["micro_test"]["agg"]["sum"][log] = df.test_micro
            T["macro_test"]["agg"].setdefault("sum", pd.DataFrame())
            T["macro_test"]["agg"]["sum"][log] = df.test_macro
        elif agg == "aggA":
            T["loss"]["agg"].setdefault("avg", pd.DataFrame())
            T["loss"]["agg"]["avg"][log] = df.Loss
            T["micro_train"]["agg"].setdefault("avg", pd.DataFrame())
            T["micro_train"]["agg"]["avg"][log] = df.train_micro
            T["macro_train"]["agg"].setdefault("avg", pd.DataFrame())
            T["macro_train"]["agg"]["avg"][log] = df.train_macro
            T["micro_test"]["agg"].setdefault("avg", pd.DataFrame())
            T["micro_test"]["agg"]["avg"][log] = df.test_micro
            T["macro_test"]["agg"].setdefault("avg", pd.DataFrame())
            T["macro_test"]["agg"]["avg"][log] = df.test_macro
        elif agg == "aggM":
            T["loss"]["agg"].setdefault("max", pd.DataFrame())
            T["loss"]["agg"]["max"][log] = df.Loss
            T["micro_train"]["agg"].setdefault("max", pd.DataFrame())
            T["micro_train"]["agg"]["max"][log] = df.train_micro
            T["macro_train"]["agg"].setdefault("max", pd.DataFrame())
            T["macro_train"]["agg"]["max"][log] = df.train_macro
            T["micro_test"]["agg"].setdefault("max", pd.DataFrame())
            T["micro_test"]["agg"]["max"][log] = df.test_micro
            T["macro_test"]["agg"].setdefault("max", pd.DataFrame())
            T["macro_test"]["agg"]["max"][log] = df.test_macro
        else:
            raise ValueError()

        if read == "readS":
            T["loss"]["read"].setdefault("sum", pd.DataFrame())
            T["loss"]["read"]["sum"][log] = df.Loss
            T["micro_train"]["read"].setdefault("sum", pd.DataFrame())
            T["micro_train"]["read"]["sum"][log] = df.train_micro
            T["macro_train"]["read"].setdefault("sum", pd.DataFrame())
            T["macro_train"]["read"]["sum"][log] = df.train_macro
            T["micro_test"]["read"].setdefault("sum", pd.DataFrame())
            T["micro_test"]["read"]["sum"][log] = df.test_micro
            T["macro_test"]["read"].setdefault("sum", pd.DataFrame())
            T["macro_test"]["read"]["sum"][log] = df.test_macro
        elif read == "readA":
            T["loss"]["read"].setdefault("avg", pd.DataFrame())
            T["loss"]["read"]["avg"][log] = df.Loss
            T["micro_train"]["read"].setdefault("avg", pd.DataFrame())
            T["micro_train"]["read"]["avg"][log] = df.train_micro
            T["macro_train"]["read"].setdefault("avg", pd.DataFrame())
            T["macro_train"]["read"]["avg"][log] = df.train_macro
            T["micro_test"]["read"].setdefault("avg", pd.DataFrame())
            T["micro_test"]["read"]["avg"][log] = df.test_micro
            T["macro_test"]["read"].setdefault("avg", pd.DataFrame())
            T["macro_test"]["read"]["avg"][log] = df.test_macro
        elif read == "readM":
            T["loss"]["read"].setdefault("max", pd.DataFrame())
            T["loss"]["read"]["max"][log] = df.Loss
            T["micro_train"]["read"].setdefault("max", pd.DataFrame())
            T["micro_train"]["read"]["max"][log] = df.train_micro
            T["macro_train"]["read"].setdefault("max", pd.DataFrame())
            T["macro_train"]["read"]["max"][log] = df.train_macro
            T["micro_test"]["read"].setdefault("max", pd.DataFrame())
            T["micro_test"]["read"]["max"][log] = df.test_micro
            T["macro_test"]["read"].setdefault("max", pd.DataFrame())
            T["macro_test"]["read"]["max"][log] = df.test_macro
        else:
            raise ValueError()

        if comb == "combT":
            T["loss"]["comb"].setdefault("simple", pd.DataFrame())
            T["loss"]["comb"]["simple"][log] = df.Loss
            T["micro_train"]["comb"].setdefault("simple", pd.DataFrame())
            T["micro_train"]["comb"]["simple"][log] = df.train_micro
            T["macro_train"]["comb"].setdefault("simple", pd.DataFrame())
            T["macro_train"]["comb"]["simple"][log] = df.train_macro
            T["micro_test"]["comb"].setdefault("simple", pd.DataFrame())
            T["micro_test"]["comb"]["simple"][log] = df.test_micro
            T["macro_test"]["comb"].setdefault("simple", pd.DataFrame())
            T["macro_test"]["comb"]["simple"][log] = df.test_macro
        elif comb == "combMLP":

            if mlp == "mlpS":
                T["loss"]["comb"]["mlp"].setdefault("sum", pd.DataFrame())
                T["loss"]["comb"]["mlp"]["sum"][log] = df.Loss
                T["micro_train"]["comb"]["mlp"].setdefault(
                    "sum", pd.DataFrame())
                T["micro_train"]["comb"]["mlp"]["sum"][log] = df.train_micro
                T["macro_train"]["comb"]["mlp"].setdefault(
                    "sum", pd.DataFrame())
                T["macro_train"]["comb"]["mlp"]["sum"][log] = df.train_macro
                T["micro_test"]["comb"]["mlp"].setdefault(
                    "sum", pd.DataFrame())
                T["micro_test"]["comb"]["mlp"]["sum"][log] = df.test_micro
                T["macro_test"]["comb"]["mlp"].setdefault(
                    "sum", pd.DataFrame())
                T["macro_test"]["comb"]["mlp"]["sum"][log] = df.test_macro
            elif mlp == "mlpA":
                T["loss"]["comb"]["mlp"].setdefault("avg", pd.DataFrame())
                T["loss"]["comb"]["mlp"]["avg"][log] = df.Loss
                T["micro_train"]["comb"]["mlp"].setdefault(
                    "avg", pd.DataFrame())
                T["micro_train"]["comb"]["mlp"]["avg"][log] = df.train_micro
                T["macro_train"]["comb"]["mlp"].setdefault(
                    "avg", pd.DataFrame())
                T["macro_train"]["comb"]["mlp"]["avg"][log] = df.train_macro
                T["micro_test"]["comb"]["mlp"].setdefault(
                    "avg", pd.DataFrame())
                T["micro_test"]["comb"]["mlp"]["avg"][log] = df.test_micro
                T["macro_test"]["comb"]["mlp"].setdefault(
                    "avg", pd.DataFrame())
                T["macro_test"]["comb"]["mlp"]["avg"][log] = df.test_macro
            elif mlp == "mlpM":
                T["loss"]["comb"]["mlp"].setdefault("max", pd.DataFrame())
                T["loss"]["comb"]["mlp"]["max"][log] = df.Loss
                T["micro_train"]["comb"]["mlp"].setdefault(
                    "max", pd.DataFrame())
                T["micro_train"]["comb"]["mlp"]["max"][log] = df.train_micro
                T["macro_train"]["comb"]["mlp"].setdefault(
                    "max", pd.DataFrame())
                T["macro_train"]["comb"]["mlp"]["max"][log] = df.train_macro
                T["micro_test"]["comb"]["mlp"].setdefault(
                    "max", pd.DataFrame())
                T["micro_test"]["comb"]["mlp"]["max"][log] = df.test_micro
                T["macro_test"]["comb"]["mlp"].setdefault(
                    "max", pd.DataFrame())
                T["macro_test"]["comb"]["mlp"]["max"][log] = df.test_macro
            elif mlp == "mlpC":
                T["loss"]["comb"]["mlp"].setdefault("cat", pd.DataFrame())
                T["loss"]["comb"]["mlp"]["cat"][log] = df.Loss
                T["micro_train"]["comb"]["mlp"].setdefault(
                    "cat", pd.DataFrame())
                T["micro_train"]["comb"]["mlp"]["cat"][log] = df.train_micro
                T["macro_train"]["comb"]["mlp"].setdefault(
                    "cat", pd.DataFrame())
                T["macro_train"]["comb"]["mlp"]["cat"][log] = df.train_macro
                T["micro_test"]["comb"]["mlp"].setdefault(
                    "cat", pd.DataFrame())
                T["micro_test"]["comb"]["mlp"]["cat"][log] = df.test_micro
                T["macro_test"]["comb"]["mlp"].setdefault(
                    "cat", pd.DataFrame())
                T["macro_test"]["comb"]["mlp"]["cat"][log] = df.test_macro
            else:
                raise ValueError()
        else:
            raise ValueError()

        if h == "h16":
            T["loss"]["h"].setdefault("16", pd.DataFrame())
            T["loss"]["h"]["16"][log] = df.Loss
            T["micro_train"]["h"].setdefault("16", pd.DataFrame())
            T["micro_train"]["h"]["16"][log] = df.train_micro
            T["macro_train"]["h"].setdefault("16", pd.DataFrame())
            T["macro_train"]["h"]["16"][log] = df.train_macro
            T["micro_test"]["h"].setdefault("16", pd.DataFrame())
            T["micro_test"]["h"]["16"][log] = df.test_micro
            T["macro_test"]["h"].setdefault("16", pd.DataFrame())
            T["macro_test"]["h"]["16"][log] = df.test_macro
        elif h == "h32":
            T["loss"]["h"].setdefault("32", pd.DataFrame())
            T["loss"]["h"]["32"][log] = df.Loss
            T["micro_train"]["h"].setdefault("32", pd.DataFrame())
            T["micro_train"]["h"]["32"][log] = df.train_micro
            T["macro_train"]["h"].setdefault("32", pd.DataFrame())
            T["macro_train"]["h"]["32"][log] = df.train_macro
            T["micro_test"]["h"].setdefault("32", pd.DataFrame())
            T["micro_test"]["h"]["32"][log] = df.test_micro
            T["macro_test"]["h"].setdefault("32", pd.DataFrame())
            T["macro_test"]["h"]["32"][log] = df.test_macro
        elif h == "h64":
            T["loss"]["h"].setdefault("64", pd.DataFrame())
            T["loss"]["h"]["64"][log] = df.Loss
            T["micro_train"]["h"].setdefault("64", pd.DataFrame())
            T["micro_train"]["h"]["64"][log] = df.train_micro
            T["macro_train"]["h"].setdefault("64", pd.DataFrame())
            T["macro_train"]["h"]["64"][log] = df.train_macro
            T["micro_test"]["h"].setdefault("64", pd.DataFrame())
            T["micro_test"]["h"]["64"][log] = df.test_micro
            T["macro_test"]["h"].setdefault("64", pd.DataFrame())
            T["macro_test"]["h"]["64"][log] = df.test_macro
        elif h == "h128":
            T["loss"]["h"].setdefault("128", pd.DataFrame())
            T["loss"]["h"]["128"][log] = df.Loss
            T["micro_train"]["h"].setdefault("128", pd.DataFrame())
            T["micro_train"]["h"]["128"][log] = df.train_micro
            T["macro_train"]["h"].setdefault("128", pd.DataFrame())
            T["macro_train"]["h"]["128"][log] = df.train_macro
            T["micro_test"]["h"].setdefault("128", pd.DataFrame())
            T["micro_test"]["h"]["128"][log] = df.test_micro
            T["macro_test"]["h"].setdefault("128", pd.DataFrame())
            T["macro_test"]["h"]["128"][log] = df.test_macro
        else:
            raise ValueError()

        if b == "b16":
            T["loss"]["batch"].setdefault("16", pd.DataFrame())
            T["loss"]["batch"]["16"][log] = df.Loss
            T["micro_train"]["batch"].setdefault("16", pd.DataFrame())
            T["micro_train"]["batch"]["16"][log] = df.train_micro
            T["macro_train"]["batch"].setdefault("16", pd.DataFrame())
            T["macro_train"]["batch"]["16"][log] = df.train_macro
            T["micro_test"]["batch"].setdefault("16", pd.DataFrame())
            T["micro_test"]["batch"]["16"][log] = df.test_micro
            T["macro_test"]["batch"].setdefault("16", pd.DataFrame())
            T["macro_test"]["batch"]["16"][log] = df.test_macro
        elif b == "b32":
            T["loss"]["batch"].setdefault("32", pd.DataFrame())
            T["loss"]["batch"]["32"][log] = df.Loss
            T["micro_train"]["batch"].setdefault("32", pd.DataFrame())
            T["micro_train"]["batch"]["32"][log] = df.train_micro
            T["macro_train"]["batch"].setdefault("32", pd.DataFrame())
            T["macro_train"]["batch"]["32"][log] = df.train_macro
            T["micro_test"]["batch"].setdefault("32", pd.DataFrame())
            T["micro_test"]["batch"]["32"][log] = df.test_micro
            T["macro_test"]["batch"].setdefault("32", pd.DataFrame())
            T["macro_test"]["batch"]["32"][log] = df.test_macro
        elif b == "b64":
            T["loss"]["batch"].setdefault("64", pd.DataFrame())
            T["loss"]["batch"]["64"][log] = df.Loss
            T["micro_train"]["batch"].setdefault("64", pd.DataFrame())
            T["micro_train"]["batch"]["64"][log] = df.train_micro
            T["macro_train"]["batch"].setdefault("64", pd.DataFrame())
            T["macro_train"]["batch"]["64"][log] = df.train_macro
            T["micro_test"]["batch"].setdefault("64", pd.DataFrame())
            T["micro_test"]["batch"]["64"][log] = df.test_micro
            T["macro_test"]["batch"].setdefault("64", pd.DataFrame())
            T["macro_test"]["batch"]["64"][log] = df.test_macro
        else:
            raise ValueError()

for plot_variable, variable_data in T.items():
    for to_plot, inner in variable_data.items():
        for _type, data in inner.items():
            if to_plot == "comb" and _type == "mlp":
                for mlp_agg, other_data in data.items():

                    p = other_data.plot(
                        figsize=(20, 10),
                        title=f"{plot_variable} for fixed {to_plot}={_type} mlp={mlp_agg}")
                    if plot_variable == "loss":
                        p.set_ylim(bottom=0)
                    else:
                        p.set_ylim((0, 1.1))
                    fig = p.get_figure()

                    os.makedirs(f"./plots/{plot_variable}/", exist_ok=True)
                    fig.savefig(
                        f"./plots/{plot_variable}/{to_plot}-{_type}-{mlp_agg}.png")
            else:

                p = data.plot(
                    figsize=(20, 10),
                    title=f"{plot_variable} for fixed {to_plot}={_type}")
                if plot_variable == "loss":
                    p.set_ylim(bottom=0)
                else:
                    p.set_ylim((0, 1.1))
                fig = p.get_figure()

                os.makedirs(f"./plots/{plot_variable}/", exist_ok=True)
                fig.savefig(
                    f"./plots/{plot_variable}/{to_plot}-{_type}.png")

            plt.close(fig)
